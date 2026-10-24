"""LucidFlux 节点（ComfyUI 原生 DiT 链，精简版：5 个节点）。

工作流（2K 复现）：

    LoadImage → LucidFluxDiffBIR ──(lq / ldr)──┐
                                                ├→ LucidFluxCondition → LucidFluxSampler
    （内部：prompt_embeddings.pt + lucidflux.pth + lucid_connector.pth + SigLIP）
                                                                      │
    EmptySD3LatentImage ──────────────────────────────────────────────┤
    VAELoader(UltraFlux) → LucidFluxVAEPatch → VAEDecode → LucidFluxColorAlign → SaveImage

可选 4K 后段（官方 PiD：只外接 latent，替换掉 VAE 解码）：
    LucidFluxSampler.latent → LucidFluxPiDDecode → SaveImage

与上一版的区别：
  * 文本不再走 CLIP/T5（官方 `prompt_embeddings.pt` 与 CLIP 推理等价，直接读 pt
    可省掉 clip_l + t5xxl 的显存/内存与一次前向）；
  * pth 分支加载 + connector(Redux) 加载 + SigLIP 编码 + LQ/LDR 注入 + guidance
    合并进 `LucidFluxCondition` 一个节点，只留一条 CONDITIONING 输出线。
"""
import logging
import os

import torch

import comfy.clip_vision
import comfy.model_management as mm
import comfy.sample
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers

from . import lucidflux_core as core
from .align_color import wavelet_reconstruction
from .swinir_model import load_swinir

MODELS_SUBDIR = "LucidFlux"
logger = logging.getLogger(__name__)


def _list(substr=None):
    files = folder_paths.get_filename_list(MODELS_SUBDIR)
    if substr:
        files = [f for f in files if substr in f.lower()]
    return files or ["none"]


def _siglip_list():
    files = [f for f in folder_paths.get_filename_list("clip_vision") if "siglip" in f.lower()]
    return files or ["none"]


def _image_to_bchw(image):
    return image.movedim(-1, 1).float()


def _bchw_to_image(x):
    return x.movedim(1, -1).float().clamp(0.0, 1.0)


def _resize_bchw(x, width, height):
    if width <= 0 or height <= 0:
        return x
    if x.shape[-1] == width and x.shape[-2] == height:
        return x
    return comfy.utils.common_upscale(x, width, height, "nearest-exact", "disabled")


# ---------------------------------------------------------------------------
# 重量级对象缓存（避免每次执行重读 3.4GB pth / connector / SigLIP）
# ---------------------------------------------------------------------------

_CACHE = {"control_model": {}, "control": {}, "style": {}, "clip_vision": {}, "pid": {}}


def _get_pid_decoder(ckpt_path, emb_path, dtype):
    from . import pid_core
    key = (os.path.abspath(ckpt_path), os.path.abspath(emb_path), str(dtype))
    dec = _CACHE["pid"].get(key)
    if dec is None:
        dec = pid_core.PidDecoder(ckpt_path, emb_path, dtype=dtype)
        _CACHE["pid"][key] = dec
    return dec


def _get_control(path, dtype):
    key = (os.path.abspath(path), str(dtype))
    ctrl = _CACHE["control"].get(key)
    if ctrl is None:
        model = _CACHE["control_model"].get(key)
        if model is None:
            model = core.load_control_model(path, dtype=dtype)
            _CACHE["control_model"][key] = model
        ctrl = core.LucidFluxControl(model)
        _CACHE["control"][key] = ctrl
    return ctrl


def _get_style(path):
    key = os.path.abspath(path)
    style = _CACHE["style"].get(key)
    if style is None:
        style = comfy.sd.load_style_model(path)
        _CACHE["style"][key] = style
    return style


def _get_clip_vision(path):
    key = os.path.abspath(path)
    cv = _CACHE["clip_vision"].get(key)
    if cv is None:
        cv = comfy.clip_vision.load(path)
        _CACHE["clip_vision"][key] = cv
    return cv


# ---------------------------------------------------------------------------


class LucidFluxCondition:
    """一个节点搞定「文本 embedding + Redux 图像 embedding + LucidFlux 双条件分支 + guidance」。

    - 文本：直接读官方 `prompt_embeddings.pt`（txt 1×512×4096 + vec 1×768），不加载 CLIP/T5；
    - 图像先验：SigLIP2-so400m-patch16-512(ldr) → `lucid_connector.pth`(ReduxImageEncoder)
      → 1024×4096 token 追加到文本 token 之后（与官方一致）；
    - 条件分支：`lucidflux.pth` 的 condition_lq/ldr + modulation，作为 comfy 原生 control
      注入 flux double block（LQ/LDR 尺寸按 latent×8 自动对齐）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_file": (_list("lucid"), {"tooltip": "lucidflux.pth（双条件分支）"}),
                "embedding": (_list("prompt"), {"tooltip": "prompt_embeddings.pt（官方预计算文本 embedding）"}),
                "connector": (_list("connector"), {"tooltip": "lucid_connector.pth（SigLIP→4096 的 Redux 映射）"}),
                "siglip": (_siglip_list(), {"tooltip": "models/clip_vision 下的 SigLIP2-512；接了 clip_vision 输入则忽略此项"}),
                "lq": ("IMAGE", {"tooltip": "低质输入（DiffBIR 的 lq，尺寸 = latent×8）"}),
                "ldr": ("IMAGE", {"tooltip": "DiffBIR 复原结果（SwinIR 输出）"}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "dtype": (["bf16", "fp32"], {"default": "bf16"}),
            },
            "optional": {
                "clip_vision": ("CLIP_VISION", {"tooltip": "可选：外部已加载的 SigLIP（省一次加载）"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "build"
    CATEGORY = "LucidFlux"
    DESCRIPTION = ("文本走官方 prompt_embeddings.pt（不加载 CLIP/T5）；"
                   "SigLIP→Redux、双条件分支、guidance 全部在这里完成，直接接 Sampler 的 positive。")

    def build(self, control_file, embedding, connector, siglip, lq, ldr, guidance, dtype, clip_vision=None):
        dt = torch.bfloat16 if dtype == "bf16" else torch.float32

        # ---- 1. 文本 embedding（官方 pt，与 CLIP 推理等价）----
        emb_path = folder_paths.get_full_path(MODELS_SUBDIR, embedding)
        if emb_path is None:
            raise ValueError(f"找不到 {embedding}（应放在 models/{MODELS_SUBDIR}/）")
        data = torch.load(emb_path, map_location="cpu", weights_only=False)
        txt = data["txt"].float()
        vec = data["vec"].float()
        if txt.ndim == 2:
            txt = txt.unsqueeze(0)
        if vec.ndim == 1:
            vec = vec.unsqueeze(0)
        txt = txt.detach()

        # ---- 2. SigLIP → Redux，拼到文本 token 之后 ----
        cv = clip_vision
        if cv is None:
            cv_path = folder_paths.get_full_path("clip_vision", siglip)
            if cv_path is None:
                raise ValueError(f"找不到 SigLIP 权重: {siglip}")
            cv = _get_clip_vision(cv_path)
        cv_out = cv.encode_image(ldr, crop=False)          # 官方是拉伸 resize（不裁剪）
        if clip_vision is None:
            try:  # 编码完就卸下，不占着显存
                mm.unload_model_and_clones(cv.patcher)
                mm.soft_empty_cache()
            except Exception:
                pass
        style = _get_style(folder_paths.get_full_path(MODELS_SUBDIR, connector))
        image_embeds = style.get_cond(cv_out).flatten(0, 1).unsqueeze(0)
        cond_txt = torch.cat((txt, image_embeds.to(txt.dtype)), dim=1)
        cond = [[cond_txt, {"pooled_output": vec}]]

        # ---- 3. 双条件分支 ----
        ctrl = _get_control(folder_paths.get_full_path(MODELS_SUBDIR, control_file), dt)

        lq_bchw = _image_to_bchw(lq)
        ldr_bchw = _image_to_bchw(ldr)
        if lq_bchw.shape[0] != ldr_bchw.shape[0]:
            ldr_bchw = comfy.utils.repeat_to_batch_size(ldr_bchw, lq_bchw.shape[0])
        applied = ctrl.copy_with(lq_bchw * 2.0 - 1.0, ldr_bchw * 2.0 - 1.0, txt)

        out = []
        for t in cond:
            d = t[1].copy()
            d["control"] = applied
            d["control_apply_to_uncond"] = False
            out.append([t[0], d])
        out = node_helpers.conditioning_set_values(out, {"guidance": float(guidance)})
        return (out,)


class LucidFluxDiffBIR:
    """DiffBIR 先验：SwinIR 通用复原（unshuffle 版本），在 latent 对应像素尺度上做一次“去退化”。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "swinir": (_list("swinir"),),
                "width": ("INT", {"default": 1024, "min": 0, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 0, "max": 8192, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("ldr", "lq")
    FUNCTION = "run"
    CATEGORY = "LucidFlux"
    DESCRIPTION = ("width/height = 目标图的一半（2K 填 1024）；0 表示保持原尺寸。"
                   "lq = 缩放后的输入，ldr = SwinIR 复原结果，两者直接接 LucidFluxCondition。")

    def run(self, image, swinir, width, height):
        lq = _resize_bchw(_image_to_bchw(image), width, height)
        # 官方：condition 先 bf16 往返再 *2-1 -> ci_01
        cond = (lq.to(torch.bfloat16) * 2.0 - 1.0)
        ci01 = torch.clamp((cond.float() + 1.0) / 2.0, 0.0, 1.0)

        model = load_swinir(folder_paths.get_full_path(MODELS_SUBDIR, swinir))
        device = mm.get_torch_device()
        model.to(device)
        with torch.no_grad():
            out = model(ci01.to(device)).float().clamp(0.0, 1.0)
        model.to("cpu")
        mm.soft_empty_cache()

        ldr = _bchw_to_image(out)
        lq_img = _bchw_to_image(lq)
        if image.shape[0] == 1 and lq_img.shape[0] != image.shape[0]:
            lq_img = lq_img.repeat(image.shape[0], 1, 1, 1)
        return (ldr, lq_img)


class LucidFluxSampler:
    """官方 get_schedule 的精确 sigma 表 + comfy 原生 euler。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "steps": ("INT", {"default": 28, "min": 1, "max": 200}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "shift": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "noise": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "LucidFlux"
    DESCRIPTION = ("cfg 恒为 1.0（flux-dev 是 guidance 蒸馏模型，guidance 已在 conditioning 里）。"
                   "noise 仅在需要与官方逐位对拍时接入手工噪声。")

    def sample(self, model, positive, latent, steps, seed, cfg, shift, noise=None):
        latent_image = latent["samples"]
        sigmas = core.build_sigmas(steps, latent_image.shape[-2], latent_image.shape[-1], shift=shift)
        if noise is None:
            noise_samples = comfy.sample.prepare_noise(latent_image, seed)
        else:
            noise_samples = noise["samples"]
        samples = core.sample_lucidflux(model, positive, latent_image, sigmas,
                                        noise=noise_samples, seed=seed, cfg=cfg)
        return ({"samples": samples},)


class LucidFluxVAEPatch:
    """UltraFlux VAE 的 F16(/16) 语义补丁（comfy 默认按 /8 装载）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",)}}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "patch"
    CATEGORY = "LucidFlux"
    DESCRIPTION = "必接：UltraFlux.safetensors 在官方是 F16 VAE。不接的话同一 latent 只解出一半分辨率。"

    def patch(self, vae):
        return (core.patch_vae_f16(vae, stride=2),)


class LucidFluxColorAlign:
    """VAE 解码后的颜色对齐（官方 wavelet_reconstruction）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "align"
    CATEGORY = "LucidFlux"
    DESCRIPTION = "把 reference（一般接 DiffBIR 的 ldr）的低频颜色/亮度搬回 image；strength=0 等于不处理。"

    def align(self, image, reference, strength):
        if strength <= 0.0:
            return (image,)
        x = _image_to_bchw(image)
        ref = _image_to_bchw(reference)
        if ref.shape[-2:] != x.shape[-2:]:
            ref = comfy.utils.common_upscale(ref, x.shape[-1], x.shape[-2], "bilinear", "disabled")
        if ref.shape[0] != x.shape[0]:
            ref = comfy.utils.repeat_to_batch_size(ref, x.shape[0])
        device = mm.get_torch_device()
        out = []
        with torch.no_grad():
            for i in range(x.shape[0]):
                fixed = wavelet_reconstruction(x[i:i + 1].to(device), ref[i:i + 1].to(device)).cpu()
                if strength < 1.0:
                    fixed = x[i:i + 1] * (1.0 - strength) + fixed * strength
                out.append(fixed)
        return (_bchw_to_image(torch.cat(out, dim=0)),)


class LucidFluxPiDDecode:
    """PiD 4-step 蒸馏 student 解码：flux latent → 4× 图像（官方 `2kto4k` 路径）。

    与官方一致，PiD **只吃采样后的 latent**，不需要 VAE 编码/解码：
      * 目标分辨率 = latent × 8 × 4（latent 128 → 4096²）
      * 文本用 `gemma_prompt_embedding.pt` 的预计算 caption_embs，不加载 Gemma-2
      * 采样 4 步（t=[0.999,0.866,0.634,0.342,0]），SDE 中间步加噪
    """

    @classmethod
    def INPUT_TYPES(cls):
        pid_files = [f for f in folder_paths.get_filename_list("diffusion_models")
                     if "pid" in f.lower()]
        pid_files += [f for f in folder_paths.get_filename_list(MODELS_SUBDIR)
                      if "pid" in f.lower() and f.lower().endswith((".pth", ".pt", ".safetensors"))]
        # "auto" = 自动挑一个可用权重；"none" 只作旧工作流兼容别名（等同于 auto）
        pid_files = ["auto"] + pid_files
        return {
            "required": {
                "latent": ("LATENT",),
                "pid_model": (pid_files + ["none"],
                              {"tooltip": "auto = 自动选 diffusion_models/LucidFlux 里文件名含 pid 的权重；"
                                          "也可手动指定。none 仅为旧工作流兼容（等同 auto）"}),
                "caption_embeddings": (_list("gemma"), {"tooltip": "gemma_prompt_embedding.pt（预计算 caption_embs，300×2304）"}),
                "num_steps": ("INT", {"default": 4, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 123456789, "min": 0, "max": 0xffffffffffffffff}),
                "degrade_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dtype": (["bf16", "fp32"], {"default": "bf16"}),
                "streaming_prefetch": ("INT", {"default": 0, "min": 0, "max": 4,
                                               "tooltip": "0=整模型上卡；1~2=层流式（省显存，官方默认 1）"}),
            },
            "optional": {
                "reference": ("IMAGE", {"tooltip": "可选：DiffBIR 的 ldr，用于颜色对齐（官方 align_color）"}),
                "color_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "unload_models": ("BOOLEAN", {"default": True,
                                              "tooltip": "解码前先卸载 comfy 已加载的模型（flux/分支/VAE）腾显存。"
                                                         "PiD 不归 comfy 管理，不卸就会 OOM；"
                                                         "若同一 prompt 后面还要用 flux，设为 False"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "LucidFlux"
    DESCRIPTION = ("官方 PiD 2kto4k：latent(128) → 4096²。需要 PiD ckpt 放在 "
                   "models/diffusion_models 或 models/LucidFlux（文件名含 pid）。\n"
                   "显存：解码前会卸载 comfy 已加载模型（unload_models=True）；"
                   "4096² 本身峰值约 18GB，12GB 卡会走 Windows 共享内存（可出图但慢），"
                   "2048²(latent 64) 仅需约 5GB。")

    def decode(self, latent, pid_model, caption_embeddings, num_steps, seed, degrade_sigma,
               dtype, streaming_prefetch, reference=None, color_strength=1.0,
               unload_models=True):
        from . import pid_core as _pid

        if pid_model in ("auto", "none"):
            cands = [f for f in folder_paths.get_filename_list("diffusion_models") if "pid" in f.lower()]
            cands += [f for f in folder_paths.get_filename_list(MODELS_SUBDIR) if "pid" in f.lower()]
            if not cands:
                raise ValueError("没有找到 PiD 权重：请把 PiD_*.pth 放到 models/diffusion_models/ "
                                 "或 models/LucidFlux/（文件名需含 pid）")
            cands.sort(key=lambda f: (0 if "2kto4k" in f.lower() else 1, f))
            pid_model = cands[0]
            logger.info("[PiD] auto-selected weights: %s", pid_model)
        path = folder_paths.get_full_path("diffusion_models", pid_model)
        if path is None:
            path = folder_paths.get_full_path(MODELS_SUBDIR, pid_model)
        if path is None:
            raise ValueError(f"找不到 PiD 权重: {pid_model}")
        emb_path = folder_paths.get_full_path(MODELS_SUBDIR, caption_embeddings)
        if emb_path is None:
            raise ValueError(f"找不到 PiD caption embeddings: {caption_embeddings}")

        dt = torch.bfloat16 if dtype == "bf16" else torch.float32
        dec = _get_pid_decoder(path, emb_path, dt)

        device = mm.get_torch_device()
        x = latent["samples"]
        try:
            out, (th, tw) = dec.decode(x, seed=seed, num_steps=num_steps,
                                       degrade_sigma=degrade_sigma,
                                       streaming_prefetch=(None if streaming_prefetch <= 0
                                                           else streaming_prefetch),
                                       release_comfy=bool(unload_models))
        finally:
            mm.soft_empty_cache()

        img = out.float().mul(0.5).add(0.5).clamp(0, 1).movedim(1, -1)   # BCHW[-1,1] -> BHWC[0,1]
        if reference is not None and color_strength > 0:
            ref = _image_to_bchw(reference)
            aligned = []
            for i in range(img.shape[0]):
                r = ref[i:i + 1] if ref.shape[0] > 1 else ref
                aligned.append(_pid.align_pid_color(img[i:i + 1].movedim(-1, 1).to(device),
                                                    r.to(device), color_strength))
            img = torch.cat(aligned, dim=0).movedim(1, -1).cpu()
        return (img,)


NODE_CLASS_MAPPINGS = {
    "LucidFluxCondition": LucidFluxCondition,
    "LucidFluxDiffBIR": LucidFluxDiffBIR,
    "LucidFluxSampler": LucidFluxSampler,
    "LucidFluxVAEPatch": LucidFluxVAEPatch,
    "LucidFluxColorAlign": LucidFluxColorAlign,
    "LucidFluxPiDDecode": LucidFluxPiDDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LucidFluxCondition": "LucidFlux Condition (text+redux+control)",
    "LucidFluxDiffBIR": "LucidFlux DiffBIR (SwinIR)",
    "LucidFluxSampler": "LucidFlux Sampler",
    "LucidFluxVAEPatch": "LucidFlux VAE Patch (F16)",
    "LucidFluxColorAlign": "LucidFlux Color Align",
    "LucidFluxPiDDecode": "LucidFlux PiD Decode (latent -> 4K)",
}
