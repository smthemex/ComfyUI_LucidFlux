"""LucidFlux -> ComfyUI 原生 DiT 推理栈的核心层。

上游：https://github.com/W2GenAI-Lab/LucidFlux
结构（逐字核对 checkpoint 后的结论）：
    flux-dev DiT（与 stock flux 完全同构）
  + dual condition branch：两个 ControlNet 式分支（LQ / LDR），
    每个分支 = img_in/time_in/vector_in/guidance_in/txt_in
              + 2 个 DoubleStreamBlock（controlnet_depth=2）
              + 2 个控制输出 Linear（controlnet_blocks）
              + input_hint_block（3 次 stride=2 卷积 = /8，末尾 zero conv）
              + pos_embed_input
  + modulation_lq / modulation_ldr：按 control_index（0..18）对分支输出做
    逐 block 的 (shift, scale) 调制（time embed + control-index embed）。

每个去噪步：residual[k] = mod_lq(out_lq[k%2], t, k) + mod_ldr(out_ldr[k%2], t, k)，
在 double block k 之后以 `img = img + residual[k]` 注入 img 流。

comfy 原生 flux 的 control 注入点（comfy/ldm/flux/model.py）就是
`img[:, :add.shape[1]] += add`，语义逐位一致——因此本插件只需要：
  1. 用 comfy 原生模块（comfy.ldm.flux.layers）拼出分支，key 与 checkpoint 1:1；
  2. 把 19 个 residual 包成 {"input": [...], "output": []} 交给采样器。
主模型加载/采样/VAE 全部走 comfy 原生。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import comfy.controlnet
import comfy.ldm.flux.layers as flux_layers
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.samplers
import comfy.utils

ops = comfy.ops.manual_cast

HIDDEN = 3072
NUM_HEADS = 24
AXES_DIM = (16, 56, 56)
THETA = 10000
IN_CHANNELS = 64
CONTEXT_DIM = 4096
VEC_DIM = 768
CTRL_DEPTH = 2          # controlnet_depth：每个分支的 double block 数
MAIN_DEPTH = 19         # 主 DiT 的 double block 数（= residual 个数）
PATCH = 2

STATE_KEYS = ("condition_lq", "condition_ldr", "modulation_lq", "modulation_ldr")


# ---------------------------------------------------------------------------
# 分支模块：key 与 checkpoint 1:1（load_state_dict(strict=True) 自校验）
# ---------------------------------------------------------------------------

class _TimeEmbedding(nn.Module):
    """key 名对齐 checkpoint：linear_1 / linear_2（SiLU 无参数）。"""

    def __init__(self, in_dim=256, out_dim=HIDDEN, dtype=None, device=None):
        super().__init__()
        self.linear_1 = ops.Linear(in_dim, out_dim, bias=True, dtype=dtype, device=device)
        self.act = nn.SiLU()
        self.linear_2 = ops.Linear(out_dim, out_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


class LucidFluxModulation(nn.Module):
    """官方 Modulation：norm(x) * (1 + scale) + shift，scale/shift 由 t 与 control_index 决定。"""

    def __init__(self, dim=HIDDEN, dtype=None, device=None):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = ops.Linear(dim, 2 * dim, bias=True, dtype=dtype, device=device)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.timestep_embedder = _TimeEmbedding(256, dim, dtype=dtype, device=device)
        self.control_index_embedder = _TimeEmbedding(256, dim, dtype=dtype, device=device)

    def forward(self, x, sigma, control_index):
        t_emb = flux_layers.timestep_embedding(sigma, 256).to(dtype=x.dtype)      # sigma*1000
        idx = torch.full_like(sigma, float(control_index))
        # 官方：control_index 不做 *1000（Timesteps.scale=1）
        idx_emb = flux_layers.timestep_embedding(idx, 256, time_factor=1.0).to(dtype=x.dtype)
        emb = self.linear(self.silu(self.timestep_embedder(t_emb) + self.control_index_embedder(idx_emb)))
        shift, scale = emb.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


def _hint_block(dtype=None, device=None):
    conv = lambda cin, cout, **kw: ops.Conv2d(cin, cout, 3, padding=1, dtype=dtype, device=device, **kw)
    return nn.Sequential(
        conv(3, 16), nn.SiLU(),
        conv(16, 16), nn.SiLU(),
        conv(16, 16, stride=2), nn.SiLU(),
        conv(16, 16), nn.SiLU(),
        conv(16, 16, stride=2), nn.SiLU(),
        conv(16, 16), nn.SiLU(),
        conv(16, 16, stride=2), nn.SiLU(),
        conv(16, 16),
    )


class LucidFluxControlBranch(nn.Module):
    """官方 SingleConditionBranch（controlnet_depth=2）。"""

    def __init__(self, dtype=None, device=None):
        super().__init__()
        self.hidden_size = HIDDEN
        self.num_heads = NUM_HEADS
        self.pe_embedder = flux_layers.EmbedND(dim=HIDDEN // NUM_HEADS, theta=THETA, axes_dim=list(AXES_DIM))
        self.img_in = ops.Linear(IN_CHANNELS, HIDDEN, bias=True, dtype=dtype, device=device)
        self.time_in = flux_layers.MLPEmbedder(256, HIDDEN, dtype=dtype, device=device, operations=ops)
        self.vector_in = flux_layers.MLPEmbedder(VEC_DIM, HIDDEN, dtype=dtype, device=device, operations=ops)
        self.guidance_in = flux_layers.MLPEmbedder(256, HIDDEN, dtype=dtype, device=device, operations=ops)
        self.txt_in = ops.Linear(CONTEXT_DIM, HIDDEN, bias=True, dtype=dtype, device=device)
        self.double_blocks = nn.ModuleList([
            flux_layers.DoubleStreamBlock(HIDDEN, NUM_HEADS, mlp_ratio=4.0, qkv_bias=True,
                                          dtype=dtype, device=device, operations=ops)
            for _ in range(CTRL_DEPTH)
        ])
        self.controlnet_blocks = nn.ModuleList([
            ops.Linear(HIDDEN, HIDDEN, bias=True, dtype=dtype, device=device) for _ in range(CTRL_DEPTH)
        ])
        self.pos_embed_input = ops.Linear(IN_CHANNELS, HIDDEN, bias=True, dtype=dtype, device=device)
        self.input_hint_block = _hint_block(dtype=dtype, device=device)

    def forward(self, img, img_ids, cond_img, txt, txt_ids, y, sigma, guidance):
        img = self.img_in(img)

        hint = self.input_hint_block(cond_img)
        hint = rearrange(hint, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=PATCH, pw=PATCH)
        img = img + self.pos_embed_input(hint)

        vec = self.time_in(flux_layers.timestep_embedding(sigma, 256).to(img.dtype))
        vec = vec + self.guidance_in(flux_layers.timestep_embedding(guidance, 256).to(img.dtype))
        vec = vec + self.vector_in(y)

        txt = self.txt_in(txt)
        pe = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))

        res = []
        # 注意：comfy 的 DoubleStreamBlock 是**原地** `img += ...`，
        # 若先收集 img 再统一投影，前面收集的引用会被后续 block 改写（别名 bug）。
        # 所以在每个 block 之后立刻做 controlnet 投影（投影产生新张量）。
        for cb, block in zip(self.controlnet_blocks, self.double_blocks):
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, transformer_options={})
            res.append(cb(img))
        return res


class LucidFluxControlModel(nn.Module):
    """官方 DualConditionBranch + 两个 Modulation 的等价实现。"""

    def __init__(self, dtype=None, device=None):
        super().__init__()
        self.lq = LucidFluxControlBranch(dtype=dtype, device=device)
        self.ldr = LucidFluxControlBranch(dtype=dtype, device=device)
        self.modulation_lq = LucidFluxModulation(dtype=dtype, device=device)
        self.modulation_ldr = LucidFluxModulation(dtype=dtype, device=device)

    def forward(self, img, img_ids, cond_lq, cond_ldr, txt, txt_ids, vec, sigma, guidance):
        out_lq = self.lq(img, img_ids, cond_lq, txt, txt_ids, vec, sigma, guidance)
        out_ldr = self.ldr(img, img_ids, cond_ldr, txt, txt_ids, vec, sigma, guidance)

        out = []
        for k in range(MAIN_DEPTH):
            j = k % CTRL_DEPTH
            out.append(self.modulation_lq(out_lq[j], sigma, k) + self.modulation_ldr(out_ldr[j], sigma, k))
        return out


# ---------------------------------------------------------------------------
# 装载
# ---------------------------------------------------------------------------

def load_control_model(path, dtype=torch.bfloat16, verbose=True):
    """读取 lucidflux.pth 的 4 个子 dict，严格装载到 comfy 原生模块。

    官方 flux 的 QKNorm 参数名是 `*.norm.query_norm.scale`，而 comfy 的
    `operations.RMSNorm`（torch.nn.RMSNorm）叫 `.weight` —— 与 comfy 自己加载
    flux1-dev 时的 `supported_models` 改名规则一致，这里做同样的 .scale -> .weight。
    """
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    missing = [k for k in STATE_KEYS if k not in sd]
    if missing:
        raise ValueError(f"{path} 缺少 LucidFlux 权重段: {missing}（文件内只有 {list(sd.keys())}）")

    model = LucidFluxControlModel(dtype=dtype)
    report = {}
    for name, module in (("condition_lq", model.lq), ("condition_ldr", model.ldr),
                         ("modulation_lq", model.modulation_lq), ("modulation_ldr", model.modulation_ldr)):
        state = {k[:-6] + ".weight" if k.endswith(".scale") else k: v for k, v in sd[name].items()}
        # assign=True：直接接管 mmap 张量，省一份 3GB 峰值拷贝
        result = module.load_state_dict(state, strict=True, assign=True)
        report[name] = (len(result.missing_keys), len(result.unexpected_keys))

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    if dtype is not None:
        model = model.to(dtype)
    del sd
    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"[LucidFlux] control model loaded: {n/1e6:.1f}M params, dtype={dtype}, strict-ok={report}")
    return model


# ---------------------------------------------------------------------------
# Control 对象（comfy ControlBase 子类，交给原生 sampler 调 get_control）
# ---------------------------------------------------------------------------

class LucidFluxControl(comfy.controlnet.ControlBase):
    """把 LucidFlux 的双分支包成 comfy 的 control 对象。

    原生采样器每步调用 get_control()，comfy flux 把返回的
    {"input": [19 x (B, N_img, 3072)]} 逐 block 加进 img 流。
    """

    def __init__(self, control_model, load_device=None):
        super().__init__()
        self.control_model = control_model
        self.load_device = load_device or comfy.model_management.get_torch_device()
        self.patcher = comfy.model_patcher.ModelPatcher(
            control_model,
            load_device=self.load_device,
            offload_device=comfy.model_management.unet_offload_device(),
        )
        self.cond_lq = None
        self.cond_ldr = None
        self.txt = None
        self.offload_device = comfy.model_management.unet_offload_device()

    # 复制体（共享权重与 patcher，只换条件图），语义同 ControlNetApply 的 copy()
    def copy_with(self, cond_lq, cond_ldr, txt):
        c = LucidFluxControl(self.control_model, self.load_device)
        c.patcher = self.patcher
        c.cond_lq = cond_lq
        c.cond_ldr = cond_ldr
        c.txt = txt
        return c

    def get_models(self):
        out = super().get_models()
        out.append(self.patcher)
        return out

    def inference_memory_requirements(self, dtype):
        return comfy.model_management.module_size(self.control_model) + super().inference_memory_requirements(dtype)

    def cleanup(self):
        # 只清 ControlBase 的临时状态；条件图必须保留 —— comfy 在采样结束后会调用
        # cleanup()，若这里把 cond_lq/txt 置空，同一份 conditioning 第二次采样（换 seed）
        # 会静默丢掉 control 引导。
        super().cleanup()

    # ---- 核心：每步算 residual -------------------------------------------
    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        if self.cond_lq is None or self.cond_ldr is None or self.txt is None:
            return None

        sigma_t = transformer_options.get("sigmas", None)
        if sigma_t is None:
            sigma_value = float(t.detach().float().flatten()[0])
        else:
            sigma_value = float(sigma_t.detach().float().flatten()[0])

        comfy.model_management.load_models_gpu(
            [self.patcher], memory_required=self.inference_memory_requirements(torch.bfloat16))

        device = x_noisy.device
        dtype = self._model_dtype()
        b, c, h, w = x_noisy.shape

        img = rearrange(x_noisy.to(dtype), "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=PATCH, pw=PATCH)
        h_lat, w_lat = h // PATCH, w // PATCH
        img_ids = torch.zeros(h_lat, w_lat, 3, device=device, dtype=torch.float32)
        img_ids[..., 1] += torch.arange(h_lat, device=device, dtype=torch.float32)
        img_ids[..., 2] += torch.arange(w_lat, device=device, dtype=torch.float32)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=b).to(dtype=dtype)

        txt = self.txt.to(device=device, dtype=dtype)
        txt_ids = torch.zeros(b, txt.shape[1], 3, device=device, dtype=dtype)

        vec = cond.get("pooled_output", None)
        if vec is None:
            vec = torch.zeros(b, VEC_DIM, device=device, dtype=dtype)
        else:
            vec = vec.to(device=device, dtype=dtype)
            if vec.shape[0] != b:
                vec = comfy.utils.repeat_to_batch_size(vec, b)

        g = cond.get("guidance", None)
        gv = float(g.detach().float().flatten()[0]) if g is not None else 4.0
        sigma = torch.full((b,), sigma_value, device=device, dtype=dtype)
        guidance = torch.full((b,), gv, device=device, dtype=dtype)

        # 分支条件图语义：像素边长 = latent * 8（官方 cond 图正是 latent*8）
        lq = self._hint(self.cond_lq, b, h, w, device, dtype)
        ldr = self._hint(self.cond_ldr, b, h, w, device, dtype)

        with torch.no_grad():
            out = self.control_model(img, img_ids, lq, ldr, txt, txt_ids, vec, sigma, guidance)
        return {"input": [x.to(dtype) for x in out], "output": []}

    def _model_dtype(self):
        try:
            return self.patcher.model_dtype()
        except Exception:
            return torch.bfloat16

    @staticmethod
    def _hint(img, batch, h_lat, w_lat, device, dtype):
        img = img.to(device=device, dtype=torch.float32)
        target_h, target_w = h_lat * 8, w_lat * 8
        if img.shape[-2] != target_h or img.shape[-1] != target_w:
            if img.shape[0] == 1 and batch > 1:
                img = img.repeat(batch, 1, 1, 1)
            img = comfy.utils.common_upscale(img, target_w, target_h, "bilinear", "disabled")
        if img.shape[0] != batch:
            img = comfy.utils.repeat_to_batch_size(img, batch)
        return img.to(dtype)


# ---------------------------------------------------------------------------
# 采样调度：官方 get_schedule 的逐字复刻
# ---------------------------------------------------------------------------

def _lin_function(x1=256.0, y1=0.5, x2=4096.0, y2=1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def _time_shift(mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def official_image_seq_len(latent_h, latent_w, infer_2k=True):
    """官方 image_seq_len 的等价式。

    2k: (w//8)*(h//8)//(32*32)，其中 w = latent*16 → 等价 latent_h*latent_w/256
    1k: (w//8)*(h//8)//(16*16)，其中 w = latent*8  → 等价 latent_h*latent_w/256
    两者同式，实测官方脚本 1024/2048 都得到 64。
    """
    return max(1, int(latent_h) * int(latent_w) // 256)


def build_sigmas(num_steps, latent_h, latent_w, shift=True):
    """返回 comfy euler 用的 sigma 表（= 官方 timesteps，含末尾 0）。"""
    seq = official_image_seq_len(latent_h, latent_w)
    ts = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)
    if shift:
        mu = _lin_function()(seq)
        exp_mu = math.exp(mu)
        ts = exp_mu / (exp_mu + (1.0 / ts - 1.0))
        ts = torch.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
    return ts.contiguous()


def sample_lucidflux(model, cond, latent_image, sigmas, noise=None, seed=0,
                     cfg=1.0, disable_pbar=False):
    sampler = comfy.samplers.sampler_object("euler")
    return comfy.sample.sample_custom(
        model, noise, cfg=cfg, sampler=sampler, sigmas=sigmas,
        positive=cond, negative=[], latent_image=latent_image,
        disable_pbar=disable_pbar, seed=seed)


# ---------------------------------------------------------------------------
# VAE：UltraFlux 的对称 F16(/16) 语义（comfy 默认按 stock flux 的 /8 装载）
# ---------------------------------------------------------------------------

class _F16ConvOut(nn.Module):
    """官方 Decoder 尾部 conv_act -> nearest x2 -> conv_out。"""

    def __init__(self, conv, stride=2):
        super().__init__()
        self.conv = conv
        self.stride = stride

    def forward(self, x):
        if self.stride <= 1:
            return self.conv(x)
        return self.conv(F.interpolate(x, scale_factor=self.stride, mode="nearest"))


def patch_vae_f16(vae, stride=2):
    fs = getattr(vae, "first_stage_model", None)
    if fs is None or not hasattr(fs, "decoder"):
        raise ValueError("这不是 comfy 的 AutoencodingEngine VAE，无法打 F16 补丁")
    dec = fs.decoder
    if getattr(dec, "_lucidflux_f16", False):
        return vae
    try:
        fs.encoder.conv_in.stride = (stride, stride)
    except Exception:
        pass
    dec.conv_out = _F16ConvOut(dec.conv_out, stride=stride)
    dec._lucidflux_f16 = True
    try:
        vae.downscale_ratio = 8 * stride
        vae.upscale_ratio = 8 * stride
    except Exception:
        pass
    return vae
