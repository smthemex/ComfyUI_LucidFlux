"""PiD（PixelDiT 4-step 蒸馏 student）解码器：flux latent → 高清图像。

对应官方 `src/pid/decoder.py::PiDFluxDecoder`，但去掉了 imaginaire / hydra / omegaconf /
transformers / accelerate 依赖：
  * 网络：`lucidflux_pid`（从官方 pid/_src/networks 逐字移植，只改 import）
  * 文本：直接吃 `gemma_prompt_embedding.pt` 里预计算的 caption_embs(1,300,2304)，
          Gemma-2 文本编码器完全不加载
  * VAE：**不需要**——PiD 只吃 flux latent（16ch），不编码也不解码图像；
         官方 2K 流程也是把 LucidFlux 的 latent 直接喂进去

关键语义（逐条对齐官方）：
  * 目标分辨率 = latent × 8 × sr_scale（4）→ latent 128 出 4096²
  * LQ_latent 必须是 diffusers 采样空间 latent：
        X = (z_raw − 0.1159) × 0.3611   （= comfy 的 latent_format.process_in）
    comfy 采样器返回的是 z_raw，故这里做一次转换
  * 采样：t_list = [0.999, 0.866, 0.634, 0.342, 0]（4 步），t×timescale(1000) 喂网络，
         velocity 预测 → x0 = x − t·v，中间步做 SDE 加噪，最后一步直接取 x0
  * 去噪 sigma：`degrade_sigma`（0 表示输入干净）
"""
import logging
import os

import torch

import comfy.model_management as mm

logger = logging.getLogger(__name__)

# ---- 官方 model_loader.py 里的 PidNet 构造参数（逐字）----
PID_NET_KWARGS = dict(
    in_channels=3,
    num_groups=24,
    hidden_size=1536,
    pixel_hidden_size=16,
    pixel_attn_hidden_size=1152,
    pixel_num_groups=16,
    patch_depth=14,
    pixel_depth=2,
    num_text_blocks=4,
    patch_size=16,
    txt_embed_dim=2304,
    txt_max_length=300,
    use_text_rope=True,
    text_rope_theta=10000.0,
    rope_mode="ntk_aware",
    rope_ref_h=1024,
    rope_ref_w=1024,
    repa_encoder_index=6,
    lq_inject_mode="controlnet",
    lq_in_channels=0,
    lq_latent_channels=16,
    lq_hidden_dim=512,
    lq_num_res_blocks=4,
    lq_gate_type="sigma_aware_per_token_per_dim",
    lq_interval=2,
    zero_init_lq=True,
    train_lq_proj_only=False,
    sr_scale=4,
    pit_lq_inject=False,
    pit_lq_gate_type="sigma_aware_per_token_per_dim",
)

GATE_PER_DIM = "sigma_aware_per_token_per_dim"   # v1.0：content_proj 输出 D
GATE_SCALAR = "sigma_aware_per_token"            # v1.5：content_proj 输出 1（通道共享）

T_LIST = [0.999, 0.866, 0.634, 0.342, 0.0]
TIMESCALE = 1000.0
SR_SCALE = 4
LATENT_SCALE = 0.3611      # flux VAE scaling_factor
LATENT_SHIFT = 0.1159      # flux VAE shift_factor
LATENT_DOWN = 8            # flux VAE /8（PiD 的 latent 语义）


def detect_net_kwargs(net_sd):
    """按 checkpoint 的真实形状推断 PidNet 配置（兼容 v1.0 / v1.5 等不同变体）。

    官方 model_loader 里是硬编码参数（v1.0：hidden 512 + per-token-per-dim gate）；
    v1.5 的 `lq_proj` 是 hidden 1024 + 标量 gate + 带 pit_head，硬编码会 size mismatch。
    这里改成从权重反推，换任何一版 ckpt 都不用改代码。
    """
    import re

    kw = dict(PID_NET_KWARGS)
    info = {}

    def shape(key):
        v = net_sd.get(key)
        return tuple(v.shape) if v is not None and hasattr(v, "shape") else None

    def count_blocks(prefix):
        ids = [int(m.group(1)) for k in net_sd for m in [re.match(rf"{prefix}\.(\d+)\.", k)] if m]
        return (max(ids) + 1) if ids else 0

    pd = count_blocks("patch_blocks")
    if pd:
        kw["patch_depth"] = pd
    qd = count_blocks("pixel_blocks")
    if qd:
        kw["pixel_depth"] = qd

    w = shape("lq_proj.latent_proj.0.weight")
    if w:
        kw["lq_hidden_dim"] = w[0]
        info["lq_hidden_dim"] = w[0]

    g = shape("lq_proj.gate_modules.0.content_proj.weight")
    if g:
        kw["lq_gate_type"] = GATE_SCALAR if g[0] == 1 else GATE_PER_DIM
        info["lq_gate_type"] = kw["lq_gate_type"]

    pg = shape("pit_lq_gate.content_proj.weight")
    kw["pit_lq_inject"] = pg is not None
    if pg is not None:
        kw["pit_lq_gate_type"] = GATE_SCALAR if pg[0] == 1 else GATE_PER_DIM
        info["pit_lq_inject"] = True

    heads = count_blocks("lq_proj.output_heads")
    if heads:
        kw["lq_interval"] = max(1, kw["patch_depth"] // heads)
        info["lq_heads"] = heads

    res_ids = sorted({int(m.group(1)) for k in net_sd
                      for m in [re.match(r"lq_proj\.latent_proj\.(\d+)\.block\.", k)] if m})
    if res_ids:
        kw["lq_num_res_blocks"] = len(res_ids)

    kw["lq_in_channels"] = 3 if any(k.startswith("lq_proj.image_conv") for k in net_sd) else 0
    info["patch_depth"] = kw["patch_depth"]
    info["pixel_depth"] = kw["pixel_depth"]
    return kw, info


def vram_stats(device=None):
    """当前显存快照，用于日志与验收。"""
    try:
        dev = device or mm.get_torch_device()
        free, total = torch.cuda.mem_get_info(dev)
        return "alloc=%.2fGB reserved=%.2fGB free=%.2fGB/%.2fGB" % (
            torch.cuda.memory_allocated(dev) / 1e9,
            torch.cuda.memory_reserved(dev) / 1e9, free / 1e9, total / 1e9)
    except Exception:
        return "n/a"


def release_comfy_vram(device=None, verbose=True):
    """把 comfy 已加载的模型（flux / 条件分支 / VAE）整块卸掉，给 PiD 让显存。

    两个必须知道的点：
      1. PiD 的网络是插件自己 ``net.to(cuda)``，**不归 comfy 显存管理**，
         所以 comfy 不会为它腾地方，必须手动释放；
      2. ``free_memory(need)`` 对 dynamic patcher 只做 ``partially_unload``，
         腾够 need 就停 —— 这一点是 4K 解码 OOM 的直接原因（flux 还剩 11GB）。
         要整块交出必须用 ``unload_all_models()``（等价 ``free_memory(1e30)``）。
    """
    import gc

    dev = device or mm.get_torch_device()
    before = vram_stats(dev)
    try:
        mm.unload_all_models()
    except Exception as e:                      # 绝不让清理失败影响出图
        logger.warning("[PiD] unload_all_models failed: %s", e)
    mm.soft_empty_cache()
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    after = vram_stats(dev)
    if verbose:
        logger.info("[PiD] comfy VRAM released: %s -> %s", before, after)
    return before, after


def to_pid_latent(latent_raw, scale=LATENT_SCALE, shift=LATENT_SHIFT):
    """comfy 采样输出（raw VAE latent z）→ PiD 需要的采样空间 latent X。"""
    return (latent_raw - shift) * scale


def _t_list_for(num_steps):
    full = torch.tensor(T_LIST, dtype=torch.float32)
    if num_steps is None or num_steps == len(T_LIST) - 1:
        return full
    idx = torch.linspace(0, len(full) - 1, num_steps).round().long()
    t = full[idx]
    t[-1] = 0.0
    return t


def _velocity_to_x0(x, v, t):
    """prediction_type = velocity → x0 = x - t*v（官方在 float64 里算再转回）。"""
    s = [x.shape[0]] + [1] * (x.ndim - 1)
    return (x.double() - t.double().view(*s) * v.double()).to(x.dtype)


class PidDecoder:
    def __init__(self, ckpt_path, caption_embeddings_path, dtype=torch.bfloat16,
                 verbose=True):
        from .lucidflux_pid import PidNet

        self.device = mm.get_torch_device()
        self.dtype = dtype
        self.ckpt_path = ckpt_path

        emb = torch.load(caption_embeddings_path, map_location="cpu", weights_only=False)
        if "caption_embs" not in emb:
            raise KeyError(f"{caption_embeddings_path} 缺少 caption_embs")
        self.caption = emb.get("caption", "")
        # 留在 CPU；decode 时再按需搬（避免常驻显存）
        self.caption_embs = emb["caption_embs"].float()
        self.null_caption_embs = None
        null_path = os.path.join(os.path.dirname(caption_embeddings_path), "null_caption_embs.pt")
        if os.path.exists(null_path):
            self.null_caption_embs = torch.load(null_path, map_location="cpu", weights_only=True)

        sd = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
        net_sd, dropped = {}, 0
        for k, v in sd.items():
            if k.startswith("net.") and not k.startswith("net_ema."):
                net_sd[k[4:]] = v
            elif k.startswith(("net_ema.", "fake_score.", "discriminator.", "model_ema.")):
                dropped += 1
            else:
                net_sd[k] = v
        del sd

        net_kwargs, detected = detect_net_kwargs(net_sd)
        self.net_kwargs = net_kwargs
        self.detected = detected
        if verbose:
            logger.info("[PiD] ckpt variant detected: %s", detected)
            logger.info("[PiD] building PidNet ...")
        # 关键：在 CPU 上构造与装载。以前这里直接 .to(cuda)，导致
        #   ① comfy 还占着 flux 时就抢显存；② 流式分支结束后永不回 CPU（显存泄漏感）。
        # fp32 装载后再转目标 dtype，装载精度与之前一致。
        net = PidNet(**net_kwargs)
        result = net.load_state_dict(net_sd, strict=False)
        del net_sd
        self.load_report = (len(result.missing_keys), len(result.unexpected_keys), dropped)
        if result.missing_keys:
            raise RuntimeError(
                "PiD 权重与网络结构不匹配（缺少 %d 个 key，前几个：%s）。"
                "detected=%s；若这是新的 ckpt 变体，请把上面这行日志发出来。"
                % (len(result.missing_keys), result.missing_keys[:6], detected))
        if verbose:
            logger.info("[PiD] ckpt loaded: missing=0 unexpected=%d skipped_prefix=%d"
                        % (len(result.unexpected_keys), dropped))
            if result.unexpected_keys[:6]:
                logger.warning("[PiD] unexpected keys (ignored): %s", result.unexpected_keys[:6])

        self.net = net.to(dtype=dtype).eval()   # 仍在 CPU
        self.net.requires_grad_(False)
        if verbose:
            n = sum(p.numel() for p in self.net.parameters())
            logger.info("[PiD] params %.2fB dtype=%s", n / 1e9, dtype)

    # ---- 主入口 ---------------------------------------------------------
    @torch.no_grad()
    def decode(self, latent_raw, seed=0, num_steps=4, degrade_sigma=0.0,
               streaming_prefetch=0, caption_embs=None, release_comfy=True, verbose=True):
        """latent_raw: comfy 采样器输出的 flux latent [B,16,h,w] → [B,3,H,W] in [-1,1]."""
        import gc

        device = self.device
        if release_comfy:
            release_comfy_vram(device, verbose=verbose)
        latent = to_pid_latent(latent_raw.to(device=device, dtype=torch.float32))
        b, _, lh, lw = latent.shape
        vae_h, vae_w = lh * LATENT_DOWN, lw * LATENT_DOWN
        target_h, target_w = vae_h * SR_SCALE, vae_w * SR_SCALE

        # 显存预算：峰值几乎全是**激活**（∝ 输出像素数），权重流式后只有 ~0.8GB。
        # 实测 bf16/streaming=1：512²=0.62GB、1024²=1.50、1536²=3.02、2048²=5.18、
        # 3072²=11.21、4096²=18.64 → 拟合 peak ≈ 1.1e-6 × pixels + 0.35 (GB)
        est_gb = 1.1e-6 * target_h * target_w + 0.35
        try:
            free_gb = torch.cuda.mem_get_info(device)[0] / 1e9
        except Exception:
            free_gb = float("inf")
        if verbose:
            logger.info("[PiD] 预计峰值 %.1fGB（输出 %d×%d）｜卸载后可用 %.1fGB",
                        est_gb, target_h, target_w, free_gb)
        if est_gb > free_gb * 1.05:
            logger.warning(
                "[PiD] 峰值 %.1fGB 超过物理可用 %.1fGB → 会走 Windows 共享内存："
                "能出图但很慢，任务管理器会看到 20GB+。12GB 卡建议 latent≤64（2048²，峰值 5.2GB）；"
                "要真 4096²（18.6GB）请上 ≥24GB 卡。", est_gb, free_gb)

        # 权重常驻 CPU，这里按需搬上卡（每张图一次，2.7MB 可忽略）
        src_embs = self.caption_embs if caption_embs is None else caption_embs
        embs = src_embs.to(device=device, dtype=torch.float32)
        if embs.shape[0] != b:
            embs = embs.expand(b, -1, -1).contiguous()
        degrade = torch.full((b,), float(degrade_sigma), device=device, dtype=torch.float32)

        gen = torch.Generator(device=device).manual_seed(int(seed))
        noise = torch.randn(b, 3, target_h, target_w, device=device, generator=gen)

        t_list = _t_list_for(num_steps).to(device=device)
        use_streaming = streaming_prefetch is not None and streaming_prefetch > 0

        if use_streaming:
            from .lucidflux_pid.streaming import _streaming_model
            ctx = _streaming_model(self.net, ["pixel_blocks", "patch_blocks"], device,
                                   streaming_prefetch)
        else:
            self.net.to(device=device, dtype=self.dtype)
            ctx = _null_ctx(self.net)

        if verbose:
            logger.info("[PiD] decode start %dx%d steps=%d streaming=%s %s",
                        target_h, target_w, len(t_list) - 1,
                        streaming_prefetch if use_streaming else "off", vram_stats(device))
        try:
            with ctx as net:
                x = noise
                with torch.autocast("cuda", dtype=self.dtype):
                    for t_cur, t_next in zip(t_list[:-1], t_list[1:]):
                        t_b = t_cur.expand(b)
                        v = net(x, t_b * TIMESCALE, embs,
                                lq_video_or_image=None, lq_latent=latent, degrade_sigma=degrade)
                        if float(t_next) > 0:
                            x0 = _velocity_to_x0(x, v, t_b)
                            eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype,
                                              generator=gen)
                            s = [b] + [1] * (x.ndim - 1)
                            tn = t_next.reshape(1).expand(s)
                            x = (1.0 - tn) * x0 + tn * eps
                        else:
                            x = _velocity_to_x0(x, v, t_b)
        finally:
            # 无论成功/异常都把权重搬回 CPU 并清缓存，否则下一张图会撞上一次的残留
            if not use_streaming:
                self.net.to("cpu")
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if verbose:
                logger.info("[PiD] decode end: peak=%.2fGB %s",
                            torch.cuda.max_memory_allocated(device) / 1e9, vram_stats(device))
        return x.clamp(-1, 1), (target_h, target_w)


class _null_ctx:
    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        return self.obj

    def __exit__(self, *a):
        return False


def align_pid_color(pid_output_01, reference_01, strength=1.0):
    """官方 decoder.align_color：pid 高频 + 参考图低频（strength 混合）。"""
    from .align_color import wavelet_decomposition

    strength = max(0.0, min(1.0, float(strength)))
    ref = reference_01
    if ref.shape[-2:] != pid_output_01.shape[-2:]:
        ref = torch.nn.functional.interpolate(ref, size=pid_output_01.shape[-2:],
                                              mode="bilinear", align_corners=False)
    pid_high, pid_low = wavelet_decomposition(pid_output_01)
    _, ref_low = wavelet_decomposition(ref)
    return (pid_high + ref_low * strength + pid_low * (1.0 - strength)).clamp(0, 1)
