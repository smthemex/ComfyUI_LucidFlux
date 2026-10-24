"""颜色对齐：官方 align_color.py 中的 wavelet 重建（纯 torch）。

官方 2K 流程尾段：
    hq = wavelet_reconstruction(vae_decoded[0,1], ldr_up[0,1])
即把“退化/复原参考图”(ldr) 的低频（颜色、亮度）搬到 VAE 解码结果上，
避免扩散阶段偏色。此处只保留需要的两个函数，去掉 PIL/torchvision 依赖。
"""
import torch
import torch.nn.functional as F
from torch import Tensor


def wavelet_blur(image: Tensor, radius: int):
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None].repeat(image.shape[1], 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(image, kernel, groups=image.shape[1], dilation=radius)


def wavelet_decomposition(image: Tensor, levels=5):
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq
    return high_freq, low_freq


def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor):
    """用 style 的低频 + content 的高频重建（官方逐字实现）。"""
    content_high_freq, _ = wavelet_decomposition(content_feat)
    _, style_low_freq = wavelet_decomposition(style_feat)
    return content_high_freq + style_low_freq
