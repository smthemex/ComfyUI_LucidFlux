import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from PIL import Image


def _expand_batch(tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return repeat(tensor, "1 ... -> b ...", b=batch_size)
    raise ValueError(f"{name} batch size {tensor.shape[0]} does not match expected batch size {batch_size}")


def move_modules_to_device(device: torch.device | str, *modules: nn.Module) -> None:
    for module in modules:
        module.to(device)


def prepare_with_embeddings(
    img: torch.Tensor,
    precomputed_txt: torch.Tensor,
    precomputed_vec: torch.Tensor,
):
    bs, _, h, w = img.shape

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    img_ids = torch.zeros(h // 2, w // 2, 3, device=img.device, dtype=img.dtype)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=img.device)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=img.device)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    txt = _expand_batch(precomputed_txt, bs, "precomputed_txt").to(img.device)
    vec = _expand_batch(precomputed_vec, bs, "precomputed_vec").to(img.device)
    txt_ids = torch.zeros(bs, txt.shape[1], 3, device=img.device, dtype=img.dtype)

    return {
        "img": img,
        "img_ids": img_ids,
        "txt": txt,
        "txt_ids": txt_ids,
        "vec": vec,
    }


def prepare_with_embeddings(
    img: torch.Tensor,
    precomputed_txt: torch.Tensor,
    precomputed_vec: torch.Tensor,
):
    bs, _, h, w = img.shape

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    img_ids = torch.zeros(h // 2, w // 2, 3, device=img.device, dtype=img.dtype)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=img.device)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=img.device)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    txt = _expand_batch(precomputed_txt, bs, "precomputed_txt").to(img.device)
    vec = _expand_batch(precomputed_vec, bs, "precomputed_vec").to(img.device)
    txt_ids = torch.zeros(bs, txt.shape[1], 3, device=img.device, dtype=img.dtype)

    return {
        "img": img,
        "img_ids": img_ids,
        "txt": txt,
        "txt_ids": txt_ids,
        "vec": vec,
    }



def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


ACT2CLS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def get_activation(act_fn: str) -> nn.Module:
    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    raise ValueError(f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}")


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Modulation(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 2 * dim, bias=bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=dim)
        self.control_index_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=dim)

    def forward(self, x, timestep, control_index):
        timesteps_proj = self.time_proj(timestep * 1000)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=x.dtype))

        if control_index.dim() == 0:
            control_index = control_index.repeat(x.shape[0])
        elif control_index.dim() == 1 and control_index.shape[0] != x.shape[0]:
            control_index = control_index.expand(x.shape[0])
        control_index = control_index.to(device=x.device, dtype=x.dtype)
        control_index_proj = self.time_proj(control_index)
        control_index_emb = self.control_index_embedder(control_index_proj.to(dtype=x.dtype))
        timesteps_emb = timesteps_emb + control_index_emb
        emb = self.linear(self.silu(timesteps_emb))
        shift_msa, scale_msa = emb.chunk(2, dim=1)
        return self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]


class DualConditionBranch(nn.Module):
    def __init__(self, condition_branch_lq: nn.Module, condition_branch_pre: nn.Module, modulation_lq: nn.Module, modulation_pre: nn.Module):
        super().__init__()
        self.lq = condition_branch_lq
        self.pre = condition_branch_pre
        self.modulation_lq = modulation_lq
        self.modulation_pre = modulation_pre

    def forward(
        self,
        *,
        img,
        img_ids,
        txt,
        txt_ids,
        y,
        timesteps,
        guidance,
        condition_cond_lq,
        condition_cond_pre,
    ):

        out_lq = self.lq(
            img=img,
            img_ids=img_ids,
            controlnet_cond=condition_cond_lq,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=timesteps,
            guidance=guidance,
        )

        out_pre = self.pre(
            img=img,
            img_ids=img_ids,
            controlnet_cond=condition_cond_pre,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=timesteps,
            guidance=guidance,
        )
        out = []
        num_blocks = 19
        for i in range(num_blocks // 2 + 1):
            for control_index, (lq, pre) in enumerate(zip(out_lq, out_pre)):
                control_index = torch.tensor(control_index, device=timesteps.device, dtype=timesteps.dtype)
                lq = self.modulation_lq(lq, timesteps, i * 2 + control_index)

                if len(out) == num_blocks:
                    break

                pre = self.modulation_pre(pre, timesteps, i * 2 + control_index)
                out.append(lq + pre)
        return out


class ConditionBranchWithRedux(nn.Module):
    def __init__(self, condition_branch: nn.Module, redux_image_encoder: nn.Module):
        super().__init__()
        self.condition_branch = condition_branch
        self.redux_image_encoder = redux_image_encoder

    def forward(
        self,
        *,
        img,
        img_ids,
        txt,
        txt_ids,
        y,
        timesteps,
        guidance,
        condition_cond_lq,
        condition_cond_pre=None,
        siglip_image_pre_fts=None,
    ):
        siglip_txt = txt
        siglip_txt_ids = txt_ids

        if siglip_image_pre_fts is not None:
            enc_dtype = self.redux_image_encoder.redux_up.weight.dtype
            image_embeds = self.redux_image_encoder(
                siglip_image_pre_fts.to(device=txt.device, dtype=enc_dtype)
            )["image_embeds"].to(dtype=txt.dtype)
            siglip_txt = torch.cat([txt, image_embeds], dim=1)
            batch_size, _, channels = txt_ids.shape
            extra_ids = torch.zeros((batch_size, 1024, channels), device=txt_ids.device, dtype=txt_ids.dtype)
            siglip_txt_ids = torch.cat([txt_ids, extra_ids], dim=1)

        return self.condition_branch(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=timesteps,
            guidance=guidance,
            condition_cond_lq=condition_cond_lq,
            condition_cond_pre=condition_cond_pre,
        ), siglip_txt, siglip_txt_ids


def preprocess_lq_image(image_path: str, width: int = 512, height: int = 512):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height))
    return image


def load_lucidflux_weights(weights_path: str):
    return torch.load(weights_path, map_location="cpu")


def load_precomputed_embeddings(embeddings_path: str, device: torch.device | str):
    embeddings_data = torch.load(embeddings_path, map_location="cpu")
    return {
        "txt": embeddings_data["txt"].to(device),
        "vec": embeddings_data["vec"].to(device),
        "prompt": embeddings_data.get("prompt", "Unknown prompt"),
    }


def load_dual_condition_branch(
    name: str,
    lucidflux_weights: dict,
    device: torch.device | str,
    offload: bool,
    branch_dtype: torch.dtype = torch.bfloat16,
    modulation_dim: int = 3072,
):
    from src.flux.util import load_single_condition_branch

    load_device = "cpu" if offload else device
    target_device = "cpu" if offload else device

    condition_lq = load_single_condition_branch(name, load_device).to(branch_dtype)
    condition_lq.load_state_dict(lucidflux_weights["condition_lq"], strict=False)
    condition_lq = condition_lq.to(device)

    condition_pre = load_single_condition_branch(name, load_device).to(branch_dtype)
    condition_pre.load_state_dict(lucidflux_weights["condition_ldr"], strict=False)

    modulation_lq = Modulation(dim=modulation_dim).to(branch_dtype)
    modulation_lq.load_state_dict(lucidflux_weights["modulation_lq"], strict=False)

    modulation_pre = Modulation(dim=modulation_dim).to(branch_dtype)
    modulation_pre.load_state_dict(lucidflux_weights["modulation_ldr"], strict=False)

    return DualConditionBranch(
        condition_lq,
        condition_pre,
        modulation_lq=modulation_lq,
        modulation_pre=modulation_pre,
    ).to(target_device)




def load_swinir(device: torch.device | str, checkpoint_path: str, offload: bool):
    if checkpoint_path is None:
        raise ValueError("SwinIR pretrained is not provided")

    from src.flux.swinir import SwinIR

    swinir = SwinIR(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2,
        sf=8,
        img_range=1.0,
        upsampler="nearest+conv",
        resi_connection="1conv",
        unshuffle=True,
        unshuffle_scale=8,
    )
    ckpt_obj = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt_obj.get("state_dict", ckpt_obj)
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    swinir.load_state_dict(new_state, strict=False)
    swinir.eval()
    for p in swinir.parameters():
        p.requires_grad_(False)
    return swinir.to("cpu" if offload else device)


def load_siglip_model(siglip_ckpt: str, device: torch.device | str, dtype: torch.dtype, offload: bool):
    from transformers import SiglipVisionModel

    siglip_model = SiglipVisionModel.from_pretrained(siglip_ckpt)
    siglip_model.eval()
    siglip_model.to("cpu" if offload else device).to(dtype=dtype)
    return siglip_model


def load_redux_image_encoder(device: torch.device | str, dtype: torch.dtype, redux_state_dict: dict):
    from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder

    redux_image_encoder = ReduxImageEncoder()
    redux_image_encoder.load_state_dict(redux_state_dict, strict=False)
    redux_image_encoder.eval()
    redux_image_encoder.to(device).to(dtype=dtype)
    return redux_image_encoder

def load_condition_branch_with_redux(
    name: str,
    lucidflux_weights: dict,
    device: torch.device | str,
    offload: bool,
    branch_dtype: torch.dtype = torch.bfloat16,
    connector_dtype: torch.dtype = torch.float32,
    modulation_dim: int = 3072,
):
    dual_condition_branch = load_dual_condition_branch(
        name=name,
        lucidflux_weights=lucidflux_weights,
        device=device,
        offload=offload,
        branch_dtype=branch_dtype,
        modulation_dim=modulation_dim,
    )
    redux_image_encoder = load_redux_image_encoder(
        "cpu" if offload else device,
        connector_dtype,
        lucidflux_weights["connector"],
    )
    return ConditionBranchWithRedux(dual_condition_branch, redux_image_encoder)



def export_lucidflux_state(model: nn.Module):
    dual_condition_branch = model.condition_branch
    return {
        "condition_lq": dual_condition_branch.lq.state_dict(),
        "condition_ldr": dual_condition_branch.pre.state_dict(),
        "modulation_lq": dual_condition_branch.modulation_lq.state_dict(),
        "modulation_ldr": dual_condition_branch.modulation_pre.state_dict(),
        "connector": model.redux_image_encoder.state_dict(),
    }




__all__ = [
    "DualConditionBranch",
    "ConditionBranchWithRedux",
    "Modulation",
    "Timesteps",
    "TimestepEmbedding",
    "_expand_batch",
    "get_activation",
    "get_timestep_embedding",
    "export_lucidflux_state",
    "load_condition_branch_with_redux",
    "load_dual_condition_branch",
    "load_lucidflux_weights",
    "load_precomputed_embeddings",
    "load_redux_image_encoder",
    "load_siglip_model",
    "load_swinir",
    "move_modules_to_device",
    "prepare_with_embeddings",
    "preprocess_lq_image",
]
