import os
import torch
import numpy as np
from PIL import Image
from einops import rearrange, repeat
from typing import Optional
import math
import torch.nn as nn
from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
from .src.flux.sampling import denoise_lucidflux, get_noise, get_schedule, unpack
from .src.flux.util import load_flow_model, load_single_condition_branch, load_safetensors
from .src.flux.swinir import SwinIR


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
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
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb

ACT2CLS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}

def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
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
        
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=x.dtype))  # (N, D)

        # Expand scalar control_index to batch dimension and project like timesteps (256-dim)
        if control_index.dim() == 0:
            control_index = control_index.repeat(x.shape[0])
        elif control_index.dim() == 1 and control_index.shape[0] != x.shape[0]:
            control_index = control_index.expand(x.shape[0])
        control_index = control_index.to(device=x.device, dtype=x.dtype)
        control_index_proj = self.time_proj(control_index)
        control_index_emb = self.control_index_embedder(control_index_proj.to(dtype=x.dtype))  # (N, D)
        timesteps_emb = timesteps_emb + control_index_emb
        emb = self.linear(self.silu(timesteps_emb))
        shift_msa, scale_msa = emb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x

class DualConditionBranch(nn.Module):
    def __init__(self, condition_branch_lq: nn.Module, condition_branch_ldr: nn.Module, modulation_lq: nn.Module, modulation_ldr: nn.Module):
        super().__init__()
        self.lq = condition_branch_lq
        self.ldr = condition_branch_ldr
        self.modulation_lq = modulation_lq
        self.modulation_ldr = modulation_ldr

    def forward(
        self,
        *,
        img,
        img_ids,
        condition_cond_lq,
        txt,
        txt_ids,
        y,
        timesteps,
        guidance,
        condition_cond_ldr=None,
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
        
        out_ldr = self.ldr(
            img=img,
            img_ids=img_ids,
            controlnet_cond=condition_cond_ldr,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=timesteps,
            guidance=guidance,
        )
        out = []
        num_blocks = 19
        for i in range(num_blocks // 2 + 1):
            for control_index, (lq, ldr) in enumerate(zip(out_lq, out_ldr)):
                control_index = torch.tensor(control_index, device=timesteps.device, dtype=timesteps.dtype)
                lq = self.modulation_lq(lq, timesteps, i * 2 + control_index)

                if len(out) == num_blocks:
                    break

                ldr = self.modulation_ldr(ldr, timesteps, i * 2 + control_index)
                out.append(lq + ldr)
        return out




def preprocess_lq_image(image_path: str, width: int = 512, height: int = 512):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((width, height))
    return image


def load_redux_image_encoder(device: torch.device, dtype: torch.dtype, redux_state_dict: str):
    redux_image_encoder = ReduxImageEncoder()
    redux_image_encoder.load_state_dict(redux_state_dict, strict=False)

    redux_image_encoder.eval()
    redux_image_encoder.to(device).to(dtype=dtype)
    return redux_image_encoder



def load_lucidflux_model(args,ckpt_path,cf_model,torch_device):
    name =args.name #"flux-dev"
    #offload = args.offload
    is_schnell = name == "flux-schnell"
    


    model=load_flow_model(name,ckpt_path,cf_model)


    condition_lq=load_single_condition_branch(name, torch_device).to(torch.bfloat16)


    # load model checkpoint
    if '.safetensors' in args.checkpoint:
        checkpoint = load_safetensors(args.checkpoint)
    else:
        checkpoint = torch.load(args.checkpoint,weights_only=False, map_location='cpu')

    condition_lq.load_state_dict(checkpoint["condition_lq"], strict=False)
    condition_lq = condition_lq.to(torch_device)

    condition_ldr = load_single_condition_branch(name, torch_device).to(torch.bfloat16)
    condition_ldr.load_state_dict(checkpoint["condition_ldr"], strict=False)

    modulation_lq = Modulation(dim=3072).to(torch.bfloat16)
    modulation_lq.load_state_dict(checkpoint["modulation_lq"], strict=False)

    modulation_ldr = Modulation(dim=3072).to(torch.bfloat16)
    modulation_ldr.load_state_dict(checkpoint["modulation_ldr"], strict=False)


    dual_condition_branch = DualConditionBranch(
            condition_lq,
            condition_ldr,
            modulation_lq=modulation_lq,
            modulation_ldr=modulation_ldr,
        ).to(torch_device)
    

    
    state_dict=checkpoint["connector"]
    del checkpoint
    pipe={"model":model,"dual_condition_branch":dual_condition_branch,"is_schnell":is_schnell}
    return pipe,state_dict

def preprocess_data(state_dict,swinir_path,siglip_model,input_pli_list, inp_cond,torch_device):
     # SwinIR prior (frozen)
    # if args.swinir_pretrained is None:
    #     raise ValueError("SwinIR pretrained is not provided")
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
    ckpt_obj = torch.load(swinir_path, weights_only=False,map_location="cpu")
    state = ckpt_obj.get("state_dict", ckpt_obj)
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    swinir.load_state_dict(new_state, strict=False)
    swinir.eval()
    del state
    for p in swinir.parameters():
        p.requires_grad_(False)
    try:
        swinir = swinir.to(torch_device)
    except:
        swinir = swinir.to(torch.device("cpu"))
    #swinir = swinir.to(torch_device)

    dtype = torch.bfloat16 if torch_device.type == 'cuda' else torch.float32
    # siglip_model = SiglipVisionModel.from_pretrained(siglip_ckpt)
    # siglip_model.eval()
    # siglip_model.to(torch_device).to(dtype=dtype)

    redux_image_encoder = load_redux_image_encoder(torch_device, dtype, state_dict)
    del state_dict
    data_list=[]
    for lq_processed in input_pli_list:
        #filename = os.path.basename(img_path).split(".")[0]
        
        # For each image, compute processed resolution and persist preview
        #lq_processed = preprocess_lq_image(img_path, width, height)
        #lq_processed.save(os.path.join(args.output_dir, f"{filename}_lq_processed.jpeg"))
        condition_cond = torch.from_numpy((np.array(lq_processed) / 127.5) - 1)
        condition_cond = condition_cond.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(torch_device)
        condition_cond_ldr = None

        with torch.no_grad():
            # SwinIR prior
            ci_01 = torch.clamp((condition_cond.float() + 1.0) / 2.0, 0.0, 1.0)
            ci_pre = swinir(ci_01).float().clamp(0.0, 1.0).to(torch_device) #(1,3,H,W) or 3 h w
            ci_pre_origin=ci_pre
            # save_image(ci_pre, os.path.join(args.output_dir, f"{filename}_swinir_pre.jpeg"))
            condition_cond_ldr = (ci_pre * 2.0 - 1.0).to(torch.bfloat16)
            # to comfy siglip
            if ci_pre.ndim == 3:
                ci_pre = ci_pre.unsqueeze(0)
            _,_,height,width=ci_pre.shape
            #print(ci_pre.shape) #torch.Size([1, 3, 512, 512])
            ci_pre=ci_pre.permute(0, 2, 3, 1) # to comfy
            siglip_image_pre_fts=siglip_model.encode_image(ci_pre)["last_hidden_state"].to(device=torch_device,dtype=torch.bfloat16)
            #print(siglip_image_pre_fts.shape) #torch.Size([1, 1024, 1152])
          


            enc_dtype = redux_image_encoder.redux_up.weight.dtype
            image_embeds = redux_image_encoder(
                siglip_image_pre_fts.to(device=torch_device, dtype=enc_dtype)
            )["image_embeds"]
            #print(image_embeds.shape) #torch.Size([1, 1024, 4096])
            # concat to txt and extend txt_ids
            txt = inp_cond["txt"].to(device=torch_device, dtype=torch.bfloat16)
            txt_ids = inp_cond["txt_ids"].to(device=torch_device, dtype=torch.bfloat16)
            siglip_txt = torch.cat([txt, image_embeds.to(dtype=torch.bfloat16)], dim=1).to(device=torch_device, dtype=torch.bfloat16)
            B, L, C = txt_ids.shape
            extra_ids = torch.zeros((B, 1024, C), device=txt_ids.device, dtype=torch.bfloat16)
            siglip_txt_ids = torch.cat([txt_ids, extra_ids], dim=1).to(device=torch_device,dtype=torch.bfloat16)

          
            data={"siglip_txt": siglip_txt, "siglip_txt_ids": siglip_txt_ids,"inp_cond":inp_cond,"txt":txt,"txt_ids":txt_ids,"size":(height, width),
                  "condition_cond":condition_cond, "condition_cond_ldr": condition_cond_ldr,"ci_pre_origin":ci_pre_origin}
        data_list.append(data)
    return data_list

def prepare_with_embeddings(img, precomputed_txt, precomputed_vec):
    """
    使用预计算embeddings的prepare函数
    """
    bs, _, h, w = img.shape

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    img_ids = torch.zeros(h // 2, w // 2, 3, device=img.device, dtype=img.dtype)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=img.device)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=img.device)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    # 直接使用预计算的embeddings
    txt = precomputed_txt
    vec = precomputed_vec
    txt_ids = torch.zeros(bs, txt.shape[1], 3, device=img.device, dtype=img.dtype)

    return {
        "img": img,
        "img_ids": img_ids,
        "txt": txt,
        "txt_ids": txt_ids,
        "vec": vec,
    }
       
def get_cond(positive,emb_path,height,width,device,bs=1):
    h=2 * math.ceil(height / 16)
    w=2 * math.ceil(width / 16)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    if emb_path is None and positive is not None:
        txt = positive[0][0]
        if txt.shape[0] == 1 and bs > 1:
            txt = repeat(txt, "1 ... -> bs ...", bs=bs)
        txt_ids = torch.zeros(bs, txt.shape[1], 3)
        vec = positive[0][1].get("pooled_output")
        if vec.shape[0] == 1 and bs > 1:
            vec = repeat(vec, "1 ... -> bs ...", bs=bs)

        return inp_cond
    elif emb_path is not None:
        # 使用预计算的embeddings
        #embeddings_path = "weights/lucidflux/prompt_embeddings.pt"
        print(f"Loading precomputed embeddings from {emb_path}")
        embeddings_data = torch.load(emb_path,weights_only=False, map_location='cpu')
        precomputed_txt = embeddings_data['txt'].to(device)
        precomputed_vec = embeddings_data['vec'].to(device)
        original_prompt = embeddings_data.get('prompt', 'Unknown prompt')
        print(f"Loaded embeddings for prompt: '{original_prompt}',txt shape: {precomputed_txt.shape}, vec shape: {precomputed_vec.shape}")
        # 直接使用预计算的embeddings
        txt = precomputed_txt
        vec = precomputed_vec
        txt_ids = torch.zeros(bs, txt.shape[1], 3)
    else:
        raise ValueError("Invalid embedding path or conditions")
    
    inp_cond={
            "img_ids": img_ids.to(device),
            "txt": txt.to(device),
            "txt_ids": txt_ids.to(device),
            "vec": vec.to(device),
                }
    return inp_cond

def load_condition_model(model,lora_path,lora_scale):
    if lora_path is None:
            return model
    try:
        model_int = model.get("model")
        _apply_lora_weights(model_int, lora_path, lora_scale)
        
        print(f"Successfully applied LoRA: {lora_path} with scale {lora_scale}")
        model["model"]=model_int

        return model
        
    except Exception as e:
        print(f"Failed to apply LoRA {str(e)}")
        return model

def _apply_lora_weights( model, lora_path, scale):

    from safetensors.torch import load_file as load_sft
    
    lora_sd = load_sft(lora_path, device="cpu")
    
    # 获取模型状态字典
    model_sd = model.state_dict()
    
    # 应用LoRA权重
    for key in lora_sd:
        if "lora_up" in key:
            # 找到对应的down权重和原始权重
            down_key = key.replace("lora_up", "lora_down")
            original_key = _get_original_key(key)
            
            if down_key in lora_sd and original_key in model_sd:
                up_weight = lora_sd[key]
                down_weight = lora_sd[down_key]
                original_weight = model_sd[original_key]
                
                # 计算LoRA增量并应用
                with torch.no_grad():
                    lora_delta = (down_weight @ up_weight) * scale
                    model_sd[original_key].copy_(original_weight + lora_delta)
    
    # 更新模型权重
    model.load_state_dict(model_sd)
    del lora_sd

def _get_original_key(lora_key):
    """从LoRA键名获取原始模型键名"""
    # 移除LoRA特定的后缀
    original_key = lora_key.replace(".lora_up.weight", ".weight")
    original_key = original_key.replace(".lora_down.weight", ".weight")
    return original_key


def lucidflux_inference(model,dual_condition_branch,input_data,guidance,num_steps,seed,torch_device,is_schnell=False):
    lat_list = []
    for data in input_data: #input_data [dict,dict...]
      
        
        with torch.no_grad():
          
            height, width=data.get("size")
            #print(f"height:{height}, width:{width}")
            torch.manual_seed(seed)
            x = get_noise(
                1, height, width, device=torch_device,
                dtype=torch.bfloat16, seed=seed
            )
            bs, c, h, w = x.shape
            img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

            if img.shape[0] == 1 and bs > 1:
                img = repeat(img, "1 ... -> bs ...", bs=bs)

            timesteps = get_schedule(
                            num_steps,
                            (width // 8) * (height // 8) // (16 * 16),
                            shift=(not is_schnell),
                        )
          
            print("start denoise...")
            x = denoise_lucidflux(
                model,
                dual_condition_model=dual_condition_branch,
                img=img,
                img_ids=data.get("inp_cond")["img_ids"],
                txt=data.get("txt"),
                txt_ids=data.get("txt_ids"),
                siglip_txt=data.get("siglip_txt"),
                siglip_txt_ids=data.get("siglip_txt_ids"),
                vec=data.get("inp_cond")["vec"],
                timesteps=timesteps,
                guidance=guidance,
                condition_cond_lq=data.get("condition_cond"),
                condition_cond_ldr=data.get("condition_cond_ldr"),
            )


            x = unpack(x.float(), height, width)

            x=(x/0.3611)+0.1159 #mean


        lat_list.append(x)

        print("start decoder...")
    
    return lat_list


