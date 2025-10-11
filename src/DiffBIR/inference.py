#from argparse import ArgumentParser, Namespace
import os

import torch
from omegaconf import OmegaConf
from accelerate.utils import set_seed
from .diffbir.inference import (
    BSRInferenceLoop,
    BFRInferenceLoop,
    BIDInferenceLoop,
    UnAlignedBFRInferenceLoop,
    CustomInferenceLoop,
)


def check_device(device: str) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            print(
                "CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled."
            )
            device = "cpu"
    else:
        if device == "mps":
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print(
                        "MPS not available because the current PyTorch install was not "
                        "built with MPS enabled."
                    )
                    device = "cpu"
                else:
                    print(
                        "MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine."
                    )
                    device = "cpu"
    print(f"using device {device}")
    return device


DEFAULT_POS_PROMPT = (
    "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
    "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, "
    "skin pore detailing, hyper sharpness, perfect without deformations."
)

DEFAULT_NEG_PROMPT = (
    "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, "
    "CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, "
    "signature, jpeg artifacts, deformed, lowres, over-smooth."
)



diffbir_opts={
    "task": "sr",
    "upscale": 4.0,
    "version": "v2.1",
    "train_cfg": "",
    "ckpt": "",
    "sampler": "edm_dpm++_3m_sde",
    "steps": 10,
    "start_point_type": "noise",
    "cleaner_tiled": False,
    "cleaner_tile_size": 512,    
    "cleaner_tile_stride": 256,
    "vae_encoder_tiled": False,
    "vae_encoder_tile_size": 256,
    "vae_decoder_tiled": False,
    "vae_decoder_tile_size": 256,
    "cldm_tiled": False,  
    "cldm_tile_size": 512,  
    "cldm_tile_stride": 256,
    "captioner": "none",#["llava", "ram"]
    "pos_prompt": DEFAULT_POS_PROMPT,
    "neg_prompt": DEFAULT_NEG_PROMPT,
    "cfg_scale": 8.0,
    "rescale_cfg": True,  
    "noise_aug": 0,
    "s_churn": 0.0,
    "s_tmin": 0.0,
    "s_tmax": 300.0,
    "s_noise": 1.0,
    "eta": 1.0,
    "order": 1,
    "strength": 1.0,
    "batch_size": 1,
    "guidance": False, 
    "g_loss": "w_mse",#["mse", "w_mse"]
    "g_start": 1001,
    "g_stop": -1,
    "g_scale": 0.0,
    "g_space": "latent",#["rgb", "latent"]
    "g_repeat": 5,
    "inputs": "",
    "n_samples": 1,
    "output": "",
    "seed": 231,
    "device": "cuda",#["cpu", "cuda", "mps"]
    "precision": "bf16",#["fp32", "fp16", "bf16"]
    "llava_bit": "4",#["16", "8", "4"]
    "swinir_realesrgan": None,#swinir realesrgan model path
    "swinir_face": None,#swinir face model path
    "config": "configs/inference/swinir.yaml",#bsrnet or swinir config path only for custom version
    "cldm_config": "configs/inference/cldm.yaml",#controlnet model path only for custom version
    "sd_v2_zsnr": None,
    "control_model": None,#controlnet model path only for custom version
    "diffusion_config": "configs/inference/diffusion_v2.1.yaml",
}


def load_diffbir_model(task,swinir_path,device,cur_path,out_path):
    if task =="sr":
        upscale=4.0
    elif task =="denoise":
        upscale=1.0
    elif task==  "face":
        upscale=1.0
    elif task== "unaligned_face":
        upscale=2.0

    args=OmegaConf.create(diffbir_opts)
    
    args.swinir_realesrgan=os.path.join (os.path.dirname(swinir_path),"realesrgan_s4_swinir_100k.pth") if os.path.isfile(os.path.join (os.path.dirname(swinir_path),"realesrgan_s4_swinir_100k.pth"))  else None
    args.swinir_face=os.path.join (os.path.dirname(swinir_path),"face_swinir_v1.ckpt") if os.path.isfile(os.path.join (os.path.dirname(swinir_path),"face_swinir_v1.ckpt"))  else None
    args.sd_v2_zsnr=swinir_path  if "zsnr" in swinir_path.lower()  else None #sd2.1-base-zsnr-laionaes5.ckpt
    args.control_model=os.path.join (os.path.dirname(swinir_path),"DiffBIR_v2.1.pt") if os.path.isfile(os.path.join (os.path.dirname(swinir_path),"DiffBIR_v2.1.pt"))  else None
    
    args.config=os.path.join (cur_path,"src/DiffBIR/configs/inference/swinir.yaml")
    args.cldm_config=os.path.join (cur_path,"src/DiffBIR/configs/inference/cldm.yaml")
    args.diffusion_config=os.path.join (cur_path,"src/DiffBIR/configs/inference/diffusion_v2.1.yaml")

    args.output=out_path
    args.task=task
    args.upscale=upscale
    args.device=device
    set_seed(args.seed)
    if args.version != "custom":
        loops = {
            "sr": BSRInferenceLoop,
            "denoise": BIDInferenceLoop,
            "face": BFRInferenceLoop,
            "unaligned_face": UnAlignedBFRInferenceLoop,
        }
        model=loops[args.task](args)
    else:
        model=CustomInferenceLoop(args)
    print("init model done!")
    return model