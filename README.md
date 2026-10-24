# ComfyUI_LucidFlux（comfy 原生 DiT 链版本）

[LucidFlux](https://github.com/W2GenAI-Lab/LucidFlux): Caption-Free Universal Image Restoration with a Large-Scale Diffusion Transformer，you can use it in ComfyUI

# Update
*  更改封装方式，使用comfyUI的核心复现方法/ change the encapsulation method to use the core method of comfyUI

1.Installation  
-----
In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_LucidFlux
```
2.checkpoints 
2.1 any flux dit / 任意flux模型， KJ的 或者官方封装的  
2.2 lucid checkpoints/ [links](https://huggingface.co/W2GenAI/LucidFlux/tree/main) /lucidflux.pth 和prompt_embeddings.pt 和gemma_prompt_embedding.pt  
2.3 siglip512 [links](https://huggingface.co/google/siglip2-so400m-patch16-512/tree/main) / model.safetensors 只下单体模型   
2.4 DiffBIR [links](https://huggingface.co/lxq007/DiffBIR/tree/main)  /  general_swinir_v1.ckpt   
2.5 turbo lora [links](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha)  #optional 可选，8 步起  
2.6 lucid_connector.pth[links](https://huggingface.co/smthem/LucidFLUX-connector)  
2.7 ultravae[links](https://huggingface.co/Owen777/UltraFlux-v1/tree/main/vae) ,rename or not  
2.8 PID[links](https://huggingface.co/nvidia/PiD)  /  PiD_res2kto4k_sr4x_official_flux_distill_4step.pth or PiD_res2k_sr4x_official_flux_distill_4step.pth  
2.9 null_caption_embs.pt[links](https://huggingface.co/smthem/LucidFLUX-connector/tree/main)  #PID use  

```
ComfyUI/models/
├── diffusion_models/flux1-dev-fp8.safetensors      # 任意 flux-dev（bf16/fp8 均可）
├── PiD_v1pt5_res2kto4k_sr4x_official_flux_distill_4step.pth      # v1.5 的 4K 模型（使用PID必需）
├── clip_vision/siglip2-so400m-patch16-512.safetensors
├── vae/UltraFlux.safetensors                       # 2K 用；1K 可换 ae.safetensors
└── LucidFlux/
    ├── lucidflux.pth                               # 双条件分支（必需）
    ├── lucid_connector.pth                         # SigLIP→4096 的 redux connector（必需）
    ├── general_swinir_v1.ckpt                      # DiffBIR 先验（必需）
    ├── prompt_embeddings.pt                        # 官方预计算文本 embedding（**必需**，替代 CLIP/T5）
    └── gemma_prompt_embedding.pt / null_caption_embs.pt   # 仅 PiD 阶段用
```

# 3 Example
* 256 to 4K
![](https://github.com/smthemex/ComfyUI_LucidFlux/blob/main/example_workflows/example.png)


# 4 Citation
------
* [FLUX ](https://github.com/black-forest-labs/flux)
* [X-flux](https://github.com/XLabs-AI/x-flux)
* [PID](https://github.com/nv-tlabs/PiD)

LucidFlux 
```
@article{fei2025lucidflux,
  title={LucidFlux: Caption-Free Universal Image Restoration via a Large-Scale Diffusion Transformer},
  author={Fei, Song and Ye, Tian and Wang, Lujia and Zhu, Lei},
  journal={arXiv preprint arXiv:2509.22414},
  year={2025}
}
```
