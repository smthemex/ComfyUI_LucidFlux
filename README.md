# ComfyUI_LucidFlux
 [LucidFlux](https://github.com/W2GenAI-Lab/LucidFlux): Caption-Free Universal Image Restoration with a Large-Scale Diffusion Transformer，you can use it in ComfyUI

# Update
* 简化流程，新增2K超分功能，去掉DiffBIR v2模型； 
* Simplify the process, add 2K super-resolution function, and remove the DiffBIR v2 model

1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_LucidFlux
```
2.requirements  
----

```
pip install -r requirements.txt
```

3.checkpoints 
----
* any flux dit / 任意flux模型， KJ的 或者官方封装的
* lucid checkpoints [links](https://huggingface.co/W2GenAI/LucidFlux/tree/main) /lucidflux.pth 和prompt_embeddings.pt
* siglip512 [links](https://huggingface.co/google/siglip2-so400m-patch16-512/tree/main) / model.safetensors 只下单体模型   
* DiffBIR [links](https://huggingface.co/lxq007/DiffBIR/tree/main)  /  general_swinir_v1.ckpt   
* turbo lora [links](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha)  #optional 可选，8 步起  
* lucid_connector.pth[links](https://huggingface.co/smthem/LucidFLUX-connector)
* comfy T5 ，clip-L and flux ae   [links](https://huggingface.co/Comfy-Org/models)   #comfy T5 ，clip-L is optional / comfy T5和clip-L可选推理prompt，但是不推荐，如图例所示，直接加载emb
* ultravae[links](https://huggingface.co/Owen777/UltraFlux-v1/tree/main/vae) ,rename or not
```
├── ComfyUI/models/
|     ├── diffusion_models/any flux dit # 任意flux dit模型 ，就用kj的或者x flux的，名字要带dev 否则跑schnell
|     ├── vae/ae.safetensors #comfy 
|     ├── clip/
|        ├──clip-l.safetensors #comfy optional 可选，不推荐，如果使用prompt_embeddings.pt
|        ├──t5xxl_fp8_e4m3fn.safetensors #comfy optional 可选，不推荐，，如果使用prompt_embeddings.pt
|     ├── clip_vision/siglip2-so400m-patch16-512.safetensors  #rename from model.safetensors  最好重命名个，不然都是siglip 的model.safetensors
|     ├── LucidFlux/
|        ├──general_swinir_v1.ckpt
|        ├──lucidflux.pth
|        ├──prompt_embeddings.pt # 已适配，使用时不要连clip
|        ├── lucid_connector.pth
```

# 4 Example
*  2k
![](https://github.com/smthemex/ComfyUI_LucidFlux/blob/main/example_workflows/example2k.png)
* 1k
![](https://github.com/smthemex/ComfyUI_LucidFlux/blob/main/example_workflows/example1k.png)


# 5 Citation
------
* [FLUX ](https://github.com/black-forest-labs/flux)
* [ X-flux](https://github.com/XLabs-AI/x-flux)

 LucidFlux 
```
@article{fei2025lucidflux,
  title={LucidFlux: Caption-Free Universal Image Restoration via a Large-Scale Diffusion Transformer},
  author={Fei, Song and Ye, Tian and Wang, Lujia and Zhu, Lei},
  journal={arXiv preprint arXiv:2509.22414},
  year={2025}
}
```

