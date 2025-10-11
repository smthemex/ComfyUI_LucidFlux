# ComfyUI_LucidFlux
 [LucidFlux](https://github.com/W2GenAI-Lab/LucidFlux): Caption-Free Universal Image Restoration with a Large-Scale Diffusion Transformer，you can use it in ComfyUI

# Update
* 新增Diffbir节点，支持Diffbir v2.1 超分，人脸优化，由此diffbir预处理也可以变成可选，如果你输入的图片只是比较模糊的那种，如果不模糊，可以用参考节点的加模糊来精炼，当然 你可以把Diffbir节点当成独立的节点，其超分去模糊也还行；常规使用v2.1超分，推荐选择sr ，使用原生v1，则选择none；
* Added Diffbir node, supporting Diffbir v2.1 super-resolution and face optimization. As a result, Diffbir preprocessing can also be optional. If the image you input is relatively blurry, you can use the reference node’s blur addition for refinement. Of course, you can treat the Diffbir node as an independent node, and its super-resolution can also handle deblurring. For regular use of v2.1 super-resolution, it is recommended to select 'sr'; for using native v1, select 'none'.  
* 图像的小波重建功能外置，在sampler节点；
* The wavelet reconstruction function of the image is independent and located on the sampler node.  
* you can use turbo and real lora now / 支持加载加速和真实lora，工作流已替换，输入尺寸必须为64的整数倍，不一定要正方形
* 测试环境cu128+torch2.8.0， Vram 4070 12G，Ram 64G ，python3.11,同步官方prompt_embeddings代码，kj dit use  [links](https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors) ,官方dit默认要加载clip模型，不建议内存小的人用‘  

  
1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_LucidFlux
```
2.requirements  
----
* 通常不需要,v2.1 需要insightface
```
pip install -r requirements.txt
```

3.checkpoints 
----
* any flux dit / 任意flux模型， KJ的 或者官方封装的
* lucid checkpoints [links](https://huggingface.co/W2GenAI/LucidFlux/tree/main) /lucidflux.pth 和prompt_embeddings.pt
* siglip512 [links](https://huggingface.co/google/siglip2-so400m-patch16-512/tree/main) / model.safetensors 只下单体模型   
* DiffBIR [links](https://huggingface.co/lxq007/DiffBIR/tree/main)  /  general_swinir_v1.ckpt   or v2 [links](https://huggingface.co/lxq007/DiffBIR-v2) 体验全部功能 最好模型全下
* turbo lora [links](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha)  #optional 可选，8 步起  
* comfy T5 ，clip-L and flux ae   [links](https://huggingface.co/Comfy-Org/models)   #comfy T5 ，clip-L is optional / comfy T5和clip-L可选，如图例所示，直接加载emb
```
├── ComfyUI/models/
|     ├── diffusion_models/any flux dit # 任意flux dit模型 ，就用kj的或者x flux的，名字要带dev 否则跑schnell
|     ├── vae/ae.safetensors #comfy 
|     ├── clip/
|        ├──clip-l.safetensors #comfy optional 可选，如果使用prompt_embeddings.pt
|        ├──t5xxl_fp8_e4m3fn.safetensors #comfy optional可选，如果使用prompt_embeddings.pt
|     ├── clip_vision/siglip2-so400m-patch16-512.safetensors  #rename from model.safetensors  最好重命名个，不然都是siglip 的model.safetensors
|     ├── LucidFlux/
|        ├──general_swinir_v1.ckpt
|        ├──lucidflux.pth
|        ├──prompt_embeddings.pt # 已适配，使用时不要连clip
|        ├── v2.1 ...
```

# 4 Example
*  not use diffbir  
![](https://github.com/smthemex/ComfyUI_LucidFlux/blob/main/example_workflows/example_nodif.png)
* use embeddings to save VRAM  and diffbir_v2
![](https://github.com/smthemex/ComfyUI_LucidFlux/blob/main/example_workflows/example.png)
* use clip to encoder prompt  
![](https://github.com/smthemex/ComfyUI_LucidFlux/blob/main/example_workflows/example1007.png)

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
DreamClear
```
@article{ai2024dreamclear,
    title={DreamClear: High-Capacity Real-World Image Restoration with Privacy-Safe Dataset Curation},
    author={Ai, Yuang and Zhou, Xiaoqiang and Huang, Huaibo and Han, Xiaotian and Chen, Zhengyu and You, Quanzeng and Yang, Hongxia},
    journal={Advances in Neural Information Processing Systems},
    volume={37},
    pages={55443--55469},
    year={2024}
}
```
diffbir
```
@misc{lin2024diffbir,
      title={DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior}, 
      author={Xinqi Lin and Jingwen He and Ziyan Chen and Zhaoyang Lyu and Bo Dai and Fanghua Yu and Wanli Ouyang and Yu Qiao and Chao Dong},
      year={2024},
      eprint={2308.15070},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
