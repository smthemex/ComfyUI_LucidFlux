 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from omegaconf import OmegaConf
from .model_loader_utils import  tensor2pillist_upscale
from .inference import load_lucidflux_model,lucidflux_inference,preprocess_data,get_cond,load_condition_model
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes
import comfy.model_management as mm
from .src.flux.align_color import wavelet_reconstruction

MAX_SEED = np.iinfo(np.int32).max

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")

node_cr_path = os.path.dirname(os.path.abspath(__file__))

weigths_LucidFlux_current_path = os.path.join(folder_paths.models_dir, "LucidFlux")
if not os.path.exists(weigths_LucidFlux_current_path):
    os.makedirs(weigths_LucidFlux_current_path)
folder_paths.add_model_folder_path("LucidFlux", weigths_LucidFlux_current_path) #  LucidFlux dir

class LucidFlux_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="LucidFlux_SM_Model",
            display_name="LucidFlux_SM_Model",
            category="LucidFlux_SM",
            inputs=[
                io.Combo.Input("LucidFlux",options= ["none"] + [i for i in folder_paths.get_filename_list("LucidFlux") if "lucid" in i.lower()]),
                io.Combo.Input("diffusion_models",options= ["none"] + folder_paths.get_filename_list("diffusion_models")),
                io.Model.Input("cf_model", optional=True),
            ],
            outputs=[
                io.Custom("LucidFlux_SM").Output(),
                io.Custom("LucidFlux_SD").Output(),
                ],
            )
    @classmethod
    def execute(cls, LucidFlux,diffusion_models,cf_model=None) -> io.NodeOutput:
        is_dev="flux-dev" if "dev" in diffusion_models.lower() else "flux-schnell"
        if cf_model is not None:
            if "guidance_in.in_layer.weight" in cf_model.model.diffusion_model.state_dict().keys():
                is_dev="flux-dev"
            else:
                is_dev="flux-schnell"
            print("flux is :",is_dev)
        LucidFlux_path=folder_paths.get_full_path("LucidFlux", LucidFlux) if LucidFlux != "none" else None
        ckpt_path=folder_paths.get_full_path("diffusion_models", diffusion_models) if diffusion_models != "none" else None
        
        assert LucidFlux_path is not None,"need LucidFlux"
        origin_dict={
            "name":is_dev,
            "offload":True,
            "device":"cuda:0",
            "output_dir":folder_paths.get_output_directory(),
            "checkpoint":LucidFlux_path,
        }
        args=OmegaConf.create(origin_dict)
        model,state=load_lucidflux_model(args,ckpt_path,cf_model,device,)
        return io.NodeOutput(model,state)
    

class LucidFlux_SM_Cond(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LucidFlux_SM_Cond",
            display_name="LucidFlux_SM_Cond",
            category="LucidFlux_SM",
            inputs=[
                io.Custom("LucidFlux_SM").Input("model"),
                io.Combo.Input("lora1",options= ["none"] + folder_paths.get_filename_list("loras")),
                io.Combo.Input("lora2",options= ["none"] + folder_paths.get_filename_list("loras")),
                io.Float.Input("scale1", default=1.0, min=0.0, max=1.0, step=0.1,display_mode=io.NumberDisplay.slider),
                io.Float.Input("scale2", default=1.0, min=0.0, max=1.0, step=0.1,display_mode=io.NumberDisplay.slider),
                ],
            outputs=[io.Custom("LucidFlux_SM").Output()],
        )
    @classmethod
    def execute(cls, model,lora1,lora2,scale1,scale2) -> io.NodeOutput:
        lora1_path=folder_paths.get_full_path("loras", lora1) if lora1!="none" else None
        lora2_path=folder_paths.get_full_path("loras", lora2) if lora2!="none" else None
        lora_list=[i for i in [lora1_path,lora2_path] if i is not None]
        lora_path= lora_list if lora_list else None
        model=load_condition_model(model,lora_path,[scale1,scale2])
        return io.NodeOutput (model)


class LucidFlux_SM_Encode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LucidFlux_SM_Encode",
            display_name="LucidFlux_SM_Encode",
            category="LucidFlux_SM",
            inputs=[
                io.Custom("LucidFlux_SD").Input("state_dict"),
                io.Image.Input("image"),
                io.ClipVision.Input("CLIP_VISION"),
                io.Combo.Input("swinir",options= ["none"] + [i for i in folder_paths.get_filename_list("LucidFlux") if "swinir" in i.lower() ] ),
                io.Int.Input("width", default=1024, min=256, max=nodes.MAX_RESOLUTION,step=64,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=1024, min=256, max=nodes.MAX_RESOLUTION,step=64,display_mode=io.NumberDisplay.number),
                io.Combo.Input("emb",options= ["none"] + [i for i in folder_paths.get_filename_list("LucidFlux") if "prompt" in i.lower() ]),
                io.Conditioning.Input("positive",optional=True),     
            ],
            outputs=[
                io.Conditioning.Output(display_name="condition"),
                ],
        )
    @classmethod
    def execute(cls, state_dict,CLIP_VISION, image,swinir,width,height,emb,positive=None) -> io.NodeOutput:
        swinir_path=folder_paths.get_full_path("LucidFlux", swinir) if swinir != "none" else None
        emb_path=folder_paths.get_full_path("LucidFlux", emb) if emb != "none" else None
        #siglip_ckpt=folder_paths.get_full_path("clip_vision", siglip_ckpt) if siglip_ckpt != "none" else None
        assert swinir_path is not None ,"need swinir"
        input_pli_list=tensor2pillist_upscale(image,width,height)
        inp_cond=get_cond(positive,emb_path,height,width,device)
        postive=preprocess_data(state_dict,swinir_path,CLIP_VISION,input_pli_list, inp_cond,device)
        cf_models=mm.loaded_models()
        for model in cf_models:   
            model.unpatch_model(device_to=torch.device("cpu"))
        mm.soft_empty_cache()
        return io.NodeOutput(postive)


class LucidFlux_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LucidFlux_SM_KSampler",
            display_name="LucidFlux_SM_KSampler",
            category="LucidFlux_SM",
            inputs=[
                io.Custom("LucidFlux_SM").Input("model"),
                io.Vae.Input("vae"),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Float.Input("cfg", default=4.0, min=0.0, max=100.0, step=0.1, round=0.01,),
                io.Conditioning.Input("condition"),
            ],
            outputs=[
                io.Image.Output(display_name="Image"),
            ],
        )
    
    @classmethod
    def execute(cls, model,vae, steps,seed, cfg, condition, ) -> io.NodeOutput:
        pipe=model.get("model")
        dual_condition_branch=model.get("dual_condition_branch")
        x=lucidflux_inference(pipe,dual_condition_branch,condition,cfg,steps,seed,device,model.get("is_schnell",False)) #torch.Size([1, 16, 128, 128])
        
        images=[]
        for i ,j in zip(x,condition):
            image=vae.decode(i).squeeze(0)#torch.Size([1024, 1024, 3])

            x1 = image.clamp(-1, 1).to(device)
            #x1 = rearrange(x1[-1], "c h w -> h w c")
            #output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
            hq = wavelet_reconstruction((x1.permute(2, 0, 1) + 1.0) / 2, j.get("ci_pre_origin").squeeze(0).to(device))
            hq = hq.clamp(0, 1)
            hq=hq.unsqueeze(0).permute(0, 2, 3, 1)
            images.append(hq)
        img = torch.cat(images, dim=0)
        return io.NodeOutput(img)


from aiohttp import web
from server import PromptServer
@PromptServer.instance.routes.get("/Edit_SM_Extension")
async def get_hello(request):
    return web.json_response("Edit_SM_Extension")

class LucidFlux_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LucidFlux_SM_Model,
            LucidFlux_SM_Cond,
            LucidFlux_SM_Encode,
            LucidFlux_SM_KSampler,
        ]


async def comfy_entrypoint() -> LucidFlux_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return LucidFlux_SM_Extension()
