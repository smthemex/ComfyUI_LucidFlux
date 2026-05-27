 # !/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import torch
import os
from omegaconf import OmegaConf
from .model_loader_utils import  tensor2pillist_upscale,clear_comfyui_cache,tensor2list
from .inference import (load_lucidflux_model,lucidflux_inference,preprocess_data,get_cond,load_condition_model,load_pid_model,infer_pid_decode,
                        load_diffbir_model,infer_diffbir_model,normal_vae_decode)
import folder_paths
from comfy_api.latest  import io
import nodes


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
                io.Boolean.Input("block_offload", default=True),
                io.Combo.Input("model_type",options= ["bf16","f32"] ),
                io.Model.Input("cf_model", optional=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, LucidFlux,diffusion_models,block_offload,model_type,cf_model=None) -> io.NodeOutput:
        clear_comfyui_cache()
        model_dtype = torch.bfloat16 if model_type == 'bf16' else torch.float32
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
        model,dual_condition_branch=load_lucidflux_model(args,ckpt_path,cf_model,block_offload,device,model_dtype)
        model.block_offload=block_offload
        model.is_schnell = is_dev=="flux-schnell" 
        return io.NodeOutput({"model": model,  "dual_condition_branch": dual_condition_branch, })
    

class LucidFlux_SM_Diffbir(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LucidFlux_SM_Diffbir",
            display_name="LucidFlux_SM_Diffbir",
            category="LucidFlux_SM",
            inputs=[
                io.Combo.Input("swinir",options= ["none"] + folder_paths.get_filename_list("LucidFlux") ),
                io.Image.Input("image"),
                io.Int.Input("width", default=1024, min=256, max=nodes.MAX_RESOLUTION,step=64,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=1024, min=256, max=nodes.MAX_RESOLUTION,step=64,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("infer_2k", default=False),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.Image.Output(display_name="image"),
                ],
            )
    @classmethod
    def execute(cls,swinir, image,width,height,infer_2k) -> io.NodeOutput:
        clear_comfyui_cache()
        swinir_path=folder_paths.get_full_path("LucidFlux", swinir) if swinir != "none" else None
        model=load_diffbir_model(swinir_path,)
        input_pli_list=tensor2pillist_upscale(image,width// 2,height// 2) if infer_2k else tensor2pillist_upscale(image,width,height)
        images,condition_cond_list,condition_cond_ldr_list=infer_diffbir_model(model, input_pli_list,device,)
        conditioning = {"condition_cond_list": condition_cond_list,"condition_cond_ldr_list":condition_cond_ldr_list, "images": images,"height":height,"width":width,"infer_2k":infer_2k}
        return io.NodeOutput(conditioning,torch.cat(images,dim=0).float().cpu())


class LucidFlux_SM_Cond(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LucidFlux_SM_Cond",
            display_name="LucidFlux_SM_Cond",
            category="LucidFlux_SM",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora",options= ["none"] + folder_paths.get_filename_list("loras")),
                io.Float.Input("scale", default=1.0, min=0.0, max=1.0, step=0.1,display_mode=io.NumberDisplay.number),
                ],
            outputs=[io.Model.Output(display_name="model")],
        )
    @classmethod
    def execute(cls, model,lora,scale) -> io.NodeOutput:
        lora_path=folder_paths.get_full_path("loras", lora) if lora!="none" else None
        model=load_condition_model(model,lora_path,scale)
        return io.NodeOutput (model)


class LucidFlux_SM_Encode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LucidFlux_SM_Encode",
            display_name="LucidFlux_SM_Encode",
            category="LucidFlux_SM",
            inputs=[
                io.ClipVision.Input("CLIP_VISION"),
                io.Conditioning.Input("conditioning"),#  B H W C C=3
                io.Combo.Input("emb",options= ["none"] + [i for i in folder_paths.get_filename_list("LucidFlux") if "prompt" in i.lower() ]),
                io.Combo.Input("connector",options= ["none"] + [i for i in folder_paths.get_filename_list("LucidFlux") if "connector" in i.lower() ]),
                io.Combo.Input("model_type",options= ["bf16","f32"] ),
                io.Conditioning.Input("positive",optional=True),     
            ],
            outputs=[
                io.Conditioning.Output(display_name="condition"),
                ],
        )
    @classmethod
    def execute(cls, CLIP_VISION, conditioning,emb,connector,model_type,positive=None) -> io.NodeOutput:
        model_dtype = torch.bfloat16 if model_type == 'bf16' else torch.float32
        clear_comfyui_cache()
        emb_path=folder_paths.get_full_path("LucidFlux", emb) if emb != "none" else None
        connector_path=folder_paths.get_full_path("LucidFlux", connector) if connector != "none" else None
        height,width=conditioning["height"],conditioning["width"]
        inp_cond=get_cond(positive,emb_path,height,width,device,conditioning["infer_2k"])      
        data_list=preprocess_data(connector_path,CLIP_VISION,conditioning, inp_cond,model_dtype,device)
        clear_comfyui_cache()
        conditioning["data_list"]=data_list
        conditioning["model_dtype"]=model_type
        return io.NodeOutput(conditioning)


class LucidFlux_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LucidFlux_SM_KSampler",
            display_name="LucidFlux_SM_KSampler",
            category="LucidFlux_SM",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Float.Input("cfg", default=4.0, min=0.0, max=100.0, step=0.1, round=0.01,),
                io.Conditioning.Input("condition"),
        
            ],
            outputs=[
                io.Latent.Output(display_name="Latent"),
            ],
        )
    @classmethod
    def execute(cls, model, steps,seed, cfg, condition, ) -> io.NodeOutput:
        dual_condition_model=model["dual_condition_branch"]
        
        model=model["model"]
        if not  model.block_offload:
            model.to(device)
        x=lucidflux_inference(model,dual_condition_model,condition,cfg,steps,seed,device,model.is_schnell) #torch.Size([1, 16, 128, 128])
        output={"samples":x,"images":condition["images"],"infer_2k":condition["infer_2k"]}
        return io.NodeOutput(output)

class LucidFlux_SM_Pid_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="LucidFlux_SM_Pid_Model",
            display_name="LucidFlux_SM_Pid_Model",
            category="LucidFlux_SM",
            inputs=[
                io.Combo.Input("pid_model",options= ["none"] + folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("model_type",options= ["bf16","f32"] ),
                io.Combo.Input("pid_type",options= ["2k","2kto4k"] ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, pid_model,model_type,pid_type,) -> io.NodeOutput:
        clear_comfyui_cache()
        model_dtype = torch.bfloat16 if model_type == 'bf16' else torch.float32
        pid_path=folder_paths.get_full_path("diffusion_models", pid_model) if pid_model != "none" else None

        model=load_pid_model(pid_path,model_dtype,pid_type,weigths_LucidFlux_current_path)
        model.model.to("cpu")
       
        return io.NodeOutput(model)
    
class LucidFlux_SM_Pid_Decoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LucidFlux_SM_Pid_Decoder",
            display_name="LucidFlux_SM_Pid_Decoder",
            category="LucidFlux_SM",
            inputs=[ 
                io.Model.Input("pid_model"),
                io.Latent.Input("latent"),
                io.Int.Input("steps", default=4, min=1, max=10000),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01,),
                io.Float.Input("degrade_sigma", default=0.0, min=0.0, max=1.0, step=0.1, round=0.01,),
                io.Float.Input("pid_color_align_strength", default=0.6, min=0.1, max=1.0, step=0.1, round=0.01,),
                io.Int.Input("streaming_prefetch_count", default=1, min=0, max=MAX_SEED),
                io.Boolean.Input("wavelet", default=True),
                io.Image.Input("images",optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )
    @classmethod
    def execute(cls,pid_model, latent,steps,seed,cfg,degrade_sigma,pid_color_align_strength,streaming_prefetch_count,wavelet,images=None) -> io.NodeOutput:
        if streaming_prefetch_count==0:
            pid_model.model.to(device)
            streaming_prefetch_count=None
        if images is not None:
            images=tensor2list(images)
        clear_comfyui_cache()
        image=infer_pid_decode(pid_model,latent,cfg,steps,seed,degrade_sigma,wavelet,pid_color_align_strength,streaming_prefetch_count,images)
        del pid_model
        torch.cuda.empty_cache()
        return io.NodeOutput(image)



class LucidFlux_SM_Decoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LucidFlux_SM_Decoder",
            display_name="LucidFlux_SM_Decoder",
            category="LucidFlux_SM",
            inputs=[ 
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae")),
                io.Latent.Input("latent"),
                io.Boolean.Input("wavelet", default=True),
                io.Vae.Input("ae",optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )
    @classmethod
    def execute(cls,vae, latent,wavelet,ae=None,) -> io.NodeOutput:
        
        images=normal_vae_decode(vae, latent,wavelet,ae,device)
        return io.NodeOutput(images)

