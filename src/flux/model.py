from dataclasses import dataclass

import torch
from torch import Tensor, nn
from einops import rearrange

from .modules.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)


class BlockGPUManager:
    def __init__(self, device="cuda",):
        self.device = device
        self.managed_modules = []
        self.embedder_modules = []
        self.output_modules = []  
        self.fp8_scale=1  
        # 跟踪哪些blocks当前在GPU上
        self.block_types = {} 
        self.blocks_on_gpu = set()
    
    def get_gpu_memory_usage(self):
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
            total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
            free = total - allocated
            
            return {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'total_mb': total,
                'free_mb': free
            }
        return None
    
    def setup_for_inference(self, transformer_model,):
        self._collect_managed_modules(transformer_model)
        self._initialize_embedder_modules()
        self._initialize_output_modules()
        return self
    
    def _has_fp8_parameters(self, module: nn.Module) -> bool:
        """检查模块是否包含FP8参数或缓冲区（带缓存）"""
        if hasattr(module, '_has_fp8_cached'):
            return module._has_fp8_cached
        
        has_fp8 = False
        for param in module.parameters(recurse=True):
            if param.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                has_fp8 = True
                break
        
        module._has_fp8_cached = has_fp8
        return has_fp8
    
    def _deep_convert_fp8_on_cpu(self, module: nn.Module):
        """在CPU上深度转换所有FP8参数"""
        if not self._has_fp8_parameters(module):
            return
        params_to_convert = []
        for submodule in module.modules():
            for param in submodule.parameters(recurse=False):
                if param.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    params_to_convert.append(param)
        
        # 批量执行转换
        with torch.no_grad():
            for param in params_to_convert:
                param.data = param.to(torch.bfloat16) * self.fp8_scale
                

    def _collect_managed_modules(self, transformer_model):
        """收集所有要管理的模块（简化版本，使用已知的block大小）"""
        self.managed_modules = []
        self.embedder_modules = []
        self.output_modules = []
        self.block_types = {}

        # 收集dual blocks (double_blocks)
        for i, block in enumerate(transformer_model.double_blocks):
            self.managed_modules.append(block)
            self.block_types[i] = 'dual'
        
        # 收集single blocks (single_blocks)
        single_start_idx = len(self.managed_modules)
        for i, block in enumerate(transformer_model.single_blocks):
            self.managed_modules.append(block)
            self.block_types[single_start_idx + i] = 'single'
        
        # 收集embedder和output模块
        if hasattr(transformer_model, 'pe_embedder'):
            self.embedder_modules.append(transformer_model.pe_embedder)
        
        if hasattr(transformer_model, 'img_in'):
            self.embedder_modules.append(transformer_model.img_in)
        
        if hasattr(transformer_model, 'time_in'):
            self.embedder_modules.append(transformer_model.time_in)
        
        if hasattr(transformer_model, 'vector_in'):
            self.embedder_modules.append(transformer_model.vector_in)
        
        if hasattr(transformer_model, 'guidance_in'):
            self.embedder_modules.append(transformer_model.guidance_in)

        if hasattr(transformer_model, 'txt_in'):
            self.embedder_modules.append(transformer_model.txt_in)

        if hasattr(transformer_model, 'final_layer'):
            self.output_modules.append(transformer_model.final_layer)

  
    def _initialize_embedder_modules(self):
        """初始化embedder模块，将它们移到GPU"""
        for module in self.embedder_modules:
            # 先转换FP8参数
            if self._has_fp8_parameters(module):
                self._deep_convert_fp8_on_cpu(module)

            if hasattr(module, 'to'):
                module.to(self.device, non_blocking=True)
        return self
    
    def _initialize_output_modules(self):
        """初始化output模块，将它们移到GPU"""
        for module in self.output_modules:
            # 先转换FP8参数
            if self._has_fp8_parameters(module):
                self._deep_convert_fp8_on_cpu(module)

            if hasattr(module, 'to'):
                module.to(self.device, non_blocking=True)
        return self

    def unload_all_blocks_to_cpu(self):
        """卸载所有block到CPU"""
        #print(f"[GPU Manager] 卸载所有block到CPU")
        
        # 将所有模块移到CPU
        for i, module in enumerate(self.managed_modules):
            if hasattr(module, 'to'):
                module.to('cpu')
        
        for module in self.embedder_modules:
            if hasattr(module, 'to'):
                module.to('cpu')
        
        for module in self.output_modules:
            if hasattr(module, 'to'):
                module.to('cpu')
        
        # 清空GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """
    _supports_gradient_checkpointing = True

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.gradient_checkpointing = False
        self._gc_use_reentrant = False 

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @property
    def attn_processors(self):
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        block_controlnet_hidden_states=None,
        guidance: Tensor   = None,
        image_proj: Tensor   = None, 
        ip_scale: float = 1.0, 
        gpu_manager=None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        #original_device = img.device
        img = self.img_in(img)
        vec_=timestep_embedding(timesteps, 256)
        if gpu_manager is not None:
            if vec_.device != gpu_manager.device:
                vec_ = vec_.to(gpu_manager.device)
        vec = self.time_in(vec_)

        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            guidance=timestep_embedding(guidance, 256)
            if gpu_manager is not None:
                if guidance.device != gpu_manager.device:
                    guidance = guidance.to(gpu_manager.device)

            vec_ = self.guidance_in(guidance) 
            vec = vec + vec_


        vec = vec + self.vector_in(y) 

        txt = self.txt_in(txt) 

        ids = torch.cat((txt_ids, img_ids), dim=1)

        pe = self.pe_embedder(ids) 
        if block_controlnet_hidden_states is not None:
            controlnet_depth = len(block_controlnet_hidden_states)
        # Double-stream blocks
        # ------ Inside double_blocks loop ------
        for index_block, block in enumerate(self.double_blocks):
            if gpu_manager is not None:
                # 加载当前block到GPU
                if index_block < len(self.double_blocks):
                    module = gpu_manager.managed_modules[index_block]
                    if hasattr(module, 'to'):
                        if gpu_manager._has_fp8_parameters(module):
                            gpu_manager._deep_convert_fp8_on_cpu(module)
                        module.to(gpu_manager.device)
                        #print(f"[GPU Manager] 加载dual block {index_block}到GPU")
                
                # 卸载上一个block（如果是第一个block，则不需要卸载）
                if index_block > 0 and (index_block - 1) < len(self.double_blocks):
                    prev_module = gpu_manager.managed_modules[index_block - 1]
                    if hasattr(prev_module, 'to'):
                        prev_module.to('cpu')
                        #print(f"[GPU Manager] 卸载dual block {index_block-1}到CPU")
            if self.training and self.gradient_checkpointing:

                # Bind _block=block as default arg to avoid late binding
                def _double_fwd(img_, txt_, vec_, pe_,
                                image_proj_=image_proj, ip_scale_=ip_scale, _block=block):
                    out_img, out_txt = _block(
                        img=img_, txt=txt_, vec=vec_, pe=pe_,
                        image_proj=image_proj_, ip_scale=ip_scale_
                    )
                    return out_img, out_txt

                img, txt = torch.utils.checkpoint.checkpoint(
                    _double_fwd, img, txt, vec, pe, use_reentrant=self._gc_use_reentrant
                )
            else:
                img, txt = block(
                    img=img, txt=txt, vec=vec, pe=pe,
                    image_proj=image_proj, ip_scale=ip_scale
                )
            

            if block_controlnet_hidden_states is not None:
                # print("len", len(block_controlnet_hidden_states), index_block)
                img = img + block_controlnet_hidden_states[index_block]

        # ------ Inside single_blocks loop ------
        img = torch.cat((txt, img), dim=1)

        if gpu_manager is not None and len(self.double_blocks) > 0:
            last_dual_idx = len(self.double_blocks) - 1
            if last_dual_idx < len(gpu_manager.managed_modules):
                module = gpu_manager.managed_modules[last_dual_idx]
                if hasattr(module, 'to'):
                    module.to('cpu')
        for block_index,block in enumerate(self.single_blocks):
            if gpu_manager is not None:
                # 计算当前single block在managed_modules中的索引（从19开始）
                single_block_idx = len(self.double_blocks) + block_index
                
                # 加载当前single block到GPU
                if single_block_idx < len(gpu_manager.managed_modules):
                    module = gpu_manager.managed_modules[single_block_idx]
                    if hasattr(module, 'to'):
                        if gpu_manager._has_fp8_parameters(module):
                            gpu_manager._deep_convert_fp8_on_cpu(module)
                        module.to(gpu_manager.device)
                        #print(f"[GPU Manager] 加载single block {block_index} (全局索引 {single_block_idx})到GPU")
                
                # 卸载上一个single block（如果是第一个single block，则不需要卸载）
                if block_index > 0:
                    prev_single_idx = len(self.double_blocks) + (block_index - 1)
                    if prev_single_idx < len(gpu_manager.managed_modules):
                        prev_module = gpu_manager.managed_modules[prev_single_idx]
                        if hasattr(prev_module, 'to'):
                            prev_module.to('cpu')
                            #print(f"[GPU Manager] 卸载single block {block_index-1} (全局索引 {prev_single_idx})到CPU")
            if self.training and self.gradient_checkpointing:

                # Same binding trick for _block=block
                def _single_fwd(img_, vec_, pe_, _block=block):
                    return _block(img_, vec=vec_, pe=pe_)

                img = torch.utils.checkpoint.checkpoint(
                    _single_fwd, img, vec, pe, use_reentrant=self._gc_use_reentrant
                )
            else:
                img = block(img, vec=vec, pe=pe)

        # strip text tokens
        img = img[:, txt.shape[1]:, ...]

        img = self.final_layer(img, vec)
        return img
