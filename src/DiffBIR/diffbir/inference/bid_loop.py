import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from .loop import InferenceLoop, MODELS
from ..utils.common import (
    instantiate_from_config,
    load_model_from_url,load_model_from_file,
    trace_vram_usage,
)
from ..pipeline import (
    SwinIRPipeline,
    SCUNetPipeline,
)
from ..model import SwinIR, SCUNet


class BIDInferenceLoop(InferenceLoop):
    def to(self, device,dtype=None):
        if dtype is not None:
            self.diffusion.to(device,dtype)
            self.cldm.to(device,dtype)
            self.cleaner.to(device,dtype) 
            if self.cond_fn is not None:
                self.cond_fn.to(device,dtype)
        else:
            self.diffusion.to(device)
            self.cldm.to(device)
            self.cleaner.to(device)
            if self.cond_fn is not None:
                self.cond_fn.to(device)

    def load_cleaner(self) -> None:
        if self.args.version == "v1":
            config = "configs/inference/swinir.yaml"
            weight = MODELS["swinir_general"]
        elif self.args.version == "v2":
            config = "configs/inference/scunet.yaml"
            weight = MODELS["scunet_psnr"]
        else:
            #config = "configs/inference/swinir.yaml"
            config=self.args.config
            weight=self.args.swinir_realesrgan
           # weight = MODELS["swinir_realesrgan"]
        self.cleaner: SCUNet | SwinIR = instantiate_from_config(OmegaConf.load(config))
        if weight is  None:
            weight = MODELS["swinir_realesrgan"]
            model_weight = load_model_from_url(weight)
        else:
            model_weight = load_model_from_file(weight)

        self.cleaner.load_state_dict(model_weight, strict=True)
        self.cleaner.eval().to(self.args.device)

    def load_pipeline(self) -> None:
        if self.args.version == "v1" or self.args.version == "v2.1":
            pipeline_class = SwinIRPipeline
        else:
            pipeline_class = SCUNetPipeline
        self.pipeline = pipeline_class(
            self.cleaner,
            self.cldm,
            self.diffusion,
            self.cond_fn,
            self.args.device,
        )

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        lq = lq.resize(
            tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
        )
        return super().after_load_lq(lq)
