import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from .loop import InferenceLoop, MODELS
from ..utils.common import (
    instantiate_from_config,
    load_model_from_url,load_model_from_file,
    trace_vram_usage,
)
from ..pipeline import SwinIRPipeline
from ..model import SwinIR


class BFRInferenceLoop(InferenceLoop):
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
        config=self.args.config
        self.cleaner: SwinIR = instantiate_from_config(
            OmegaConf.load(config)
        )
        if self.args.swinir_face is None:
            weight = MODELS["swinir_face"]
            weight = load_model_from_url(weight)
        else:
            weight = load_model_from_file(self.args.swinir_face)
 
        self.cleaner.load_state_dict(weight, strict=True)
        self.cleaner.eval().to(self.args.device)

    def load_pipeline(self) -> None:
        self.pipeline = SwinIRPipeline(
            self.cleaner, self.cldm, self.diffusion, self.cond_fn, self.args.device
        )

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        lq = lq.resize(
            tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
        )
        return super().after_load_lq(lq)
