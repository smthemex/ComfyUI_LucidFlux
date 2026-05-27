
from .LucidFlux_node import LucidFlux_SM_Model,LucidFlux_SM_Diffbir,LucidFlux_SM_Pid_Model,LucidFlux_SM_Cond,LucidFlux_SM_Encode,LucidFlux_SM_KSampler,LucidFlux_SM_Decoder,LucidFlux_SM_Pid_Decoder
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

class LucidFlux_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LucidFlux_SM_Model,
            LucidFlux_SM_Diffbir,
            LucidFlux_SM_Cond,
            LucidFlux_SM_Encode,
            LucidFlux_SM_KSampler,
            LucidFlux_SM_Pid_Model,
            LucidFlux_SM_Pid_Decoder,
            LucidFlux_SM_Decoder,
        ]


async def comfy_entrypoint() -> LucidFlux_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return LucidFlux_SM_Extension()




