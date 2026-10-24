"""ComfyUI_LucidFlux — ComfyUI 原生 DiT 推理栈（不再依赖 diffusers / transformers）。

链路只用到 comfy 原生能力 + torch：
    UNETLoader(flux1-dev) → LucidFluxCondition(control) → LucidFluxSampler
    VAELoader(UltraFlux) → LucidFluxVAEPatch → VAEDecode → LucidFluxColorAlign
    可选 4K：LucidFluxSampler.latent → LucidFluxPiDDecode

旧的 diffusers wrapper（src/、inference.py、LucidFlux_node.py）已移除。
"""
import os

import folder_paths

_MODELS_DIR = os.path.join(folder_paths.models_dir, "LucidFlux")
if not os.path.isdir(_MODELS_DIR):
    os.makedirs(_MODELS_DIR, exist_ok=True)
if "LucidFlux" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("LucidFlux", _MODELS_DIR)

from .lucidflux_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
