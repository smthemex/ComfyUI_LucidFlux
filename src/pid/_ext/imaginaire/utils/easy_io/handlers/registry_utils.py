# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .base import BaseFileHandler
from .byte_handler import ByteHandler
from .csv_handler import CsvHandler
from .gzip_handler import GzipHandler
from .imageio_video_handler import ImageioVideoHandler
from .json_handler import JsonHandler
from .jsonl_handler import JsonlHandler
from .np_handler import NumpyHandler
from .pandas_handler import PandasHandler
from .pickle_handler import PickleHandler
from .pil_handler import PILHandler
from .tarfile_handler import TarHandler
from .torch_handler import TorchHandler
from .torchjit_handler import TorchJitHandler
from .txt_handler import TxtHandler
from .yaml_handler import YamlHandler

file_handlers = {
    "json": JsonHandler(),
    "yaml": YamlHandler(),
    "yml": YamlHandler(),
    "pickle": PickleHandler(),
    "pkl": PickleHandler(),
    "tar": TarHandler(),
    "jit": TorchJitHandler(),
    "npy": NumpyHandler(),
    "txt": TxtHandler(),
    "csv": CsvHandler(),
    "pandas": PandasHandler(),
    "gz": GzipHandler(),
    "jsonl": JsonlHandler(),
    "byte": ByteHandler(),
}

for torch_type in ["pt", "pth", "ckpt"]:
    file_handlers[torch_type] = TorchHandler()
for img_type in ["jpg", "jpeg", "png", "bmp", "gif"]:
    file_handlers[img_type] = PILHandler()
    file_handlers[img_type].format = img_type
try:
    from .trimesh_handler import TrimeshHandler

    for mesh_type in ["ply", "stl", "obj", "glb"]:
        file_handlers[mesh_type] = TrimeshHandler()
        file_handlers[mesh_type].format = mesh_type
except ImportError:
    pass
for video_type in ["mp4", "avi", "mov", "webm", "flv", "wmv"]:
    file_handlers[video_type] = ImageioVideoHandler()


def _register_handler(handler, file_formats):
    """Register a handler for some file extensions.

    Args:
        handler (:obj:`BaseFileHandler`): Handler to be registered.
        file_formats (str or list[str]): File formats to be handled by this
            handler.
    """
    if not isinstance(handler, BaseFileHandler):
        raise TypeError(f"handler must be a child of BaseFileHandler, not {type(handler)}")
    if isinstance(file_formats, str):
        file_formats = [file_formats]
    if not all([isinstance(item, str) for item in file_formats]):
        raise TypeError("file_formats must be a str or a list of str")
    for ext in file_formats:
        file_handlers[ext] = handler


def register_handler(file_formats, **kwargs):
    def wrap(cls):
        _register_handler(cls(**kwargs), file_formats)
        return cls

    return wrap
