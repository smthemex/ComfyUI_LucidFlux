# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import os

import torch
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf
from ..._ext.imaginaire.checkpointer.dcp import DefaultLoadPlanner, DistributedCheckpointer, ModelWrapper
from ..._ext.imaginaire.lazy_config import instantiate
from ..._ext.imaginaire.utils import log, misc
from ..._ext.imaginaire.utils.config_helper import get_config_module, override
from ..._ext.imaginaire.utils.easy_io import easy_io
from ..models.pid_distill_model import PidDistillModel, PidDistillModelConfig
from ..networks.pid_net import PidNet
from ..modules.conditioner import CaptionStringDrop, LQTensorDrop,PidConditioner

def load_model_from_checkpoint(
    checkpoint_path,
    caption_embeddings_path,
    enable_fsdp=False,
    strict=True,
    dtype=torch.bfloat16,
):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    conditioner = PidConditioner(
            caption=CaptionStringDrop(
                input_key="caption",
                output_key="caption",
                dropout_rate=0.1,
            ),
            lq_video_or_image=LQTensorDrop(
                input_key="LQ_video_or_image",
                output_key="lq_video_or_image",
                dropout_rate=0.1,
                is_primary=True,
            ),
            lq_latent=LQTensorDrop(
                input_key="LQ_latent",
                output_key="lq_latent",
                dropout_rate=0.1,
                is_primary=False,
            )
        )
    net_instance = PidNet(
        # T2I backbone args
        in_channels=3,
        num_groups=24,
        hidden_size=1536,
        pixel_hidden_size=16,
        pixel_attn_hidden_size=1152,
        pixel_num_groups=16,
        patch_depth=14,
        pixel_depth=2,
        num_text_blocks=4,
        patch_size=16,
        txt_embed_dim=2304,
        txt_max_length=300,
        use_text_rope=True,
        text_rope_theta=10000.0,
        rope_mode='ntk_aware',
        rope_ref_h=1024,
        rope_ref_w=1024,
        repa_encoder_index=6,
        # SR-specific defaults
        lq_inject_mode="controlnet",
        lq_in_channels=0,
        lq_latent_channels=16,
        lq_hidden_dim=512,
        lq_num_res_blocks=4,
        lq_gate_type="sigma_aware_per_token_per_dim",
        lq_interval=2,
        zero_init_lq=True,
        train_lq_proj_only=False,
        sr_scale=4,
        # PiT LQ injection
        pit_lq_inject=False,
        pit_lq_gate_type="sigma_aware_per_token_per_dim",
    )
    
    model = PidDistillModel(
        PidDistillModelConfig(
            net=net_instance,
            precomputed_caption_embeddings_path=caption_embeddings_path,
            conditioner=conditioner,
        )
    )
    model.on_train_start()
    log.info(f"Loading model from consolidated checkpoint {checkpoint_path}")
    sd= easy_io.load(checkpoint_path)
    model.load_state_dict(sd, strict=strict,)
    del sd
    #print(model.precision)
    # if not enable_fsdp:
    model = model.to(dtype=dtype)

    torch.cuda.empty_cache()
    return model



def load_model_from_checkpoint_(
    experiment_name,
    checkpoint_path,
    config_file="pid/_src/configs/pid/config.py",
    enable_fsdp=False,
    instantiate_ema=True,
    load_ema_to_reg=False,
    seed=0,
    experiment_opts: list[str] = [],
    strict=True,
):
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={experiment_name}"] + experiment_opts)

    if instantiate_ema is False and hasattr(config.model.config, "ema") and config.model.config.ema.enabled:
        config.model.config.ema.enabled = False

    config.validate()
    config.freeze()  # type: ignore
    misc.set_random_seed(seed=seed, by_rank=True)
    torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
    torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True

    if not enable_fsdp and hasattr(config.model.config, "fsdp_shard_size"):
        config.model.config.fsdp_shard_size = 1

    with misc.timer("instantiate model"):
        model = instantiate(config.model) #.cuda()
        model.on_train_start()

    if checkpoint_path.endswith(".pth"):
        log.info(f"Loading model from consolidated checkpoint {checkpoint_path}")
        model.load_state_dict(easy_io.load(checkpoint_path), strict=strict)
    else:
        log.info(f"Loading model from dcp checkpoint {checkpoint_path}")
        checkpointer = DistributedCheckpointer(config.checkpoint, config.job, callbacks=None, disable_async=True)
        cur_key_ckpt_full_path = os.path.join(checkpoint_path, "model")
        storage_reader = checkpointer.get_storage_reader(cur_key_ckpt_full_path)

        _model_wrapper = ModelWrapper(model, load_ema_to_reg=load_ema_to_reg)
        _state_dict = _model_wrapper.state_dict()
        dcp.load(
            _state_dict,
            storage_reader=storage_reader,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        _model_wrapper.load_state_dict(_state_dict)

    if not enable_fsdp:
        model = model.to(dtype=model.precision)

    torch.cuda.empty_cache()

    return model, config
