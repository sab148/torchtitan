# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.models.llama3.parallelize import disable_fsdp_gradient_division
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def parallelize_flux2(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    del model_converters, parallelism, dump_folder

    if ac_config.mode != "none":
        apply_ac(model, ac_config)

    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "FLUX.2 context parallel is not implemented in this TorchTitan integration."
        )

    if compile_config.enable and "model" in compile_config.components:
        apply_compile(model, compile_config)

    if parallel_dims.fsdp_enabled:
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(names)
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            cpu_offload=training.enable_cpu_offload,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the FLUX.2 model")
        else:
            logger.info("Applied FSDP to the FLUX.2 model")

    return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    sharded_modules = [
        model.img_in,
        model.time_in,
        model.txt_in,
        model.double_stream_modulation_img,
        model.double_stream_modulation_txt,
        model.single_stream_modulation,
    ]
    if hasattr(model, "guidance_in"):
        sharded_modules.append(model.guidance_in)

    for module in sharded_modules:
        fully_shard(module, **fsdp_config)

    for block in model.double_blocks:
        fully_shard(block, **fsdp_config)

    for block in model.single_blocks:
        fully_shard(block, **fsdp_config)

    fully_shard(model.final_layer, **fsdp_config, reshard_after_forward=False)
    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    for block in model.double_blocks:
        block.compile(backend=compile_config.backend, fullgraph=True)

    for block in model.single_blocks:
        block.compile(backend=compile_config.backend, fullgraph=True)

    logger.info("Compiling each FLUX.2 transformer block with torch.compile")


def apply_ac(model: nn.Module, ac_config: ActivationCheckpointConfig) -> None:
    for layer_id, block in model.double_blocks.named_children():
        wrapped = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
        model.double_blocks.register_module(layer_id, wrapped)

    for layer_id, block in model.single_blocks.named_children():
        wrapped = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
        model.single_blocks.register_module(layer_id, wrapped)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the FLUX.2 model")


def _get_submodule_or_none(module: nn.Module, path: str) -> nn.Module | None:
    current = module
    for name in path.split("."):
        if not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current if isinstance(current, nn.Module) else None


def _find_transformer_blocks(module: nn.Module) -> tuple[str, nn.ModuleList] | None:
    preferred_paths = (
        "model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "model.model.layers",
        "transformer.h",
        "encoder.block",
    )
    for path in preferred_paths:
        candidate = _get_submodule_or_none(module, path)
        if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
            return path, candidate

    candidates: list[tuple[str, nn.ModuleList]] = []
    for name, submodule in module.named_modules():
        if not isinstance(submodule, nn.ModuleList) or len(submodule) == 0:
            continue
        lowered = name.lower()
        if any(token in lowered for token in ("layers", "blocks", "block", ".h", "h.")):
            candidates.append((name, submodule))

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            len(item[1]),
            sum(p.numel() for p in item[1].parameters()),
        ),
        reverse=True,
    )
    return candidates[0]


def parallelize_text_encoder(
    text_encoder: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
) -> nn.Module:
    if not parallel_dims.dp_shard_enabled:
        return text_encoder

    model_root = getattr(text_encoder, "model", None)
    if not isinstance(model_root, nn.Module):
        model_root = getattr(text_encoder, "hf_module", None)
    if not isinstance(model_root, nn.Module):
        logger.warning(
            "Skipping FLUX.2 text encoder FSDP because no shardable model root was found."
        )
        return text_encoder

    names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
    )
    fsdp_config: dict[str, Any] = {
        "mesh": parallel_dims.get_mesh(names),
        "mp_policy": mp_policy,
    }
    if training.enable_cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    found_blocks = _find_transformer_blocks(model_root)
    if found_blocks is not None:
        block_path, blocks = found_blocks
        for block in blocks:
            fully_shard(block, **fsdp_config)
        logger.info(f"Applied block-wise FSDP to FLUX.2 text encoder at {block_path}")
    else:
        logger.warning(
            "Falling back to root-only FSDP for the FLUX.2 text encoder; no block list was found."
        )

    fully_shard(model_root, **fsdp_config)
    disable_fsdp_gradient_division(model_root)
    return text_encoder
