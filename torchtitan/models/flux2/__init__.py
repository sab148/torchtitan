# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_mse_loss
from torchtitan.protocols.model_spec import ModelSpec

from .flux_datasets import Flux2DataLoader
from .model.model import Flux2Model
from .parallelize import parallelize_flux2
from .src.flux2.autoencoder import AutoEncoderParams
from .src.flux2.model import Flux2Params, Klein4BParams, Klein9BParams

__all__ = ["Flux2Model", "Flux2DataLoader", "flux2_configs", "parallelize_flux2"]


def _build_model_config(
    params: Flux2Params | Klein4BParams | Klein9BParams,
    *,
    text_encoder_kind: str,
    guidance: float,
) -> Flux2Model.Config:
    return Flux2Model.Config(
        in_channels=params.in_channels,
        context_in_dim=params.context_in_dim,
        hidden_size=params.hidden_size,
        num_heads=params.num_heads,
        depth=params.depth,
        depth_single_blocks=params.depth_single_blocks,
        axes_dim=tuple(params.axes_dim),
        theta=params.theta,
        mlp_ratio=params.mlp_ratio,
        use_guidance_embed=params.use_guidance_embed,
        autoencoder_params=AutoEncoderParams(),
        text_encoder_kind=text_encoder_kind,
        guidance=guidance,
    )


flux2_configs = {
    "flux.2-dev": _build_model_config(
        Flux2Params(),
        text_encoder_kind="mistral",
        guidance=4.0,
    ),
    "flux.2-klein-4b": _build_model_config(
        Klein4BParams(),
        text_encoder_kind="qwen3-4b",
        guidance=1.0,
    ),
    "flux.2-klein-9b": _build_model_config(
        Klein9BParams(),
        text_encoder_kind="qwen3-8b",
        guidance=1.0,
    ),
    "flux.2-klein-9b-kv": _build_model_config(
        Klein9BParams(),
        text_encoder_kind="qwen3-8b",
        guidance=1.0,
    ),
    "flux.2-klein-base-4b": _build_model_config(
        Klein4BParams(),
        text_encoder_kind="qwen3-4b",
        guidance=4.0,
    ),
    "flux.2-klein-base-9b": _build_model_config(
        Klein9BParams(),
        text_encoder_kind="qwen3-8b",
        guidance=4.0,
    ),
    "flux.2-debug": Flux2Model.Config(
        in_channels=128,
        context_in_dim=15360,
        hidden_size=1024,
        num_heads=8,
        depth=2,
        depth_single_blocks=4,
        axes_dim=(32, 32, 32, 32),
        theta=2000,
        mlp_ratio=3.0,
        use_guidance_embed=True,
        autoencoder_params=AutoEncoderParams(),
        text_encoder_kind="mistral",
        guidance=4.0,
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="flux2",
        flavor=flavor,
        model=flux2_configs[flavor],
        parallelize_fn=parallelize_flux2,
        pipelining_fn=None,
        build_loss_fn=build_mse_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
