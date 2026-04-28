# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torch import nn
from torchtitan.protocols import BaseModel
from torchtitan.tools.logging import logger

from ..src.flux2.autoencoder import AutoEncoderParams
from ..src.flux2.model import (
    DoubleStreamBlock,
    Flux2 as Flux2Base,
    Flux2Params,
    LastLayer,
    MLPEmbedder,
    Modulation,
    RMSNorm,
    SelfAttention,
    SingleStreamBlock,
)


def _init_mlp_embedder(module: MLPEmbedder, init_std: float = 0.02) -> None:
    nn.init.normal_(module.in_layer.weight, std=init_std)
    if module.in_layer.bias is not None:
        nn.init.zeros_(module.in_layer.bias)
    nn.init.normal_(module.out_layer.weight, std=init_std)
    if module.out_layer.bias is not None:
        nn.init.zeros_(module.out_layer.bias)


def _init_rms_norm(module: RMSNorm) -> None:
    nn.init.ones_(module.scale)


def _init_self_attention(module: SelfAttention) -> None:
    nn.init.xavier_uniform_(module.qkv.weight)
    if module.qkv.bias is not None:
        nn.init.zeros_(module.qkv.bias)
    nn.init.xavier_uniform_(module.proj.weight)
    if module.proj.bias is not None:
        nn.init.zeros_(module.proj.bias)
    _init_rms_norm(module.norm.query_norm)
    _init_rms_norm(module.norm.key_norm)


def _zero_modulation(module: Modulation) -> None:
    nn.init.zeros_(module.lin.weight)
    if module.lin.bias is not None:
        nn.init.zeros_(module.lin.bias)


def _init_double_stream_block(module: DoubleStreamBlock) -> None:
    _init_self_attention(module.img_attn)
    _init_self_attention(module.txt_attn)
    nn.init.xavier_uniform_(module.img_mlp[0].weight)
    nn.init.xavier_uniform_(module.img_mlp[2].weight)
    nn.init.xavier_uniform_(module.txt_mlp[0].weight)
    nn.init.xavier_uniform_(module.txt_mlp[2].weight)
    module.img_norm1.reset_parameters()
    module.img_norm2.reset_parameters()
    module.txt_norm1.reset_parameters()
    module.txt_norm2.reset_parameters()


def _init_single_stream_block(module: SingleStreamBlock) -> None:
    nn.init.xavier_uniform_(module.linear1.weight)
    if module.linear1.bias is not None:
        nn.init.zeros_(module.linear1.bias)
    nn.init.xavier_uniform_(module.linear2.weight)
    if module.linear2.bias is not None:
        nn.init.zeros_(module.linear2.bias)
    _init_rms_norm(module.norm.query_norm)
    _init_rms_norm(module.norm.key_norm)
    module.pre_norm.reset_parameters()


def _init_last_layer(module: LastLayer) -> None:
    module.norm_final.reset_parameters()
    nn.init.zeros_(module.adaLN_modulation[-1].weight)
    if module.adaLN_modulation[-1].bias is not None:
        nn.init.zeros_(module.adaLN_modulation[-1].bias)
    nn.init.zeros_(module.linear.weight)
    if module.linear.bias is not None:
        nn.init.zeros_(module.linear.bias)


class Flux2Model(Flux2Base, BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        in_channels: int = 128
        context_in_dim: int = 15360
        hidden_size: int = 6144
        num_heads: int = 48
        depth: int = 8
        depth_single_blocks: int = 48
        axes_dim: tuple[int, ...] = (32, 32, 32, 32)
        theta: int = 2000
        mlp_ratio: float = 3.0
        use_guidance_embed: bool = True
        autoencoder_params: AutoEncoderParams = field(default_factory=AutoEncoderParams)
        text_encoder_kind: str = "mistral"
        guidance: float = 4.0

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            del kwargs

            img_size = trainer_config.dataloader.img_size
            ae_downscale = 2 ** (len(self.autoencoder_params.ch_mult) - 1)
            if img_size % ae_downscale != 0:
                raise ValueError(
                    f"FLUX.2 expects image sizes divisible by {ae_downscale}, got {img_size}"
                )

            if trainer_config.parallelism.context_parallel_degree > 1:
                raise NotImplementedError(
                    "FLUX.2 context parallel is not implemented in this TorchTitan integration."
                )

            self.autoencoder_params.resolution = img_size

            if trainer_config.guidance >= 0:
                self.guidance = trainer_config.guidance

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            del seq_len
            nparams = sum(p.numel() for p in model.parameters())
            logger.warning("FLUX.2 get_nparams_and_flops() is not implemented yet")
            return nparams, 1

    def __init__(self, config: Config):
        self.config = config
        super().__init__(
            Flux2Params(
                in_channels=config.in_channels,
                context_in_dim=config.context_in_dim,
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                depth=config.depth,
                depth_single_blocks=config.depth_single_blocks,
                axes_dim=list(config.axes_dim),
                theta=config.theta,
                mlp_ratio=config.mlp_ratio,
                use_guidance_embed=config.use_guidance_embed,
            )
        )

    def verify_module_protocol(self) -> None:
        # FLUX.2 wraps vendored nn.Module internals that do not implement
        # TorchTitan's Module protocol.
        pass

    def init_weights(self, *, buffer_device=None, **kwargs) -> None:
        del buffer_device
        del kwargs

        nn.init.xavier_uniform_(self.img_in.weight)
        if self.img_in.bias is not None:
            nn.init.zeros_(self.img_in.bias)
        nn.init.xavier_uniform_(self.txt_in.weight)
        if self.txt_in.bias is not None:
            nn.init.zeros_(self.txt_in.bias)

        _init_mlp_embedder(self.time_in)
        if self.use_guidance_embed:
            _init_mlp_embedder(self.guidance_in)

        for block in self.double_blocks:
            _init_double_stream_block(block)
        for block in self.single_blocks:
            _init_single_stream_block(block)

        _zero_modulation(self.double_stream_modulation_img)
        _zero_modulation(self.double_stream_modulation_txt)
        _zero_modulation(self.single_stream_modulation)
        _init_last_layer(self.final_layer)
