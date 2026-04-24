# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


@dataclass
class Training:
    classifier_free_guidance_prob: float = 0.0
    """Probability of replacing a caption with an empty prompt during training."""

    img_size: int = 256
    """Square image size used by the dataloader."""

    guidance: float = -1.0
    """Guidance value fed into distilled FLUX.2 variants. Negative keeps the flavor default."""

    test_mode: bool = False
    """Whether to use randomly initialized encoder components for lightweight smoke tests."""


@dataclass
class Encoder:
    autoencoder_path: str = ""
    """Optional local path to `ae.safetensors`. If empty, the loader falls back to `AE_MODEL_PATH` or HF."""

    max_text_encoding_len: int = 512
    """Maximum text sequence length requested from the FLUX.2 text encoder."""

    text_encoder_model: str = ""
    """Optional local or Hugging Face model path overriding the default FLUX.2 text encoder."""

    text_encoder_processor_model: str = ""
    """Optional processor override for Mistral-based FLUX.2 text encoders."""


@dataclass
class JobConfig:
    """
    Extend the TorchTitan JobConfig with FLUX.2-specific training and encoder fields.
    """

    training: Training = field(default_factory=Training)
    encoder: Encoder = field(default_factory=Encoder)
