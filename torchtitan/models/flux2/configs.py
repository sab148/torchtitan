# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class Flux2EncoderConfig:
    """Configuration for FLUX.2 encoder components."""

    autoencoder_path: str = ""
    """Optional local path to ``ae.safetensors``. Empty falls back to ``AE_MODEL_PATH`` or HF."""

    text_encoder_model: str = ""
    """Optional local or Hugging Face override for the FLUX.2 text encoder weights."""

    text_encoder_processor_model: str = ""
    """Optional processor override for Mistral-based FLUX.2 text encoders."""

    random_init: bool = False
    """If True, initialize the autoencoder randomly instead of loading pretrained weights."""
