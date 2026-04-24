# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.components.tokenizer import BaseTokenizer


class Flux2TokenizerContainer(BaseTokenizer):
    """Lightweight tokenizer placeholder for FLUX.2.

    FLUX.2 tokenization is handled inside the real text encoder, but TorchTitan's
    Trainer still expects a tokenizer component to exist. This container only
    carries the requested max text length through the config system.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        max_text_encoding_len: int = 512

    def __init__(self, config: Config, **kwargs):
        super().__init__()
        del kwargs
        self.max_text_encoding_len = config.max_text_encoding_len

    def encode(self, *args, **kwargs) -> list[int]:
        del args, kwargs
        return []

    def decode(self, *args, **kwargs) -> str:
        del args, kwargs
        return ""

    def get_vocab_size(self) -> int:
        return 0
