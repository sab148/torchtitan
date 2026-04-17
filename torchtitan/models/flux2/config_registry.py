# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from pathlib import Path

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, TrainingConfig

from . import model_registry
from .configs import Flux2EncoderConfig
from .flux_datasets import Flux2DataLoader
from .tokenizer import Flux2TokenizerContainer
from .trainer import Flux2Trainer


_DATA_CONSOLIDATION_CONFIG = (
    Path(__file__).resolve().parent / "DataConsolidation" / "config" / "joint_train_dataset_prefetch.yaml"
)

_KLEIN_4B_TRANSFORMER_PATH = (
    Path(__file__).resolve().parents[3] / "flux2_klein" / "flux-2-klein-4b.safetensors"
)


def _base_flux2_config(flavor: str) -> Flux2Trainer.Config:
    return Flux2Trainer.Config(
        tokenizer=Flux2TokenizerContainer.Config(max_text_encoding_len=512),
        encoder=Flux2EncoderConfig(
            autoencoder_path="assets/hf/FLUX.2-dev/ae.safetensors",
        ),
        metrics=MetricsProcessor.Config(
            log_freq=100,
            enable_tensorboard=True,
            save_tb_folder="tb",
        ),
        model_spec=model_registry(flavor),
        optimizer=OptimizersContainer.Config(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=3000,
            decay_ratio=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=16,
            max_norm=1.0,
            steps=30000,
        ),
        dataloader=Flux2DataLoader.Config(
            dataset="cc12m-wds",
            prompt_dropout_prob=0.0,
            img_size=256,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        checkpoint=CheckpointManager.Config(interval=1000),
    )


def flux2_debugmodel() -> Flux2Trainer.Config:
    return Flux2Trainer.Config(
        tokenizer=Flux2TokenizerContainer.Config(max_text_encoding_len=512),
        encoder=Flux2EncoderConfig(
            autoencoder_path="assets/hf/FLUX.2-dev/ae.safetensors",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("flux.2-debug"),
        optimizer=OptimizersContainer.Config(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=5,
            decay_ratio=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            max_norm=1.0,
            steps=10,
        ),
        dataloader=Flux2DataLoader.Config(
            dataset="cc12m-test",
            prompt_dropout_prob=0.0,
            img_size=256,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        checkpoint=CheckpointManager.Config(enable=False),
    )


def flux2_dev() -> Flux2Trainer.Config:
    return _base_flux2_config("flux.2-dev")


def flux2_klein_4b() -> Flux2Trainer.Config:
    return _base_flux2_config("flux.2-klein-4b")


def flux2_klein_9b() -> Flux2Trainer.Config:
    return _base_flux2_config("flux.2-klein-9b")


def flux2_klein_9b_kv() -> Flux2Trainer.Config:
    return _base_flux2_config("flux.2-klein-9b-kv")


def flux2_klein_base_4b() -> Flux2Trainer.Config:
    return _base_flux2_config("flux.2-klein-base-4b")


def flux2_klein_base_9b() -> Flux2Trainer.Config:
    return _base_flux2_config("flux.2-klein-base-9b")


def flux2_klein_4b_dataconsolidation() -> Flux2Trainer.Config:
    config = _base_flux2_config("flux.2-klein-4b")
    config.encoder.transformer_path = str(_KLEIN_4B_TRANSFORMER_PATH)
    config.dataloader.dataset = "data-consolidation"
    config.dataloader.dataset_path = str(_DATA_CONSOLIDATION_CONFIG)
    config.dataloader.num_workers = 0
    config.dataloader.persistent_workers = False
    config.dataloader.prefetch_factor = None
    return config
