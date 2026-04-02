# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.trainer import Trainer

from .configs import Flux2EncoderConfig
from .flux_datasets import Flux2DataLoader
from .parallelize import parallelize_text_encoder
from .tokenizer import Flux2TokenizerContainer
from .utils import (
    create_image_position_ids,
    create_text_position_ids,
    flatten_image_tokens,
    load_flux2_autoencoder,
    load_flux2_text_encoder,
    preprocess_flux2_batch,
)


class Flux2Trainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        tokenizer: Flux2TokenizerContainer.Config = (  # pyrefly: ignore [bad-override]
            field(default_factory=Flux2TokenizerContainer.Config)
        )
        dataloader: Flux2DataLoader.Config = field(default_factory=Flux2DataLoader.Config)
        encoder: Flux2EncoderConfig = field(default_factory=Flux2EncoderConfig)
        guidance: float = -1.0
        """Override the FLUX.2 flavor guidance value. Negative keeps the flavor default."""

    def __init__(self, config: Config):
        assert config.model_spec is not None
        model_config = config.model_spec.model

        img_size = config.dataloader.img_size
        ae_downscale = 2 ** (len(model_config.autoencoder_params.ch_mult) - 1)
        if img_size % ae_downscale != 0:
            raise ValueError(
                f"FLUX.2 expects image sizes divisible by {ae_downscale}, got {img_size}"
            )

        latent_side = img_size // ae_downscale
        seq_len_img = latent_side * latent_side
        seq_len_txt = config.tokenizer.max_text_encoding_len
        config.training.seq_len = seq_len_img + seq_len_txt

        super().__init__(config)

        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["fsdp", "dp_replicate"],
        )

        self._dtype = (
            TORCH_DTYPE_MAP[config.training.mixed_precision_param]
            if self.parallel_dims.dp_shard_enabled
            else torch.float32
        )

        self.autoencoder = load_flux2_autoencoder(
            ckpt_path=config.encoder.autoencoder_path,
            autoencoder_params=model_config.autoencoder_params,
            device=self.device,
            dtype=self._dtype,
            random_init=config.encoder.random_init,
        )

        self.text_encoder = load_flux2_text_encoder(
            model_config=model_config,
            encoder_config=config.encoder,
            max_length=config.tokenizer.max_text_encoding_len,
            device=self.device,
        )
        self.text_encoder = parallelize_text_encoder(
            self.text_encoder,
            parallel_dims=self.parallel_dims,
            training=config.training,
        )

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, Any], torch.Tensor]]
    ) -> Iterator[tuple[dict[str, Any], torch.Tensor]]:
        data_iterator = iter(data_iterable)
        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            input_dict, labels = batch
            bsz = labels.shape[0]
            ntokens_batch = bsz * self.config.training.seq_len
            self.ntokens_seen += ntokens_batch
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )
            yield input_dict, labels

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, Any],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert (
            global_valid_tokens is None
        ), "FLUX.2 does not rescale loss by global valid token count from the caller."

        input_dict["image"] = labels
        input_dict = preprocess_flux2_batch(
            device=self.device,
            dtype=self._dtype,
            autoencoder=self.autoencoder,
            text_encoder=self.text_encoder,
            batch=input_dict,
        )
        labels = input_dict["img_encodings"]

        local_valid_tokens = torch.tensor(
            labels.numel(), dtype=torch.float32, device=self.device
        )
        if self.parallel_dims.dp_enabled:
            batch_mesh = self.parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum(local_valid_tokens, batch_mesh)
        else:
            global_valid_tokens = local_valid_tokens

        model = self.model_parts[0]
        text_encodings = input_dict["text_encodings"]
        batch_size = labels.shape[0]

        with torch.no_grad():
            noise = torch.randn_like(labels)
            timesteps = torch.rand(
                (batch_size,),
                device=self.device,
                dtype=labels.dtype,
            )
            sigmas = timesteps.view(-1, 1, 1, 1)
            noisy_latents = (1 - sigmas) * labels + sigmas * noise

            _, _, latent_height, latent_width = noisy_latents.shape
            latent_pos_ids = create_image_position_ids(
                batch_size,
                latent_height,
                latent_width,
                self.device,
            )
            text_pos_ids = create_text_position_ids(
                batch_size,
                text_encodings.shape[1],
                self.device,
            )
            noisy_latents = flatten_image_tokens(noisy_latents)
            target = flatten_image_tokens(noise - labels)

        if self.parallel_dims.cp_enabled:
            raise NotImplementedError(
                "FLUX.2 context parallel support is not implemented in this TorchTitan integration."
            )

        guidance = None
        if model.use_guidance_embed:
            guidance = torch.full(
                (batch_size,),
                model.config.guidance,
                device=self.device,
                dtype=labels.dtype,
            )

        with self.train_context():
            with self.maybe_enable_amp:
                latent_noise_pred = model(
                    x=noisy_latents,
                    x_ids=latent_pos_ids,
                    timesteps=timesteps,
                    ctx=text_encodings,
                    ctx_ids=text_pos_ids,
                    guidance=guidance,
                )
                loss = self.loss_fn(latent_noise_pred, target) / global_valid_tokens

            del latent_noise_pred, noise, target
            loss.backward()

        return loss

    def train_step(
        self, data_iterator: Iterable[tuple[dict[str, Any], torch.Tensor]]
    ) -> None:
        self.optimizers.zero_grad()
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
        parallel_dims = self.parallel_dims

        if self.gradient_accumulation_steps > 1:
            raise ValueError("FLUX.2 does not support gradient accumulation for now.")

        input_dict, labels = next(data_iterator)
        loss = self.forward_backward_step(input_dict=input_dict, labels=labels)

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_sum(loss, loss_mesh),
                dist_utils.dist_max(loss, loss_mesh),
                dist_utils.dist_sum(
                    torch.tensor(self.ntokens_seen, dtype=torch.int64, device=self.device),
                    loss_mesh,
                ),
            )
        else:
            global_avg_loss = global_max_loss = float(loss.detach().item())
            global_ntokens_seen = self.ntokens_seen

        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            extra_metrics={
                "n_tokens_seen": global_ntokens_seen,
                "lr": lr,
            },
        )
