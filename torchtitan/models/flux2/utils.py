# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file as load_sft
from torch import Tensor, nn

from .configs import Flux2EncoderConfig
from .src.flux2.autoencoder import AutoEncoder, AutoEncoderParams

if TYPE_CHECKING:
    from .model.model import Flux2Model

AE_FILENAME = "ae.safetensors"
AE_REPO_ID = "black-forest-labs/FLUX.2-dev"
DEFAULT_MISTRAL_MODEL = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
DEFAULT_MISTRAL_PROCESSOR = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
QWEN_MODEL_SPECS = {
    "qwen3-4b": "Qwen/Qwen3-4B-FP8",
    "qwen3-8b": "Qwen/Qwen3-8B-FP8",
}


def _candidate_hf_hub_caches() -> list[str]:
    candidates: list[str] = []
    for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(value)

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(os.path.join(hf_home, "hub"))

    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if user:
        candidates.append(f"/p/scratch/atmlaml/{user}/.cache/hub")
        candidates.append(f"/p/home/jusers/{user}/juwels/.cache/huggingface/hub")

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    candidates.append(os.path.join(repo_root, "assets", "hf", "hub"))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        norm = os.path.abspath(candidate)
        if norm not in seen:
            seen.add(norm)
            deduped.append(norm)
    return deduped


def _resolve_local_hf_asset_dir(repo_id: str) -> str | None:
    repo_name = repo_id.split('/')[-1]
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    candidate_dirs = [
        os.path.join(repo_root, "assets", "hf", repo_name),
        os.path.join(os.path.dirname(repo_root), "assets", "hf", repo_name),
    ]

    required_weight_files = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )

    for candidate in candidate_dirs:
        if not os.path.isfile(os.path.join(candidate, "config.json")):
            continue
        if any(os.path.isfile(os.path.join(candidate, filename)) for filename in required_weight_files):
            return candidate

    return None


def _resolve_local_hf_snapshot(repo_id: str, *, repo_type: str = "model") -> str | None:
    repo_prefix = "models" if repo_type == "model" else "datasets"
    repo_cache_name = f"{repo_prefix}--{repo_id.replace('/', '--')}"

    for hub_cache in _candidate_hf_hub_caches():
        repo_cache = os.path.join(hub_cache, repo_cache_name)
        snapshots_dir = os.path.join(repo_cache, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue

        revisions: list[str] = []
        ref_path = os.path.join(repo_cache, "refs", "main")
        if os.path.isfile(ref_path):
            with open(ref_path) as ref_file:
                revision = ref_file.read().strip()
            if revision:
                revisions.append(revision)

        for revision in sorted(os.listdir(snapshots_dir), reverse=True):
            if revision not in revisions:
                revisions.append(revision)

        for revision in revisions:
            snapshot_path = os.path.join(snapshots_dir, revision)
            if os.path.isfile(os.path.join(snapshot_path, "config.json")):
                return snapshot_path

    return None


def _resolve_text_encoder_model_spec(model_spec: str) -> str:
    if not model_spec:
        return model_spec
    if os.path.exists(model_spec):
        return model_spec

    local_asset_dir = _resolve_local_hf_asset_dir(model_spec)
    if local_asset_dir is not None:
        return local_asset_dir

    local_snapshot = _resolve_local_hf_snapshot(model_spec, repo_type="model")
    if local_snapshot is not None:
        return local_snapshot
    return model_spec


def _resolve_flux2_autoencoder_path(ckpt_path: str) -> str:
    if ckpt_path:
        if not os.path.exists(ckpt_path):
            raise ValueError(
                f"Autoencoder path {ckpt_path} does not exist. Please update encoder.autoencoder_path."
            )
        return ckpt_path

    env_path = os.environ.get("AE_MODEL_PATH")
    if env_path:
        if not os.path.exists(env_path):
            raise ValueError(
                f"AE_MODEL_PATH={env_path} does not exist. Please update the environment variable."
            )
        return env_path

    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError(
            "FLUX.2 autoencoder weights were not provided locally and huggingface_hub is not available."
        ) from exc

    return huggingface_hub.hf_hub_download(
        repo_id=AE_REPO_ID,
        filename=AE_FILENAME,
        repo_type="model",
    )


def load_flux2_transformer_state_dict(ckpt_path: str, *, device: str | torch.device = "cpu") -> dict[str, Tensor]:
    if not ckpt_path:
        raise ValueError("Transformer checkpoint path is empty")
    if not os.path.exists(ckpt_path):
        raise ValueError(
            f"Transformer checkpoint path {ckpt_path} does not exist. Please update encoder.transformer_path."
        )
    return load_sft(ckpt_path, device=str(device))


def load_flux2_autoencoder(
    ckpt_path: str,
    autoencoder_params: AutoEncoderParams,
    device: str | torch.device,
    dtype: torch.dtype,
    random_init: bool = False,
) -> AutoEncoder:
    with torch.device(device):
        ae = AutoEncoder(autoencoder_params)

    if random_init:
        return ae.to(dtype=dtype)

    weight_path = _resolve_flux2_autoencoder_path(ckpt_path)
    sd = load_sft(weight_path, device=str(device))
    ae.load_state_dict(sd, strict=True, assign=True)
    return ae.to(dtype=dtype)


def load_flux2_text_encoder(
    *,
    model_config: "Flux2Model.Config",
    encoder_config: Flux2EncoderConfig,
    max_length: int,
    device: torch.device,
) -> nn.Module:
    try:
        from .src.flux2.text_encoder import Mistral3SmallEmbedder, Qwen3Embedder
    except ImportError as exc:
        raise ImportError(
            "FLUX.2 text encoder support could not be imported. Install the vendored flux2 dependencies first."
        ) from exc

    override_model = encoder_config.text_encoder_model
    override_processor = encoder_config.text_encoder_processor_model

    if model_config.text_encoder_kind == "mistral":
        encoder = Mistral3SmallEmbedder(
            model_spec=_resolve_text_encoder_model_spec(
                override_model or DEFAULT_MISTRAL_MODEL
            ),
            model_spec_processor=(
                _resolve_text_encoder_model_spec(
                    override_processor or override_model or DEFAULT_MISTRAL_PROCESSOR
                )
            ),
        ).to(device)
    elif model_config.text_encoder_kind in QWEN_MODEL_SPECS:
        encoder = Qwen3Embedder(
            model_spec=_resolve_text_encoder_model_spec(
                override_model or QWEN_MODEL_SPECS[model_config.text_encoder_kind]
            ),
            device=device,
        )
    else:
        raise ValueError(
            f"Unsupported FLUX.2 text encoder kind: {model_config.text_encoder_kind}"
        )

    if hasattr(encoder, "max_length"):
        encoder.max_length = max_length

    return encoder.eval().requires_grad_(False)


def normalize_prompts(prompts: Any) -> list[str]:
    if isinstance(prompts, str):
        return [prompts]
    if isinstance(prompts, tuple):
        return [str(prompt) for prompt in prompts]
    if isinstance(prompts, list):
        return [str(prompt) for prompt in prompts]
    raise TypeError(f"Unsupported prompt batch type: {type(prompts)!r}")


def preprocess_flux2_batch(
    device: torch.device,
    dtype: torch.dtype,
    *,
    autoencoder: AutoEncoder,
    text_encoder: nn.Module,
    batch: dict[str, Any],
) -> dict[str, Any]:
    prompts = normalize_prompts(batch["prompt"])
    images = batch["image"].to(device=device, dtype=dtype)

    with torch.no_grad():
        img_encodings = autoencoder.encode(images).to(device=device, dtype=dtype)
        text_encodings = text_encoder(prompts)
        if not isinstance(text_encodings, torch.Tensor):
            raise TypeError(
                "FLUX.2 text encoder must return a Tensor of hidden states."
            )
        text_encodings = text_encodings.to(device=device, dtype=dtype)

    batch["prompt"] = prompts
    batch["img_encodings"] = img_encodings
    batch["text_encodings"] = text_encodings
    return batch


def flatten_image_tokens(latents: Tensor) -> Tensor:
    return latents.permute(0, 2, 3, 1).reshape(
        latents.shape[0], latents.shape[2] * latents.shape[3], latents.shape[1]
    )


def create_image_position_ids(
    batch_size: int,
    latent_height: int,
    latent_width: int,
    device: torch.device,
) -> Tensor:
    ids = torch.cartesian_prod(
        torch.arange(1, device=device, dtype=torch.long),
        torch.arange(latent_height, device=device, dtype=torch.long),
        torch.arange(latent_width, device=device, dtype=torch.long),
        torch.arange(1, device=device, dtype=torch.long),
    )
    return ids.unsqueeze(0).expand(batch_size, -1, -1)


def create_text_position_ids(
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Tensor:
    positions = torch.arange(seq_len, device=device, dtype=torch.long)
    ids = torch.stack(
        (
            torch.zeros_like(positions),
            torch.zeros_like(positions),
            torch.zeros_like(positions),
            positions,
        ),
        dim=-1,
    )
    return ids.unsqueeze(0).expand(batch_size, -1, -1)
