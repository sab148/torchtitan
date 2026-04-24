# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import sys
from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import PIL.Image
import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger


DATA_CONSOLIDATION_DATASET = "data-consolidation"


def _load_hf_dataset(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _split_hf_dataset_by_node(dataset, dp_rank: int, dp_world_size: int):
    from datasets.distributed import split_dataset_by_node

    return split_dataset_by_node(dataset, dp_rank, dp_world_size)


def _is_hf_map_dataset(dataset: Any) -> bool:
    try:
        from datasets import Dataset as HFDataset
    except Exception:
        return False

    return isinstance(dataset, HFDataset)


def _is_map_style_dataset(dataset: Any) -> bool:
    return hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")


def _process_tensor_image(
    img: torch.Tensor | np.ndarray,
    output_size: int = 256,
) -> torch.Tensor | None:
    if not torch.is_tensor(img):
        img = torch.as_tensor(img)

    if img.ndim != 3:
        raise ValueError(f"Expected a 3D image tensor, got shape {tuple(img.shape)}")

    if img.shape[0] not in (1, 3, 4) and img.shape[-1] in (1, 3, 4):
        img = img.permute(2, 0, 1)

    channels = img.shape[0]
    if channels == 1:
        img = img.expand(3, -1, -1)
    elif channels == 4:
        img = img[:3]
    elif channels != 3:
        raise ValueError(f"Expected 1, 3, or 4 channels, got {channels}")

    img = img.to(dtype=torch.float32)
    height, width = img.shape[-2:]
    if width < output_size or height < output_size:
        return None

    img_min = float(img.min())
    img_max = float(img.max())
    if img_min >= 0.0 and img_max <= 1.01:
        normalized = img * 2.0 - 1.0
    elif img_min >= -1.01 and img_max <= 1.01:
        normalized = img
    else:
        normalized = img / 127.5 - 1.0

    if width >= height:
        new_width, new_height = math.ceil(output_size / height * width), output_size
        left = torch.randint(0, new_width - output_size + 1, (1,)).item()
        top = 0
    else:
        new_width, new_height = output_size, math.ceil(output_size / width * height)
        left = 0
        top = torch.randint(0, new_height - output_size + 1, (1,)).item()

    resized = torch.nn.functional.interpolate(
        normalized.unsqueeze(0),
        size=(new_height, new_width),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)

    return resized[:, top : top + output_size, left : left + output_size].contiguous()


def _data_consolidation_data_processor(
    sample: dict[str, Any],
    output_size: int = 256,
) -> dict[str, Any]:
    if 'jpg' not in sample:
        raise KeyError("DataConsolidation samples must expose a 'jpg' field")
    if 'txt' not in sample:
        raise KeyError("DataConsolidation samples must expose a 'txt' field")

    prompt = sample['txt']
    if isinstance(prompt, bytes):
        prompt = prompt.decode('utf-8')
    elif isinstance(prompt, (list, tuple)):
        prompt = prompt[0] if prompt else ''
    elif prompt is None:
        prompt = ''

    return {
        'image': _process_tensor_image(sample['jpg'], output_size=output_size),
        'prompt': str(prompt),
    }


def _load_data_consolidation_dataset(config_path: str, split_name: str = 'train_dataloader'):
    package_root = Path(__file__).resolve().parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    from omegaconf import OmegaConf
    from DataConsolidation.src.utils import instantiate_from_config

    cfg = OmegaConf.load(config_path)
    split_cfg = cfg.get(split_name)
    if split_cfg is None:
        raise KeyError(f"DataConsolidation config {config_path} has no {split_name!r} section")

    params = split_cfg.get('params')
    if params is None or 'dataset' not in params:
        raise KeyError(f"DataConsolidation config {config_path} is missing {split_name}.params.dataset")

    dataset = instantiate_from_config(params['dataset'], recursive=True, debug=False)
    if not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'):
        raise TypeError('DataConsolidation config must instantiate a map-style dataset')
    return dataset


def _has_local_tar_shards(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    with os.scandir(path) as entries:
        return any(entry.name.endswith(".tar") for entry in entries)


def _resolve_cached_hf_dataset_snapshot(repo_id: str) -> str | None:
    hub_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not hub_cache:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            hub_cache = os.path.join(hf_home, "hub")
    if not hub_cache:
        return None

    repo_cache = os.path.join(hub_cache, f"datasets--{repo_id.replace('/', '--')}")
    snapshots_dir = os.path.join(repo_cache, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

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
        if _has_local_tar_shards(snapshot_path):
            return snapshot_path

    return None


def _load_cc12m_wds_dataset(path: str):
    if _has_local_tar_shards(path):
        logger.info(f"Loading cc12m-wds from local tar shard directory {path}")
        return _load_hf_dataset(
            path,
            split="train",
            data_files={"train": "*.tar"},
            streaming=True,
        )

    cached_snapshot_path = _resolve_cached_hf_dataset_snapshot(path)
    if cached_snapshot_path is not None:
        logger.info(
            f"Loading cc12m-wds from cached HuggingFace snapshot {cached_snapshot_path}"
        )
        return _load_hf_dataset(
            cached_snapshot_path,
            split="train",
            data_files={"train": "*.tar"},
            streaming=True,
        )

    return _load_hf_dataset(path, split="train", streaming=True)


def _process_image(
    img: PIL.Image.Image,
    output_size: int = 256,
) -> torch.Tensor | None:
    width, height = img.size
    if width < output_size or height < output_size:
        return None

    if width >= height:
        new_width, new_height = math.ceil(output_size / height * width), output_size
        img = img.resize((new_width, new_height))
        left = torch.randint(0, new_width - output_size + 1, (1,)).item()
        resized_img = img.crop((left, 0, left + output_size, output_size))
    else:
        new_width, new_height = output_size, math.ceil(output_size / width * height)
        img = img.resize((new_width, new_height))
        lower = torch.randint(0, new_height - output_size + 1, (1,)).item()
        resized_img = img.crop((0, lower, output_size, lower + output_size))

    if resized_img.mode != "RGB":
        resized_img = resized_img.convert("RGB")

    np_img = np.array(resized_img).transpose((2, 0, 1))
    return torch.tensor(np_img).float() / 255.0 * 2.0 - 1.0


def _cc12m_wds_data_processor(
    sample: dict[str, Any],
    output_size: int = 256,
) -> dict[str, Any]:
    return {
        "image": _process_image(sample["jpg"], output_size=output_size),
        "prompt": sample["txt"],
    }


def _coco_data_processor(
    sample: dict[str, Any],
    output_size: int = 256,
) -> dict[str, Any]:
    prompt = sample["caption"]
    if isinstance(prompt, list):
        prompt = prompt[0]

    return {
        "image": _process_image(sample["image"], output_size=output_size),
        "prompt": prompt,
    }


DATASETS = {
    "cc12m-wds": DatasetConfig(
        path="pixparse/cc12m-wds",
        loader=_load_cc12m_wds_dataset,
        sample_processor=_cc12m_wds_data_processor,
    ),
    "cc12m-test": DatasetConfig(
        path="tests/assets/cc12m_test",
        loader=lambda path: _load_hf_dataset(
            path, split="train", data_files={"train": "*.tar"}, streaming=True
        ),
        sample_processor=_cc12m_wds_data_processor,
    ),
    "coco-validation": DatasetConfig(
        path="howard-hou/COCO-Text",
        loader=lambda path: _load_hf_dataset(path, split="validation", streaming=True),
        sample_processor=_coco_data_processor,
    ),
    DATA_CONSOLIDATION_DATASET: DatasetConfig(
        path=str(Path(__file__).resolve().parent / "DataConsolidation" / "config" / "joint_train_dataset.yaml"),
        loader=lambda path: _load_data_consolidation_dataset(path, split_name='train_dataloader'),
        sample_processor=_data_consolidation_data_processor,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    if dataset_name == DATA_CONSOLIDATION_DATASET and not path:
        raise ValueError("Dataset 'data-consolidation' requires dataset_path to point to a DataConsolidation YAML config")
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class Flux2Dataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        *,
        prompt_dropout_prob: float,
        img_size: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        dataset_name = dataset_name.lower()

        path, dataset_loader, data_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size
        if dataset_name == DATA_CONSOLIDATION_DATASET:
            self._data = ds
        else:
            self._data = _split_hf_dataset_by_node(ds, dp_rank, dp_world_size)
        self._is_hf_map = _is_hf_map_dataset(self._data)
        self._is_map_style = _is_map_style_dataset(self._data)
        self._data_processor = data_processor
        self.prompt_dropout_prob = prompt_dropout_prob
        self.img_size = img_size
        self.infinite = infinite
        self._sample_idx = 0

    def _get_data_iter(self):
        if self._is_hf_map:
            if self._sample_idx == len(self._data):
                return iter([])
            return iter(self._data.skip(self._sample_idx))

        if self._is_map_style:
            start_idx = self._dp_rank + self._sample_idx * self._dp_world_size
            return (self._data[idx] for idx in range(start_idx, len(self._data), self._dp_world_size))

        return iter(self._data)

    def __iter__(self):
        dataset_iterator = self._get_data_iter()
        while True:
            try:
                sample = next(dataset_iterator)
            except StopIteration:
                if not self.infinite:
                    logger.warning(
                        f"Dataset {self.dataset_name} has run out of data. "
                        "This might cause NCCL timeout if data parallelism is enabled."
                    )
                    break

                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped.")
                dataset_iterator = self._get_data_iter()
                if not self._is_map_style:
                    if hasattr(self._data, "set_epoch") and hasattr(self._data, "epoch"):
                        self._data.set_epoch(self._data.epoch + 1)
                continue

            self._sample_idx += 1
            sample_dict = self._data_processor(
                sample,
                output_size=self.img_size,
            )

            if sample_dict["image"] is None:
                sample_id = sample.get("__key__", "unknown") if isinstance(sample, dict) else "unknown"
                logger.warning(
                    f"Low quality image {sample_id} is skipped in Flux2 Dataloader."
                )
                continue

            dropout_prob = self.prompt_dropout_prob
            if dropout_prob > 0.0 and torch.rand(1).item() < dropout_prob:
                sample_dict["prompt"] = ""

            labels = sample_dict.pop("image")
            yield sample_dict, labels

    def load_state_dict(self, state_dict):
        if self._is_map_style:
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        if self._is_map_style:
            return {"sample_idx": self._sample_idx}
        return {"data": self._data.state_dict()}


class Flux2DataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = "cc12m-test"
        infinite: bool = True
        prompt_dropout_prob: float = 0.0
        img_size: int = 256

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        local_batch_size: int,
        tokenizer=None,
        **kwargs,
    ):
        del tokenizer, kwargs

        dataloader_num_workers = config.num_workers
        dataloader_persistent_workers = config.persistent_workers
        dataloader_prefetch_factor = config.prefetch_factor
        if config.dataset.lower() == DATA_CONSOLIDATION_DATASET and config.num_workers != 0:
            logger.warning(
                "DataConsolidation integration uses a single iterable wrapper; overriding num_workers to 0 to avoid duplicate samples"
            )
            dataloader_num_workers = 0
            dataloader_persistent_workers = False
            dataloader_prefetch_factor = None

        ds = Flux2Dataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            prompt_dropout_prob=config.prompt_dropout_prob,
            img_size=config.img_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )

        dataloader_kwargs = {
            "num_workers": dataloader_num_workers,
            "persistent_workers": dataloader_persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": dataloader_prefetch_factor,
            "batch_size": local_batch_size,
        }

        super().__init__(
            ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
