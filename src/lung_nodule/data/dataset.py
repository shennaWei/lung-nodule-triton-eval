"""Datasets for the two training stages.

Stage 1 uses MONAI's patch samplers on top of the cached volumes; stage 2 uses a
lightweight torch dataset that crops a fixed-size box around each candidate. Both
read from the same 1 mm ``npz`` cache produced by :mod:`lung_nodule.data.preprocess`.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from lung_nodule.data.candidates import CandidateSample
from lung_nodule.data.luna16 import VolumeGeometry, load_preprocessed
from lung_nodule.utils.arrays import crop_3d

_NPZ_CACHE_SIZE = 4  # a handful of ~50-100 MB volumes; raise only if RAM allows


@lru_cache(maxsize=_NPZ_CACHE_SIZE)
def _load_cached(npz_path: str) -> tuple[np.ndarray, np.ndarray, VolumeGeometry]:
    image, mask, _heatmap, geometry = load_preprocessed(npz_path)
    return image, mask, geometry


def clear_volume_cache() -> None:
    _load_cached.cache_clear()


def build_stage1_data_list(preprocessed_dir: str | Path, series_uids: Sequence[str]) -> list[dict[str, Any]]:
    """Dict list accepted by ``monai.data.Dataset`` / ``CacheDataset``."""
    root = Path(preprocessed_dir)
    data: list[dict[str, Any]] = []
    for uid in series_uids:
        path = root / f"{uid}.npz"
        if not path.exists():
            continue
        data.append({"npz_path": str(path), "series_uid": str(uid)})
    return data


def missing_scans(preprocessed_dir: str | Path, series_uids: Sequence[str]) -> list[str]:
    root = Path(preprocessed_dir)
    return [uid for uid in series_uids if not (root / f"{uid}.npz").exists()]


class Stage2CandidateDataset(Dataset):
    """Fixed-size candidate crops for the false-positive reducer.

    Each item exposes two single-channel arrays (``image`` and ``mask``) so the
    same augmentation can be applied to both, plus the binary ``label`` and the
    nodule ``diameter`` in millimetres.
    """

    def __init__(
        self,
        samples: Sequence[CandidateSample],
        preprocessed_dir: str | Path,
        crop_size: Sequence[int] = (48, 48, 48),
        use_mask_channel: bool = True,
        transform: Any | None = None,
    ) -> None:
        if len(crop_size) != 3:
            raise ValueError(f"crop_size must have 3 entries, got {tuple(crop_size)}")
        self.samples = list(samples)
        self.root = Path(preprocessed_dir)
        self.crop_size = tuple(int(s) for s in crop_size)
        self.use_mask_channel = bool(use_mask_channel)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        npz_path = self.root / f"{sample.series_uid}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"missing preprocessed volume for {sample.series_uid}: run `lndet-preprocess` first"
            )

        image, mask, geometry = _load_cached(str(npz_path))
        voxel_center = geometry.world_to_voxel(sample.coord)
        image_crop = crop_3d(image, voxel_center, self.crop_size)
        mask_crop = crop_3d(mask.astype(np.float32), voxel_center, self.crop_size)

        item: dict[str, Any] = {
            "image": image_crop[None, ...].astype(np.float32),
            "mask": mask_crop[None, ...].astype(np.float32),
            "label": float(sample.label),
            "diameter": float(max(sample.diameter_mm, 1.0)),
            "series_uid": sample.series_uid,
        }
        if self.transform is not None:
            item = self.transform(item)
        return item

    def labels(self) -> np.ndarray:
        return np.asarray([sample.label for sample in self.samples], dtype=np.float32)


class Stage2CropBatch:
    """Stack dataset items into the tensors expected by :class:`Stage2Classifier`."""

    def __init__(self, use_mask_channel: bool = True) -> None:
        self.use_mask_channel = use_mask_channel

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        images = torch.stack([item["image"] for item in batch], dim=0)
        masks = torch.stack([item["mask"] for item in batch], dim=0)
        inputs = torch.cat([images, masks], dim=1) if self.use_mask_channel else images
        return {
            "image": inputs,
            "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
            "diameter": torch.tensor([item["diameter"] for item in batch], dtype=torch.float32),
            "series_uid": [item["series_uid"] for item in batch],
        }


__all__ = [
    "Stage2CandidateDataset",
    "Stage2CropBatch",
    "build_stage1_data_list",
    "clear_volume_cache",
    "missing_scans",
]
