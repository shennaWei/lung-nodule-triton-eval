"""MONAI transform chains for the two training stages.

Stage 1 samples positive/negative patches from the cached volumes; stage 2
augments small candidate crops. Both chains are intentionally deterministic in
their keys so that image, mask and heatmap stay aligned.
"""

from __future__ import annotations

import json
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureTyped,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    Transform,
)

from lung_nodule.data.luna16 import VolumeGeometry

STAGE1_KEYS = ("image", "mask", "heatmap")
STAGE2_KEYS = ("image", "mask")


class LoadPreprocessedNpzd(Transform):
    """Load a cached ``.npz`` scan and expose channel-first arrays in the dict.

    ``keep_geometry`` is off by default: the geometry object cannot be collated by
    ``monai.data.list_data_collate``, and stage-1 training works in voxel space.
    """

    def __init__(
        self,
        path_key: Hashable = "npz_path",
        keys: Sequence[Hashable] = STAGE1_KEYS,
        keep_geometry: bool = False,
    ) -> None:
        self.path_key = path_key
        self.keys = tuple(keys)
        self.keep_geometry = keep_geometry
        super().__init__()

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        out = dict(data)
        path = str(out[self.path_key])
        with np.load(path, allow_pickle=False) as archive:
            for key in self.keys:
                array = archive[str(key)].astype(np.float32)
                out[key] = array[None, ...]  # (D, H, W) -> (1, D, H, W)
            if self.keep_geometry:
                out["geometry"] = VolumeGeometry.from_dict(json.loads(str(archive["meta"])))
        return out


def build_stage1_train_transforms(
    patch_size: Sequence[int] = (96, 96, 96),
    pos: int = 2,
    neg: int = 1,
    num_samples: int = 2,
    flip_prob: float = 0.5,
    rotate_prob: float = 0.3,
    affine_prob: float = 0.3,
) -> Compose:
    """Random pos/neg patch sampling plus flips, 90-degree rotations and slight warps."""
    keys = list(STAGE1_KEYS)
    return Compose(
        [
            LoadPreprocessedNpzd(),
            # Crop transform from MONAI, imported lazily to keep module import cheap.
            _pos_neg_crop(patch_size=patch_size, pos=pos, neg=neg, num_samples=num_samples, keys=keys),
            # Arrays are channel-first, so the spatial axes are 1..3.
            RandFlipd(keys=keys, prob=flip_prob, spatial_axis=1),
            RandFlipd(keys=keys, prob=flip_prob, spatial_axis=2),
            RandFlipd(keys=keys, prob=flip_prob, spatial_axis=3),
            RandRotate90d(keys=keys, prob=rotate_prob, max_k=3, spatial_axes=(2, 3)),
            RandAffined(
                keys=keys,
                prob=affine_prob,
                rotate_range=(0.15, 0.15, 0.15),
                scale_range=(0.08, 0.08, 0.08),
                translate_range=(4, 4, 4),
                mode=("bilinear", "nearest", "bilinear"),
                padding_mode="zeros",
            ),
            EnsureTyped(keys=keys, dtype=torch.float32),
        ]
    )


def build_stage1_eval_transforms(patch_size: Sequence[int] | None = None) -> Compose:
    """Deterministic loader used for fixed-size patches (export / smoke checks)."""
    transforms: list[Any] = [LoadPreprocessedNpzd()]
    if patch_size is not None:
        transforms.append(_pad_to_patch(patch_size))
    transforms.append(EnsureTyped(keys=list(STAGE1_KEYS), dtype=torch.float32))
    return Compose(transforms)


def build_stage1_val_transforms(
    patch_size: Sequence[int] = (96, 96, 96),
    pos: int = 1,
    neg: int = 1,
    num_samples: int = 1,
) -> Compose:
    """Patch-level validation: same sampling as training, no augmentation.

    This only tracks mask Dice / centre loss. Sensitivity and FP/scan are measured
    end-to-end with ``lndet-predict`` followed by ``lndet-eval-froc``.
    """
    keys = list(STAGE1_KEYS)
    return Compose(
        [
            LoadPreprocessedNpzd(),
            _pos_neg_crop(patch_size=patch_size, pos=pos, neg=neg, num_samples=num_samples, keys=keys),
            EnsureTyped(keys=keys, dtype=torch.float32),
        ]
    )


def build_stage2_transforms(augment: bool = True) -> Compose:
    """Augment 48^3 candidate crops (arrays are prepared by the dataset)."""
    keys = list(STAGE2_KEYS)
    if not augment:
        return Compose([EnsureTyped(keys=keys, dtype=torch.float32)])
    return Compose(
        [
            EnsureTyped(keys=keys, dtype=torch.float32),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=3),
            RandRotate90d(keys=keys, prob=0.35, max_k=3, spatial_axes=(2, 3)),
            RandAffined(
                keys=keys,
                prob=0.3,
                rotate_range=(0.2, 0.2, 0.2),
                scale_range=(0.1, 0.1, 0.1),
                translate_range=(3, 3, 3),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.02),
            EnsureTyped(keys=keys, dtype=torch.float32),
        ]
    )


def _pos_neg_crop(
    patch_size: Sequence[int],
    pos: int,
    neg: int,
    num_samples: int,
    keys: Sequence[str],
):
    from monai.transforms import RandCropByPosNegLabeld

    return RandCropByPosNegLabeld(
        keys=list(keys),
        label_key="mask",
        spatial_size=tuple(int(s) for s in patch_size),
        pos=pos,
        neg=neg,
        num_samples=num_samples,
        image_key="image",
        image_threshold=0.0,
        allow_smaller=True,
    )


def _pad_to_patch(patch_size: Sequence[int]):
    from monai.transforms import SpatialPadd

    return SpatialPadd(keys=list(STAGE1_KEYS), spatial_size=tuple(int(s) for s in patch_size))


__all__ = [
    "STAGE1_KEYS",
    "STAGE2_KEYS",
    "LoadPreprocessedNpzd",
    "build_stage1_eval_transforms",
    "build_stage1_train_transforms",
    "build_stage1_val_transforms",
    "build_stage2_transforms",
]
