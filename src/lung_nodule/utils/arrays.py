"""NumPy array helpers for 3D CT patches.

All volumes here are plain arrays in ``(D, H, W)`` order; the channel dimension
is only added right before a tensor is handed to a network.
"""

from __future__ import annotations

import numpy as np

DEFAULT_HU_WINDOW = (-1000.0, 400.0)


def normalize_hu(volume: np.ndarray, hu_min: float = -1000.0, hu_max: float = 400.0) -> np.ndarray:
    """Clip a Hounsfield-unit volume to a lung window and rescale to ``[0, 1]``."""
    volume = np.asarray(volume, dtype=np.float32)
    lo, hi = float(hu_min), float(hu_max)
    if hi <= lo:
        raise ValueError(f"invalid HU window ({lo}, {hi})")
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0, out=volume)


def crop_3d(
    volume: np.ndarray,
    center: tuple[float, float, float] | np.ndarray,
    size: tuple[int, int, int] | np.ndarray,
    fill: float = 0.0,
) -> np.ndarray:
    """Crop ``size`` voxels around a (possibly non-integer) centre with zero padding."""
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {volume.shape}")
    size_arr = np.asarray(size, dtype=int)
    center_arr = np.asarray(center, dtype=np.float64).reshape(3)
    start = np.floor(center_arr - (size_arr - 1) / 2.0).astype(int)

    out = np.full(tuple(int(s) for s in size_arr), fill, dtype=volume.dtype)
    src_start = np.maximum(start, 0)
    src_end = np.minimum(start + size_arr, np.asarray(volume.shape))
    if np.any(src_end <= src_start):
        return out

    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    out[tuple(slice(int(s), int(e)) for s, e in zip(dst_start, dst_end, strict=True))] = volume[
        tuple(slice(int(s), int(e)) for s, e in zip(src_start, src_end, strict=True))
    ]
    return out


def sphere_mask(
    shape: tuple[int, int, int],
    center: tuple[float, float, float] | np.ndarray,
    radius: float,
) -> np.ndarray:
    """Boolean sphere inside a volume of ``shape`` (radius in voxels)."""
    if radius <= 0:
        return np.zeros(shape, dtype=bool)
    grids = np.ogrid[tuple(slice(0, int(s)) for s in shape)]
    dist_sq = np.zeros(shape, dtype=np.float32)
    for axis, grid in enumerate(grids):
        dist_sq += (grid - float(center[axis])) ** 2
    return dist_sq <= float(radius) ** 2


def gaussian_3d(
    shape: tuple[int, int, int],
    center: tuple[float, float, float] | np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Normalised 3D Gaussian blob used as the stage-1 centre heatmap target."""
    sigma = max(float(sigma), 1e-3)
    grids = np.ogrid[tuple(slice(0, int(s)) for s in shape)]
    dist_sq = np.zeros(shape, dtype=np.float32)
    for axis, grid in enumerate(grids):
        dist_sq += (grid - float(center[axis])) ** 2
    heat = np.exp(-dist_sq / (2.0 * sigma * sigma), dtype=np.float32)
    peak = heat.max()
    if peak > 0:
        heat /= peak
    return heat


def ensure_channel_first(array: np.ndarray) -> np.ndarray:
    """``(D, H, W)`` -> ``(1, D, H, W)``; idempotent for 4D input."""
    array = np.asarray(array)
    if array.ndim == 3:
        return array[None, ...]
    if array.ndim == 4:
        return array
    raise ValueError(f"expected 3D or 4D array, got shape {array.shape}")


def bbox_from_mask(mask: np.ndarray) -> tuple[slice, slice, slice]:
    """Bounding-box slices of the foreground; full extent when the mask is empty."""
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return tuple(slice(0, int(s)) for s in mask.shape)  # type: ignore[return-value]
    coords = np.argwhere(mask)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))  # type: ignore[return-value]


__all__ = [
    "DEFAULT_HU_WINDOW",
    "bbox_from_mask",
    "crop_3d",
    "ensure_channel_first",
    "gaussian_3d",
    "normalize_hu",
    "sphere_mask",
]
