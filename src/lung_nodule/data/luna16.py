"""LUNA16 metadata handling: CSV I/O, world/voxel geometry and target rasterisation.

Preprocessed volumes are cached as ``<series_uid>.npz`` containing:

* ``image``   -- ``float16`` volume in ``[0, 1]`` at 1 mm isotropic spacing
* ``mask``    -- ``uint8`` nodule mask
* ``heatmap`` -- ``float16`` Gaussian centre heatmap in ``[0, 1]``
* ``meta``    -- JSON string with origin / spacing / direction / shape
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ANNOTATION_COLUMNS = ("seriesuid", "coordX", "coordY", "coordZ", "diameter_mm")
CANDIDATE_COLUMNS = ("seriesuid", "coordX", "coordY", "coordZ", "class")
PREDICTION_COLUMNS = ("seriesuid", "coordX", "coordY", "coordZ", "diameter_mm", "score")

IMAGE_SUFFIXES = (".mhd", ".mha", ".nii", ".nii.gz")


@dataclass(frozen=True)
class VolumeGeometry:
    """Voxel-to-world mapping of a CT volume (world coordinates in millimetres, LPS)."""

    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    direction: tuple[float, ...]
    shape: tuple[int, int, int]

    @property
    def affine(self) -> np.ndarray:
        """4x4 affine mapping continuous voxel indices to world coordinates."""
        rotation = np.asarray(self.direction, dtype=np.float64).reshape(3, 3)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation @ np.diag(np.asarray(self.spacing, dtype=np.float64))
        matrix[:3, 3] = np.asarray(self.origin, dtype=np.float64)
        return matrix

    def world_to_voxel(self, world: tuple[float, float, float] | np.ndarray) -> np.ndarray:
        point = np.asarray(world, dtype=np.float64).reshape(3)
        inverse = np.linalg.inv(self.affine)
        return inverse[:3, :3] @ point + inverse[:3, 3]

    def voxel_to_world(self, index: tuple[float, float, float] | np.ndarray) -> np.ndarray:
        point = np.asarray(index, dtype=np.float64).reshape(3)
        return self.affine[:3, :3] @ point + self.affine[:3, 3]

    def voxel_to_world_batch(self, indices: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`voxel_to_world` for an ``(N, 3)`` array of indices."""
        points = np.asarray(indices, dtype=np.float64).reshape(-1, 3)
        affine = self.affine
        return points @ affine[:3, :3].T + affine[:3, 3]

    def crop_bounds(
        self,
        center_world: tuple[float, float, float] | np.ndarray,
        size_voxels: tuple[int, int, int],
    ) -> tuple[slice, slice, slice]:
        """Integer bounding box around a world-space point, clamped to the volume."""
        center = self.world_to_voxel(center_world)
        half = (np.asarray(size_voxels, dtype=np.float64) - 1.0) / 2.0
        lo = np.floor(center - half).astype(int)
        hi = lo + np.asarray(size_voxels, dtype=int)
        return tuple(
            slice(int(max(a, 0)), int(min(b, s))) for a, b, s in zip(lo, hi, self.shape, strict=True)
        )  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": list(self.origin),
            "spacing": list(self.spacing),
            "direction": list(self.direction),
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VolumeGeometry:
        return cls(
            origin=tuple(float(v) for v in payload["origin"]),  # type: ignore[arg-type]
            spacing=tuple(float(v) for v in payload["spacing"]),  # type: ignore[arg-type]
            direction=tuple(float(v) for v in payload["direction"]),
            shape=tuple(int(v) for v in payload["shape"]),  # type: ignore[arg-type]
        )

    @classmethod
    def from_sitk(cls, image: Any) -> VolumeGeometry:
        return cls(
            origin=tuple(float(v) for v in image.GetOrigin()),  # type: ignore[arg-type]
            spacing=tuple(float(v) for v in image.GetSpacing()),  # type: ignore[arg-type]
            direction=tuple(float(v) for v in image.GetDirection()),
            shape=tuple(int(v) for v in image.GetSize()[::-1]),  # type: ignore[arg-type]
        )


def read_annotations(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(frame, ANNOTATION_COLUMNS, path)
    frame = frame.loc[:, list(ANNOTATION_COLUMNS)].copy()
    frame["seriesuid"] = frame["seriesuid"].astype(str)
    for column in ("coordX", "coordY", "coordZ", "diameter_mm"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["coordX", "coordY", "coordZ", "diameter_mm"])
    return frame.reset_index(drop=True)


def read_candidates(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(frame, CANDIDATE_COLUMNS, path)
    frame = frame.loc[:, list(CANDIDATE_COLUMNS)].copy()
    frame["seriesuid"] = frame["seriesuid"].astype(str)
    frame["class"] = frame["class"].astype(int)
    return frame.reset_index(drop=True)


def read_predictions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(frame, ("seriesuid", "coordX", "coordY", "coordZ", "score"), path)
    frame["seriesuid"] = frame["seriesuid"].astype(str)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0)
    if "diameter_mm" not in frame.columns:
        frame["diameter_mm"] = 0.0
    frame["diameter_mm"] = pd.to_numeric(frame["diameter_mm"], errors="coerce").fillna(0.0)
    return frame.loc[:, list(PREDICTION_COLUMNS)].reset_index(drop=True)


def write_predictions(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    for column in PREDICTION_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0 if column in {"diameter_mm", "score"} else ""
    out = out.sort_values(["seriesuid", "score"], ascending=[True, False])
    out.loc[:, list(PREDICTION_COLUMNS)].to_csv(target, index=False, float_format="%.4f")
    return target


def find_series_file(images_dir: str | Path, series_uid: str) -> Path:
    """Locate ``<series_uid>.mhd`` / ``.nii.gz`` inside ``images_dir``."""
    images_dir = Path(images_dir)
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{series_uid}{suffix}"
        if candidate.exists():
            return candidate
    matches = [p for suffix in IMAGE_SUFFIXES for p in images_dir.rglob(f"{series_uid}{suffix}")]
    if not matches:
        raise FileNotFoundError(f"no image found for series {series_uid} under {images_dir}")
    return matches[0]


def load_volume(path: str | Path) -> tuple[np.ndarray, VolumeGeometry]:
    """Read a CT scan as ``(D, H, W)`` float32 Hounsfield units plus its geometry."""
    import SimpleITK as sitk  # imported lazily: evaluation tooling does not need it

    image = sitk.ReadImage(str(path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)  # SimpleITK is (x, y, z)
    return volume, VolumeGeometry.from_sitk(image)


def load_preprocessed(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, VolumeGeometry]:
    """Load a cached ``npz`` written by :mod:`lung_nodule.data.preprocess`."""
    with np.load(path, allow_pickle=False) as archive:
        image = archive["image"].astype(np.float32)
        mask = archive["mask"]
        heatmap = archive["heatmap"].astype(np.float32)
        geometry = VolumeGeometry.from_dict(json.loads(str(archive["meta"])))
    return image, mask, heatmap, geometry


def annotations_for(annotations: pd.DataFrame, series_uid: str) -> pd.DataFrame:
    return annotations.loc[annotations["seriesuid"] == series_uid].reset_index(drop=True)


def rasterize_annotations(
    shape: tuple[int, int, int],
    geometry: VolumeGeometry,
    annotations: pd.DataFrame,
    heatmap_sigma_scale: float = 0.35,
    min_sigma_mm: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw nodule spheres and Gaussian centres for one scan.

    ``heatmap_sigma_scale`` sets the Gaussian sigma as a fraction of the nodule
    diameter (0.35 keeps the heatmap compact around the centre while remaining
    easy to regress).
    """
    mask = np.zeros(shape, dtype=np.uint8)
    heatmap = np.zeros(shape, dtype=np.float32)
    spacing = np.asarray(geometry.spacing, dtype=np.float64)

    for row in annotations.itertuples(index=False):
        world = np.asarray([row.coordX, row.coordY, row.coordZ], dtype=np.float64)
        radius_mm = float(row.diameter_mm) / 2.0
        if radius_mm <= 0:
            continue
        center = geometry.world_to_voxel(world)
        radius_vox = radius_mm / spacing
        lo = np.floor(center - radius_vox - 3.0).astype(int)
        hi = np.ceil(center + radius_vox + 3.0).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(shape))
        if np.any(hi <= lo):
            continue

        slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
        grids = np.ogrid[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))]
        dist_mm_sq = np.zeros(tuple(int(b - a) for a, b in zip(lo, hi, strict=True)), dtype=np.float64)
        for axis, grid in enumerate(grids):
            dist_mm_sq += ((grid - center[axis]) * spacing[axis]) ** 2
        dist_mm = np.sqrt(dist_mm_sq)

        mask[slices] = np.maximum(mask[slices], (dist_mm <= radius_mm).astype(np.uint8))
        sigma = max(float(row.diameter_mm) * heatmap_sigma_scale, min_sigma_mm)
        blob = np.exp(-(dist_mm**2) / (2.0 * sigma * sigma)).astype(np.float32)
        heatmap[slices] = np.maximum(heatmap[slices], blob)

    return mask, heatmap


def load_split(path: str | Path) -> dict[str, list[str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "train" not in payload or "val" not in payload:
        raise ValueError(f"{path} must define 'train' and 'val' series UID lists")
    train = [str(uid) for uid in payload["train"]]
    val = [str(uid) for uid in payload["val"]]
    overlap = set(train) & set(val)
    if overlap:
        raise ValueError(f"split leaks {len(overlap)} scans between train and val")
    return {"train": train, "val": val}


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], path: str | Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")


__all__ = [
    "ANNOTATION_COLUMNS",
    "CANDIDATE_COLUMNS",
    "PREDICTION_COLUMNS",
    "VolumeGeometry",
    "annotations_for",
    "find_series_file",
    "load_preprocessed",
    "load_split",
    "load_volume",
    "rasterize_annotations",
    "read_annotations",
    "read_candidates",
    "read_predictions",
    "write_predictions",
]
