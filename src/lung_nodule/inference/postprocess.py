"""Candidate extraction, diameter estimation and 3D NMS.

This is deliberately pure NumPy/Pandas: the same functions post-process logits
coming from PyTorch, from ONNX Runtime, or from Triton.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage

from lung_nodule.data.luna16 import VolumeGeometry
from lung_nodule.utils.arrays import crop_3d

CANDIDATE_COLUMNS = (
    "seriesuid",
    "coordX",
    "coordY",
    "coordZ",
    "diameter_mm",
    "score",
    "score_stage1",
    "voxel_z",
    "voxel_y",
    "voxel_x",
)

SCORE_MODES = ("heatmap", "mask", "heatmap_x_mask")


@dataclass
class CandidateParams:
    heatmap_threshold: float = 0.35
    peak_window: int = 5
    score_mode: str = "heatmap_x_mask"
    mask_threshold: float = 0.5
    mask_radius_vox: int = 8
    nms_distance_mm: float = 5.0
    max_candidates_per_scan: int = 64
    min_diameter_mm: float = 3.0
    max_diameter_mm: float = 40.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> CandidateParams:
        mapping = mapping or {}
        params = cls(
            heatmap_threshold=float(mapping.get("heatmap_threshold", 0.35)),
            peak_window=int(mapping.get("peak_window", 5)),
            score_mode=str(mapping.get("score_mode", "heatmap_x_mask")),
            mask_threshold=float(mapping.get("mask_threshold", 0.5)),
            mask_radius_vox=int(mapping.get("mask_radius_vox", 8)),
            nms_distance_mm=float(mapping.get("nms_distance_mm", 5.0)),
            max_candidates_per_scan=int(mapping.get("max_candidates_per_scan", 64)),
            min_diameter_mm=float(mapping.get("min_diameter_mm", 3.0)),
            max_diameter_mm=float(mapping.get("max_diameter_mm", 40.0)),
        )
        if params.score_mode not in SCORE_MODES:
            raise ValueError(f"score_mode must be one of {SCORE_MODES}, got {params.score_mode!r}")
        return params


def extract_candidates(
    mask_prob: np.ndarray,
    center_prob: np.ndarray,
    geometry: VolumeGeometry,
    params: CandidateParams,
    series_uid: str = "",
) -> pd.DataFrame:
    """Local maxima of the centre heatmap -> world-space candidates with diameters."""
    mask_prob = _squeeze3d(mask_prob)
    center_prob = _squeeze3d(center_prob)
    if mask_prob.shape != center_prob.shape:
        raise ValueError(f"mask/centre shapes differ: {mask_prob.shape} vs {center_prob.shape}")

    coords, peak_scores = _peak_components(center_prob, params.peak_window, params.heatmap_threshold)
    if coords.size == 0:
        return empty_candidates()

    mask_support = mask_prob[coords[:, 0], coords[:, 1], coords[:, 2]]
    if params.score_mode == "mask":
        scores = mask_support
    elif params.score_mode == "heatmap_x_mask":
        scores = peak_scores * mask_support
    else:
        scores = peak_scores

    world = geometry.voxel_to_world_batch(coords)
    diameters = np.asarray(
        [estimate_diameter(mask_prob, coord, params, geometry.spacing) for coord in coords],
        dtype=np.float32,
    )

    frame = pd.DataFrame(
        {
            "seriesuid": series_uid,
            "coordX": world[:, 0],
            "coordY": world[:, 1],
            "coordZ": world[:, 2],
            "diameter_mm": diameters,
            "score_stage1": scores.astype(np.float32),
            "score": scores.astype(np.float32),
            "voxel_z": coords[:, 0].astype(np.float32),
            "voxel_y": coords[:, 1].astype(np.float32),
            "voxel_x": coords[:, 2].astype(np.float32),
        }
    )
    frame = frame.loc[frame["diameter_mm"] >= params.min_diameter_mm]
    frame = frame.sort_values("score", ascending=False).head(max(1, params.max_candidates_per_scan))
    return frame.loc[:, list(CANDIDATE_COLUMNS)].reset_index(drop=True)


def estimate_diameter(
    mask_prob: np.ndarray,
    voxel_center: np.ndarray,
    params: CandidateParams,
    spacing: tuple[float, float, float],
) -> float:
    """Equivalent-sphere diameter of the thresholded mask in a small neighbourhood."""
    radius = int(max(1, params.mask_radius_vox))
    patch = crop_3d(mask_prob, voxel_center, (2 * radius + 1,) * 3)
    voxels = float(np.count_nonzero(patch >= params.mask_threshold))
    if voxels <= 0:
        return float(params.min_diameter_mm)
    voxel_volume_mm3 = float(np.prod(np.asarray(spacing, dtype=np.float64)))
    diameter = 2.0 * (3.0 * voxels * voxel_volume_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
    return float(np.clip(diameter, params.min_diameter_mm, params.max_diameter_mm))


def distance_nms(frame: pd.DataFrame, min_distance_mm: float, score_column: str = "score") -> pd.DataFrame:
    """Greedy 3D non-maximum suppression by Euclidean centre distance."""
    if frame.empty:
        return frame
    ordered = frame.sort_values(score_column, ascending=False).reset_index(drop=True)
    points = ordered[["coordX", "coordY", "coordZ"]].to_numpy(dtype=np.float64)
    suppressed = np.zeros(len(ordered), dtype=bool)
    keep: list[int] = []

    for index in range(len(ordered)):
        if suppressed[index]:
            continue
        keep.append(index)
        distances = np.linalg.norm(points - points[index], axis=1)
        suppressed |= distances < float(min_distance_mm)

    return ordered.iloc[keep].reset_index(drop=True)


def postprocess_candidates(
    mask_prob: np.ndarray,
    center_prob: np.ndarray,
    geometry: VolumeGeometry,
    params: CandidateParams,
    series_uid: str = "",
) -> pd.DataFrame:
    """Extract peaks, estimate diameters, then apply distance NMS."""
    candidates = extract_candidates(mask_prob, center_prob, geometry, params, series_uid=series_uid)
    if candidates.empty:
        return candidates
    candidates = distance_nms(candidates, params.nms_distance_mm)
    candidates = candidates.head(max(1, params.max_candidates_per_scan))
    return candidates.loc[:, list(CANDIDATE_COLUMNS)].reset_index(drop=True)


def empty_candidates() -> pd.DataFrame:
    """Column-correct empty candidate table (string UID, float32 geometry/score)."""
    frame = pd.DataFrame({column: pd.Series(dtype="float32") for column in CANDIDATE_COLUMNS})
    frame["seriesuid"] = pd.Series(dtype="object")
    return frame.loc[:, list(CANDIDATE_COLUMNS)]


def _peak_components(
    heatmap: np.ndarray,
    window: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima as connected components, returning integer voxel centres."""
    local_max = ndimage.maximum_filter(heatmap, size=max(1, int(window)), mode="nearest")
    candidate = (heatmap >= float(threshold)) & (heatmap >= local_max - 1e-6)
    labeled, count = ndimage.label(candidate)
    if count == 0:
        return np.zeros((0, 3), dtype=int), np.zeros(0, dtype=np.float32)

    index = np.arange(1, count + 1)
    centers = np.asarray(ndimage.center_of_mass(candidate, labeled, index), dtype=np.float64)
    scores = np.asarray(ndimage.maximum(heatmap, labels=labeled, index=index), dtype=np.float32)
    coords = np.rint(centers).astype(int)

    # Two components can round to the same voxel; keep the strongest one.
    best: dict[tuple[int, int, int], float] = {}
    for coord, score in zip(coords, scores, strict=True):
        key = (int(coord[0]), int(coord[1]), int(coord[2]))
        best[key] = max(best.get(key, -np.inf), float(score))

    ordered = sorted(best.items(), key=lambda item: item[1], reverse=True)
    return (
        np.asarray([key for key, _ in ordered], dtype=int),
        np.asarray([value for _, value in ordered], dtype=np.float32),
    )


def _squeeze3d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"expected a (D, H, W) array, got shape {array.shape}")
    return array


__all__ = [
    "CANDIDATE_COLUMNS",
    "CandidateParams",
    "distance_nms",
    "empty_candidates",
    "estimate_diameter",
    "extract_candidates",
    "postprocess_candidates",
]
