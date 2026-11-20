"""Runner interfaces shared by the local PyTorch path and the Triton path.

Keeping both backends behind the same two-method protocol means the candidate
extraction, NMS and scoring code is written (and tested) exactly once.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Stage1Runner(Protocol):
    """Turns a full ``(D, H, W)`` volume into probability maps."""

    patch_size: tuple[int, int, int]

    def predict_volume(self, volume: np.ndarray) -> dict[str, np.ndarray]:
        """Return ``mask_prob`` and ``center_prob`` arrays of the volume shape."""
        ...


@runtime_checkable
class Stage2Runner(Protocol):
    """Scores fixed-size candidate crops."""

    def score_crops(self, crops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(nodule_probability, diameter_mm)`` for ``(N, C, D, H, W)`` crops."""
        ...


__all__ = ["Stage1Runner", "Stage2Runner"]
