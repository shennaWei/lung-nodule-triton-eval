"""Model factory shared by training, export and inference."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch import nn

from lung_nodule.models.stage1_detector import Stage1Config, Stage1Detector, build_stage1, load_stage1
from lung_nodule.models.stage2_classifier import Stage2Classifier, Stage2Config, build_stage2, load_stage2

MODEL_KINDS = ("stage1", "stage2")


def build_model(kind: str, config: Mapping[str, Any] | None = None) -> nn.Module:
    """Build a stage-1 detector or a stage-2 classifier from a config mapping."""
    if kind == "stage1":
        return build_stage1(dict(config or {}))
    if kind == "stage2":
        return build_stage2(dict(config or {}))
    raise ValueError(f"unknown model kind {kind!r}; expected one of {MODEL_KINDS}")


def load_model(kind: str, checkpoint: str | Path, device: str = "cpu") -> nn.Module:
    if kind == "stage1":
        return load_stage1(checkpoint, device=device)
    if kind == "stage2":
        return load_stage2(checkpoint, device=device)
    raise ValueError(f"unknown model kind {kind!r}; expected one of {MODEL_KINDS}")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "MODEL_KINDS",
    "Stage1Config",
    "Stage1Detector",
    "Stage2Classifier",
    "Stage2Config",
    "build_model",
    "load_model",
    "parameter_count",
    "trainable_parameter_count",
]
