"""Stage 1: 3D nodule detector built on MONAI's SwinUNETR.

The network predicts two channels on 96^3 patches:

* ``mask_logits``   -- voxel-wise nodule segmentation (used for diameter / support)
* ``center_logits`` -- Gaussian centre heatmap (used for candidate peaks)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

DEFAULT_PATCH_SIZE = (96, 96, 96)


@dataclass
class Stage1Config:
    in_channels: int = 1
    out_channels: int = 2
    feature_size: int = 48
    patch_size: tuple[int, int, int] = DEFAULT_PATCH_SIZE
    use_checkpoint: bool = False
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0
    spatial_dims: int = 3
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Stage1Config:
        patch_size = mapping.get("patch_size", DEFAULT_PATCH_SIZE)
        return cls(
            in_channels=int(mapping.get("in_channels", 1)),
            out_channels=int(mapping.get("out_channels", 2)),
            feature_size=int(mapping.get("feature_size", 48)),
            patch_size=tuple(int(v) for v in patch_size),  # type: ignore[arg-type]
            use_checkpoint=bool(mapping.get("use_checkpoint", False)),
            drop_rate=float(mapping.get("drop_rate", 0.0)),
            attn_drop_rate=float(mapping.get("attn_drop_rate", 0.0)),
            dropout_path_rate=float(mapping.get("dropout_path_rate", 0.0)),
            spatial_dims=int(mapping.get("spatial_dims", 3)),
        )


class Stage1Detector(nn.Module):
    """SwinUNETR wrapper returning a dict of logits."""

    is_3d = True
    input_key = "image"

    def __init__(self, config: Stage1Config | None = None) -> None:
        super().__init__()
        from monai.networks.nets import SwinUNETR

        self.config = config or Stage1Config()
        if self.config.out_channels < 2:
            raise ValueError("stage 1 needs at least two output channels (mask + centre)")
        patch = self.config.patch_size
        if any(size % 32 != 0 for size in patch):
            raise ValueError(f"patch_size must be divisible by 32 for SwinUNETR, got {patch}")

        self.backbone = SwinUNETR(
            img_size=patch,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            feature_size=self.config.feature_size,
            use_checkpoint=self.config.use_checkpoint,
            spatial_dims=self.config.spatial_dims,
            drop_rate=self.config.drop_rate,
            attn_drop_rate=self.config.attn_drop_rate,
            dropout_path_rate=self.config.dropout_path_rate,
            downsample="merging",
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.backbone(image)
        return {
            "mask_logits": logits[:, :1],
            "center_logits": logits[:, 1:2],
        }

    @torch.no_grad()
    def predict_proba(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.forward(image)
        return {key: torch.sigmoid(value) for key, value in outputs.items()}


class Stage1ExportWrapper(nn.Module):
    """ONNX-friendly wrapper: explicit tensor outputs, no dicts, fixed names."""

    def __init__(self, model: Stage1Detector) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(image)
        return outputs["mask_logits"], outputs["center_logits"]


def build_stage1(config: Stage1Config | Mapping[str, Any] | None = None) -> Stage1Detector:
    if config is None:
        config = Stage1Config()
    elif isinstance(config, Mapping):
        config = Stage1Config.from_mapping(config)
    return Stage1Detector(config)


def load_stage1(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    patch_size: tuple[int, int, int] | None = None,
) -> Stage1Detector:
    """Instantiate and populate a stage-1 model from a training checkpoint.

    The architecture (``feature_size``, channels) always comes from the checkpoint
    so weights and config cannot drift apart. ``patch_size`` only fixes the spatial
    input shape: SwinUNETR's parameters do not depend on ``img_size``, which is what
    lets the exported graph reuse the same weights at a different tiling size.
    """
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = Stage1Config.from_mapping(_model_section(payload))
    config.use_checkpoint = False
    if patch_size is not None:
        config.patch_size = tuple(int(v) for v in patch_size)
    model = Stage1Detector(config)
    model.load_state_dict(payload.get("model", payload))
    return model.to(device).eval()


def _model_section(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """``checkpoint['config']['model']`` when present, else the checkpoint itself."""
    stored = payload.get("config", {})
    if isinstance(stored, Mapping):
        section = stored.get("model", stored)
        if isinstance(section, Mapping):
            return section
    return {}


__all__ = [
    "DEFAULT_PATCH_SIZE",
    "Stage1Config",
    "Stage1Detector",
    "Stage1ExportWrapper",
    "build_stage1",
    "load_stage1",
]
