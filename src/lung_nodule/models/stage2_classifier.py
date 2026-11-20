"""Stage 2: compact 3D ResNet false-positive reducer.

Input is a 48^3 crop around a stage-1 candidate (CT values in channel 0, optional
stage-1 mask probability in channel 1). Outputs a nodule logit and, optionally, a
log-diameter regression used as an auxiliary task and for reporting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

ARCHITECTURES: dict[str, tuple[int, int, int, int]] = {
    "resnet18": (2, 2, 2, 2),
    "resnet34": (3, 4, 6, 3),
}

DEFAULT_CROP_SIZE = (48, 48, 48)


def _norm3d(kind: str, channels: int) -> nn.Module:
    kind = kind.lower()
    if kind in {"instance", "in"}:
        # InstanceNorm behaves like GroupNorm(1) and is stable for tiny 3D batches.
        return nn.InstanceNorm3d(channels, affine=True)
    if kind in {"batch", "bn"}:
        return nn.BatchNorm3d(channels)
    if kind == "group":
        return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
    raise ValueError(f"unknown norm layer: {kind!r}")


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, norm: str = "instance") -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = _norm3d(norm, planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = _norm3d(norm, planes)

        self.downsample: nn.Module | None = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                _norm3d(norm, planes),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.activation(out + identity)


class ResNet3D(nn.Module):
    """CIFAR-style 3D ResNet trunk without the classification head."""

    def __init__(
        self,
        arch: str = "resnet18",
        in_channels: int = 1,
        norm: str = "instance",
        stem_channels: int = 32,
    ) -> None:
        super().__init__()
        if arch not in ARCHITECTURES:
            raise ValueError(f"unknown 3D ResNet arch {arch!r}; choose from {sorted(ARCHITECTURES)}")
        layers = ARCHITECTURES[arch]

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _norm3d(norm, stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.inplanes = stem_channels
        self.layer1 = self._make_layer(64, layers[0], stride=1, norm=norm)
        self.layer2 = self._make_layer(128, layers[1], stride=2, norm=norm)
        self.layer3 = self._make_layer(256, layers[2], stride=2, norm=norm)
        self.layer4 = self._make_layer(512, layers[3], stride=2, norm=norm)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.out_features = 512

    def _make_layer(self, planes: int, blocks: int, stride: int, norm: str) -> nn.Sequential:
        layers = [BasicBlock3D(self.inplanes, planes, stride=stride, norm=norm)]
        self.inplanes = planes
        layers.extend(BasicBlock3D(planes, planes, stride=1, norm=norm) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return torch.flatten(self.pool(x), 1)


@dataclass
class Stage2Config:
    arch: str = "resnet18"
    in_channels: int = 2
    num_classes: int = 1
    dropout: float = 0.1
    predict_diameter: bool = True
    norm: str = "instance"
    crop_size: tuple[int, int, int] = DEFAULT_CROP_SIZE

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Stage2Config:
        crop_size = mapping.get("crop_size", DEFAULT_CROP_SIZE)
        return cls(
            arch=str(mapping.get("arch", "resnet18")),
            in_channels=int(mapping.get("in_channels", 2)),
            num_classes=int(mapping.get("num_classes", 1)),
            dropout=float(mapping.get("dropout", 0.1)),
            predict_diameter=bool(mapping.get("predict_diameter", True)),
            norm=str(mapping.get("norm", "instance")),
            crop_size=tuple(int(v) for v in crop_size),  # type: ignore[arg-type]
        )


class Stage2Classifier(nn.Module):
    """3D ResNet with a nodule logit head and an optional diameter head."""

    is_3d = True
    input_key = "image"

    def __init__(self, config: Stage2Config | None = None) -> None:
        super().__init__()
        self.config = config or Stage2Config()
        if self.config.num_classes != 1:
            raise ValueError("the FP reducer is binary; set num_classes: 1")

        self.trunk = ResNet3D(
            arch=self.config.arch,
            in_channels=self.config.in_channels,
            norm=self.config.norm,
        )
        features = self.trunk.out_features
        self.dropout = nn.Dropout(self.config.dropout)
        self.nodule_head = nn.Linear(features, 1)
        self.diameter_head = nn.Linear(features, 1) if self.config.predict_diameter else None

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.dropout(self.trunk(image))
        outputs = {"logit": self.nodule_head(features)}
        if self.diameter_head is not None:
            # log-millimetres: keeps the regression well scaled over 3-40 mm nodules
            outputs["diameter"] = self.diameter_head(features)
        return outputs

    @torch.no_grad()
    def predict_proba(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(image)["logit"]).squeeze(1)


class Stage2ExportWrapper(nn.Module):
    """ONNX wrapper: ``(logit, diameter)`` with the diameter head always present."""

    def __init__(self, model: Stage2Classifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, crop: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(crop)
        diameter = outputs.get("diameter")
        if diameter is None:
            diameter = torch.zeros_like(outputs["logit"])
        return outputs["logit"], diameter


def build_stage2(config: Stage2Config | Mapping[str, Any] | None = None) -> Stage2Classifier:
    if config is None:
        config = Stage2Config()
    elif isinstance(config, Mapping):
        config = Stage2Config.from_mapping(config)
    return Stage2Classifier(config)


def load_stage2(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    crop_size: tuple[int, int, int] | None = None,
) -> Stage2Classifier:
    """Architecture from the checkpoint; ``crop_size`` is informational only."""
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = Stage2Config.from_mapping(_model_section(payload))
    if crop_size is not None:
        config.crop_size = tuple(int(v) for v in crop_size)
    model = Stage2Classifier(config)
    model.load_state_dict(payload.get("model", payload))
    return model.to(device).eval()


def _model_section(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    stored = payload.get("config", {})
    if isinstance(stored, Mapping):
        section = stored.get("model", stored)
        if isinstance(section, Mapping):
            return section
    return {}


__all__ = [
    "ARCHITECTURES",
    "DEFAULT_CROP_SIZE",
    "ResNet3D",
    "Stage2Classifier",
    "Stage2Config",
    "Stage2ExportWrapper",
    "build_stage2",
    "load_stage2",
]
