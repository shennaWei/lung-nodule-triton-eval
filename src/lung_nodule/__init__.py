"""Two-stage 3D lung nodule detection pipeline.

Stage 1 is a MONAI ``SwinUNETR`` that turns a 1 mm CT volume into a nodule mask
and a Gaussian centre heatmap. Stage 2 is a small 3D ResNet that rescores the
resulting candidates and suppresses false positives. Both stages can be exported
to ONNX and served by NVIDIA Triton Inference Server; FROC/CPM regression is
tracked against a committed baseline.

The package is intentionally import-light: :mod:`torch` and :mod:`monai` are only
imported by the training/inference modules, so the evaluation tooling works on a
CPU-only CI machine.
"""

from __future__ import annotations

__version__ = "0.3.0"

__all__ = ["__version__"]
