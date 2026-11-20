"""Deployment: ONNX export and Triton-backed inference (lazy exports)."""

from __future__ import annotations

from lung_nodule.utils.lazy import make_dir, make_lazy_getattr

_EXPORTS = {
    "TritonClientFactory": ("lung_nodule.deploy.triton_runners", "TritonClientFactory"),
    "TritonError": ("lung_nodule.deploy.triton_runners", "TritonError"),
    "TritonStage1Runner": ("lung_nodule.deploy.triton_runners", "TritonStage1Runner"),
    "TritonStage2Runner": ("lung_nodule.deploy.triton_runners", "TritonStage2Runner"),
    "build_triton_runners": ("lung_nodule.deploy.triton_runners", "build_triton_runners"),
    "export_stage1": ("lung_nodule.deploy.export_onnx", "export_stage1"),
    "export_stage2": ("lung_nodule.deploy.export_onnx", "export_stage2"),
    "verify_onnx": ("lung_nodule.deploy.export_onnx", "verify_onnx"),
}

__all__ = sorted(_EXPORTS)
__getattr__ = make_lazy_getattr(__name__, _EXPORTS)
__dir__ = make_dir(globals(), __all__)
