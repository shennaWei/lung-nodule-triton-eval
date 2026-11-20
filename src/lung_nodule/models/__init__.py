"""Network definitions for both stages (lazy: importing this package stays cheap)."""

from __future__ import annotations

from lung_nodule.utils.lazy import make_dir, make_lazy_getattr

_EXPORTS = {
    "ResNet3D": ("lung_nodule.models.stage2_classifier", "ResNet3D"),
    "Stage1Config": ("lung_nodule.models.stage1_detector", "Stage1Config"),
    "Stage1Detector": ("lung_nodule.models.stage1_detector", "Stage1Detector"),
    "Stage1ExportWrapper": ("lung_nodule.models.stage1_detector", "Stage1ExportWrapper"),
    "Stage2Classifier": ("lung_nodule.models.stage2_classifier", "Stage2Classifier"),
    "Stage2Config": ("lung_nodule.models.stage2_classifier", "Stage2Config"),
    "Stage2ExportWrapper": ("lung_nodule.models.stage2_classifier", "Stage2ExportWrapper"),
    "build_model": ("lung_nodule.models.factory", "build_model"),
    "build_stage1": ("lung_nodule.models.stage1_detector", "build_stage1"),
    "build_stage2": ("lung_nodule.models.stage2_classifier", "build_stage2"),
    "load_model": ("lung_nodule.models.factory", "load_model"),
    "load_stage1": ("lung_nodule.models.stage1_detector", "load_stage1"),
    "load_stage2": ("lung_nodule.models.stage2_classifier", "load_stage2"),
    "parameter_count": ("lung_nodule.models.factory", "parameter_count"),
}

__all__ = sorted(_EXPORTS)
__getattr__ = make_lazy_getattr(__name__, _EXPORTS)
__dir__ = make_dir(globals(), __all__)
