"""Data handling: LUNA16 IO, preprocessing, transforms and datasets.

Exports are resolved lazily so that torch-free tooling (FROC evaluation) can import
:mod:`lung_nodule.data.luna16` without pulling in the training stack.
"""

from __future__ import annotations

from lung_nodule.utils.lazy import make_dir, make_lazy_getattr

_EXPORTS = {
    "CandidateSample": ("lung_nodule.data.candidates", "CandidateSample"),
    "Stage2CandidateDataset": ("lung_nodule.data.dataset", "Stage2CandidateDataset"),
    "Stage2CropBatch": ("lung_nodule.data.dataset", "Stage2CropBatch"),
    "VolumeGeometry": ("lung_nodule.data.luna16", "VolumeGeometry"),
    "build_stage1_data_list": ("lung_nodule.data.dataset", "build_stage1_data_list"),
    "load_preprocessed": ("lung_nodule.data.luna16", "load_preprocessed"),
    "load_split": ("lung_nodule.data.luna16", "load_split"),
    "load_training_samples": ("lung_nodule.data.candidates", "load_training_samples"),
    "load_volume": ("lung_nodule.data.luna16", "load_volume"),
    "mine_hard_negatives": ("lung_nodule.data.candidates", "mine_hard_negatives"),
    "read_annotations": ("lung_nodule.data.luna16", "read_annotations"),
    "read_candidates": ("lung_nodule.data.luna16", "read_candidates"),
    "read_predictions": ("lung_nodule.data.luna16", "read_predictions"),
    "save_hard_negatives": ("lung_nodule.data.candidates", "save_hard_negatives"),
    "write_predictions": ("lung_nodule.data.luna16", "write_predictions"),
}

__all__ = sorted(_EXPORTS)
__getattr__ = make_lazy_getattr(__name__, _EXPORTS)
__dir__ = make_dir(globals(), __all__)
