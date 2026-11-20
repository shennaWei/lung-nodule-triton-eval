"""Training utilities for both stages (lazy exports)."""

from __future__ import annotations

from lung_nodule.utils.lazy import make_dir, make_lazy_getattr

_EXPORTS = {
    "MetricTracker": ("lung_nodule.training.engine", "MetricTracker"),
    "Stage1Loss": ("lung_nodule.training.losses", "Stage1Loss"),
    "Stage2Loss": ("lung_nodule.training.losses", "Stage2Loss"),
    "build_optimizer": ("lung_nodule.training.engine", "build_optimizer"),
    "build_scheduler": ("lung_nodule.training.engine", "build_scheduler"),
    "build_stage1_loss": ("lung_nodule.training.losses", "build_stage1_loss"),
    "build_stage2_loss": ("lung_nodule.training.losses", "build_stage2_loss"),
    "evaluate": ("lung_nodule.training.engine", "evaluate"),
    "load_checkpoint": ("lung_nodule.training.engine", "load_checkpoint"),
    "resolve_device": ("lung_nodule.training.engine", "resolve_device"),
    "save_checkpoint": ("lung_nodule.training.engine", "save_checkpoint"),
    "train_one_epoch": ("lung_nodule.training.engine", "train_one_epoch"),
}

__all__ = sorted(_EXPORTS)
__getattr__ = make_lazy_getattr(__name__, _EXPORTS)
__dir__ = make_dir(globals(), __all__)
