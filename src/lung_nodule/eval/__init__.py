"""Evaluation: FROC/CPM scoring, regression gating and run tracking.

This subpackage is intentionally free of torch/MONAI so it can run on a minimal
CPU-only runner (see ``eval-cli`` in ``.github/workflows/ci.yml``). Exports are
lazy, which also keeps ``python -m lung_nodule.eval.froc`` free of import warnings.
"""

from __future__ import annotations

from lung_nodule.utils.lazy import make_dir, make_lazy_getattr

_EXPORTS = {
    "FP_LEVELS": ("lung_nodule.eval.froc", "FP_LEVELS"),
    "FrocResult": ("lung_nodule.eval.froc", "FrocResult"),
    "RunRecord": ("lung_nodule.eval.tracking", "RunRecord"),
    "append_run": ("lung_nodule.eval.tracking", "append_run"),
    "binary_auroc": ("lung_nodule.eval.metrics", "binary_auroc"),
    "bootstrap_cpm": ("lung_nodule.eval.froc", "bootstrap_cpm"),
    "compute_froc": ("lung_nodule.eval.froc", "compute_froc"),
    "evaluate_regression": ("lung_nodule.eval.regression", "evaluate_regression"),
    "load_baseline": ("lung_nodule.eval.regression", "load_baseline"),
    "load_history": ("lung_nodule.eval.tracking", "load_history"),
    "make_run_record": ("lung_nodule.eval.tracking", "make_run_record"),
    "match_predictions": ("lung_nodule.eval.froc", "match_predictions"),
    "measure_trend": ("lung_nodule.eval.tracking", "measure_trend"),
    "render_leaderboard": ("lung_nodule.eval.tracking", "render_leaderboard"),
    "save_froc_report": ("lung_nodule.eval.froc", "save_froc_report"),
    "sensitivity_specificity": ("lung_nodule.eval.metrics", "sensitivity_specificity"),
    "update_baseline": ("lung_nodule.eval.regression", "update_baseline"),
}

__all__ = sorted(_EXPORTS)
__getattr__ = make_lazy_getattr(__name__, _EXPORTS)
__dir__ = make_dir(globals(), __all__)
