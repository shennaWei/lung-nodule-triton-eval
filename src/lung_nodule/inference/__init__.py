"""Inference: candidate post-processing, local runners and the end-to-end pipeline.

Post-processing itself is pure NumPy and is exported eagerly; the torch-backed
runners are resolved lazily.
"""

from __future__ import annotations

from lung_nodule.inference.postprocess import (
    CANDIDATE_COLUMNS,
    CandidateParams,
    distance_nms,
    extract_candidates,
    postprocess_candidates,
)
from lung_nodule.utils.lazy import make_dir, make_lazy_getattr

_EXPORTS = {
    "NodulePipeline": ("lung_nodule.inference.pipeline", "NodulePipeline"),
    "PipelineSettings": ("lung_nodule.inference.pipeline", "PipelineSettings"),
    "Stage1Runner": ("lung_nodule.inference.base", "Stage1Runner"),
    "Stage2Runner": ("lung_nodule.inference.base", "Stage2Runner"),
    "TorchStage1Runner": ("lung_nodule.inference.torch_runners", "TorchStage1Runner"),
    "TorchStage2Runner": ("lung_nodule.inference.torch_runners", "TorchStage2Runner"),
}

__all__ = [
    "CANDIDATE_COLUMNS",
    "CandidateParams",
    "NodulePipeline",
    "PipelineSettings",
    "Stage1Runner",
    "Stage2Runner",
    "TorchStage1Runner",
    "TorchStage2Runner",
    "distance_nms",
    "extract_candidates",
    "postprocess_candidates",
]
__getattr__ = make_lazy_getattr(__name__, _EXPORTS)
__dir__ = make_dir(globals(), __all__)
