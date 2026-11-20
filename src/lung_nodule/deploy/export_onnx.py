"""Export both stages to ONNX and stage the graphs in the Triton model repository.

Layout produced::

    deploy/triton/model_repository/
    ├── stage1_swinunetr/1/model.onnx      + ../model_metadata.json
    └── stage2_fp_reducer/1/model.onnx     + ../model_metadata.json

Stage 1 is exported with a *dynamic batch* but a fixed 96^3 spatial shape: the
Swin window partitioning reshapes are static, and Triton tiles full volumes into
fixed windows anyway. Stage 2 exports with dynamic batch over a fixed 48^3 crop.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lung_nodule.models.stage1_detector import Stage1ExportWrapper, load_stage1
from lung_nodule.models.stage2_classifier import Stage2ExportWrapper, load_stage2
from lung_nodule.utils.config import load_config
from lung_nodule.utils.io import ensure_dir, git_sha, sha256_file, utc_now_iso, write_json
from lung_nodule.utils.logging import setup_logging

STAGE1_MODEL_NAME = "stage1_swinunetr"
STAGE2_MODEL_NAME = "stage2_fp_reducer"
STAGE1_INPUT = "image"
STAGE1_OUTPUTS = ("mask_logits", "center_logits")
STAGE2_INPUT = "crop"
STAGE2_OUTPUTS = ("logit", "diameter")


@dataclass
class ExportResult:
    stage: str
    path: str
    sha256: str
    opset: int
    fp16: bool
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    input_shape: tuple[int, ...]
    verification: dict[str, float] | None = None


def export_stage1(
    checkpoint: str | Path,
    out_path: str | Path,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    opset: int = 17,
    fp16: bool = False,
    verify: bool = True,
) -> ExportResult:
    model = load_stage1(checkpoint, device="cpu", patch_size=patch_size)
    wrapper = Stage1ExportWrapper(model).eval()
    dummy = torch.randn(1, 1, *patch_size)

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=[STAGE1_INPUT],
        output_names=list(STAGE1_OUTPUTS),
        dynamic_axes={
            STAGE1_INPUT: {0: "batch"},
            STAGE1_OUTPUTS[0]: {0: "batch"},
            STAGE1_OUTPUTS[1]: {0: "batch"},
        },
        opset_version=int(opset),
        do_constant_folding=True,
        dynamo=False,
    )
    if fp16:
        convert_to_fp16(out_path)

    verification = verify_onnx(out_path, wrapper, dummy) if verify else None
    return ExportResult(
        stage="stage1",
        path=str(out_path),
        sha256=sha256_file(out_path),
        opset=int(opset),
        fp16=bool(fp16),
        input_names=(STAGE1_INPUT,),
        output_names=STAGE1_OUTPUTS,
        input_shape=(1, 1, *patch_size),
        verification=verification,
    )


def export_stage2(
    checkpoint: str | Path,
    out_path: str | Path,
    crop_size: tuple[int, int, int] = (48, 48, 48),
    in_channels: int | None = None,
    opset: int = 17,
    fp16: bool = False,
    verify: bool = True,
) -> ExportResult:
    model = load_stage2(checkpoint, device="cpu", crop_size=crop_size)
    if in_channels is not None and int(in_channels) != model.config.in_channels:
        raise ValueError(
            f"in_channels={in_channels} does not match the checkpoint ({model.config.in_channels}); "
            "update stage2.model.in_channels in the pipeline config"
        )
    in_channels = model.config.in_channels
    wrapper = Stage2ExportWrapper(model).eval()
    dummy = torch.randn(1, in_channels, *crop_size)

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=[STAGE2_INPUT],
        output_names=list(STAGE2_OUTPUTS),
        dynamic_axes={
            STAGE2_INPUT: {0: "batch"},
            STAGE2_OUTPUTS[0]: {0: "batch"},
            STAGE2_OUTPUTS[1]: {0: "batch"},
        },
        opset_version=int(opset),
        do_constant_folding=True,
        dynamo=False,
    )
    if fp16:
        convert_to_fp16(out_path)

    verification = verify_onnx(out_path, wrapper, dummy) if verify else None
    return ExportResult(
        stage="stage2",
        path=str(out_path),
        sha256=sha256_file(out_path),
        opset=int(opset),
        fp16=bool(fp16),
        input_names=(STAGE2_INPUT,),
        output_names=STAGE2_OUTPUTS,
        input_shape=(1, in_channels, *crop_size),
        verification=verification,
    )


def convert_to_fp16(path: str | Path) -> None:
    """Convert weights/activations to fp16 while keeping fp32 I/O for Triton."""
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(path))
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(converted, str(path))


def verify_onnx(path: str | Path, wrapper: torch.nn.Module, sample: torch.Tensor) -> dict[str, float]:
    """Compare ONNX Runtime against PyTorch on one random input."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: sample.numpy()})

    with torch.no_grad():
        expected = wrapper(sample)
    expected = expected if isinstance(expected, (tuple, list)) else (expected,)

    metrics: dict[str, float] = {}
    cosines: list[float] = []
    for index, (reference, actual) in enumerate(zip(expected, outputs, strict=True)):
        reference_np = reference.detach().numpy().ravel().astype(np.float64)
        actual_np = np.asarray(actual, dtype=np.float64).ravel()
        if reference_np.size != actual_np.size:
            raise RuntimeError(f"output {index} shape mismatch: {reference_np.size} vs {actual_np.size}")
        diff = float(np.abs(reference_np - actual_np).max())
        cosine = float(
            np.dot(reference_np, actual_np)
            / (np.linalg.norm(reference_np) * np.linalg.norm(actual_np) + 1e-12)
        )
        metrics[f"max_abs_diff_{index}"] = diff
        metrics[f"cosine_{index}"] = cosine
        cosines.append(cosine)

    metrics["min_cosine"] = min(cosines) if cosines else float("nan")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="pipeline.yaml (shapes and opset)")
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--model-repository", default="deploy/triton/model_repository")
    parser.add_argument("--stage", choices=("both", "stage1", "stage2"), default="both")
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--report", default="artifacts/export/report.json")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging(args.log_level, name="lung_nodule.export", force=True)
    cfg = load_config(args.config)

    opset = int(args.opset or cfg.get_path("onnx.opset", 17))
    verify = not args.no_verify and bool(cfg.get_path("onnx.verify", True))
    repository = Path(args.model_repository)
    patch_size = tuple(int(v) for v in cfg.get_path("stage1.patch_size", (96, 96, 96)))
    crop_size = tuple(int(v) for v in cfg.get_path("stage2.crop_size", (48, 48, 48)))
    in_channels = int(cfg.get_path("stage2.model.in_channels", 2))

    results: list[ExportResult] = []
    if args.stage in {"both", "stage1"}:
        result = export_stage1(
            args.stage1_checkpoint,
            repository / STAGE1_MODEL_NAME / "1" / "model.onnx",
            patch_size=patch_size,
            opset=opset,
            fp16=args.fp16,
            verify=verify,
        )
        results.append(result)
        logger.info("stage-1 exported: %s", result.path)

    if args.stage in {"both", "stage2"}:
        result = export_stage2(
            args.stage2_checkpoint,
            repository / STAGE2_MODEL_NAME / "1" / "model.onnx",
            crop_size=crop_size,
            in_channels=in_channels,
            opset=opset,
            fp16=args.fp16,
            verify=verify,
        )
        results.append(result)
        logger.info("stage-2 exported: %s", result.path)

    report: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "git_sha": git_sha(),
        "opset": opset,
        "fp16": bool(args.fp16),
        "model_repository": str(repository),
        "exports": [asdict(result) for result in results],
    }
    for result in results:
        if result.verification is not None and result.verification["min_cosine"] < 0.9999:
            logger.warning(
                "%s ONNX parity below 0.9999 (min cosine %.6f) - check unsupported ops",
                result.stage,
                result.verification["min_cosine"],
            )
        write_json(
            Path(result.path).parent.parent / "model_metadata.json",
            {
                "created_at": report["created_at"],
                "git_sha": report["git_sha"],
                "stage": result.stage,
                "sha256": result.sha256,
                "opset": result.opset,
                "fp16": result.fp16,
                "input_shape": list(result.input_shape),
                "verification": result.verification,
            },
        )

    write_json(args.report, report)
    logger.info("export report -> %s", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
