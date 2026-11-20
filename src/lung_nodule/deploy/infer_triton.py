"""Run a cohort through NVIDIA Triton Inference Server.

The Triton path reuses :class:`lung_nodule.inference.pipeline.NodulePipeline`
verbatim, so candidate extraction, diameter estimation, NMS and score fusion are
bit-identical to the local PyTorch path. Differences can therefore only come from
the exported graphs themselves, which is exactly what the FROC gate measures.

Example::

    python -m lung_nodule.deploy.infer_triton \
        --config configs/triton_client.yaml --pipeline-config configs/pipeline.yaml \
        --images-dir data/raw/luna16/subset0 --split configs/splits/luna16_fold0.json \
        --output artifacts/predictions/luna16_val_triton.csv --workers 4
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from lung_nodule.data.luna16 import find_series_file, load_split, read_annotations
from lung_nodule.deploy.triton_runners import TritonError, build_triton_runners
from lung_nodule.inference.pipeline import NodulePipeline, PipelineSettings
from lung_nodule.inference.postprocess import empty_candidates
from lung_nodule.utils.config import load_config
from lung_nodule.utils.io import ensure_dir, git_sha, utc_now_iso, write_json
from lung_nodule.utils.logging import setup_logging


def predict_concurrently(
    pipeline: NodulePipeline,
    series_uids: list[str],
    images_dir: str | Path,
    workers: int,
) -> tuple[pd.DataFrame, list[float]]:
    """Fan scans out over threads; each thread gets its own gRPC client."""
    timings: list[float] = []
    frames: list[pd.DataFrame] = []

    def run(uid: str) -> tuple[str, pd.DataFrame, float]:
        started = time.time()
        try:
            frame = pipeline.predict_scan(find_series_file(images_dir, uid))
        except Exception:
            pipeline.logger.exception("triton inference failed for %s", uid)
            frame = empty_candidates()
        frame["seriesuid"] = uid
        return uid, frame, time.time() - started

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for _uid, frame, elapsed in tqdm(pool.map(run, series_uids), total=len(series_uids), desc="triton"):
            frames.append(frame)
            timings.append(elapsed)

    usable = [frame for frame in frames if not frame.empty]
    predictions = pd.concat(usable, ignore_index=True) if usable else empty_candidates()
    return predictions, timings


def fetch_prometheus_metrics(url: str, timeout_s: float = 5.0) -> str | None:
    """Best-effort snapshot of Triton's Prometheus endpoint."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="triton_client.yaml")
    parser.add_argument("--pipeline-config", required=True, help="pipeline.yaml (post-processing)")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--split-key", default="val", choices=("train", "val"))
    parser.add_argument("--annotations", default=None, help="Use every annotated scan as the cohort")
    parser.add_argument("--series-uid", action="append", default=[])
    parser.add_argument("--output", required=True, help="Predictions CSV")
    parser.add_argument("--url", default=None, help="Override the Triton URL from the config")
    parser.add_argument(
        "--workers", type=int, default=1, help="Concurrent scans (>=2 needs a multi-instance model)"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metrics", action="store_true", help="Snapshot Triton Prometheus metrics")
    parser.add_argument("--override", dest="overrides", action="append", default=[])
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging(args.log_level, name="lung_nodule.triton", force=True)

    pipeline_cfg = load_config(args.pipeline_config)
    triton_cfg = load_config(args.config, overrides=args.overrides)
    settings = PipelineSettings.from_config(pipeline_cfg)

    try:
        stage1, stage2, factory, status = build_triton_runners(triton_cfg, url=args.url)
    except TritonError as exc:
        logger.error("%s", exc)
        return 2

    pipeline = NodulePipeline(stage1, stage2, settings, logger=logger)

    if args.series_uid:
        series_uids = list(args.series_uid)
    elif args.split:
        series_uids = load_split(args.split)[args.split_key]
    elif args.annotations:
        series_uids = sorted(read_annotations(args.annotations)["seriesuid"].unique().tolist())
    else:
        series_uids = sorted(path.stem for path in Path(args.images_dir).glob("*.mhd"))
    if args.limit:
        series_uids = series_uids[: args.limit]
    if not series_uids:
        logger.error("empty cohort: pass --split, --annotations or --series-uid")
        return 1

    started = time.time()
    if args.workers > 1:
        predictions, timings = predict_concurrently(pipeline, series_uids, args.images_dir, args.workers)
    else:
        timings = []
        predictions = pipeline.predict_cohort(series_uids, args.images_dir, args.output, timings=timings)
    wall_time = time.time() - started

    output = Path(args.output)
    if args.workers > 1:
        from lung_nodule.data.luna16 import write_predictions

        write_predictions(predictions, output)

    summary: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "git_sha": git_sha(),
        "backend": "triton",
        "server": status,
        "triton_config": str(args.config),
        "pipeline_config": str(args.pipeline_config),
        "series": len(series_uids),
        "candidates": len(predictions),
        "workers": int(args.workers),
        "wall_time_s": round(wall_time, 2),
        "latency_s": {
            "p50": percentile(timings, 50),
            "p90": percentile(timings, 90),
            "p99": percentile(timings, 99),
        },
        "predictions_csv": str(output.resolve()),
    }

    if args.metrics:
        metrics_url = str(triton_cfg.get_path("metrics.url", "localhost:8002")) + "/metrics"
        snapshot = fetch_prometheus_metrics(metrics_url)
        if snapshot is not None:
            metrics_path = ensure_dir(output.parent) / f"{output.stem}.triton_metrics.txt"
            metrics_path.write_text(snapshot, encoding="utf-8")
            summary["metrics_snapshot"] = str(metrics_path)
        else:
            logger.warning("could not read Prometheus metrics from %s", metrics_url)

    write_json(output.with_name(output.stem + ".summary.json"), summary)
    logger.info(
        "triton: %d scans, %d candidates, p50 %.2fs -> %s",
        len(series_uids),
        len(predictions),
        summary["latency_s"]["p50"] or float("nan"),
        output,
    )
    factory.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
