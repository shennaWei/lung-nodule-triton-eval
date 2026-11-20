#!/usr/bin/env python
"""Smoke-test a running Triton server without any patient data.

Sends random tensors to both models, checks the returned shapes and prints the
observed latency. Useful as a deployment gate in CI and after `make triton-up`.

    python scripts/triton_smoke_test.py --url localhost:8001 --repeat 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lung_nodule.deploy.triton_runners import TritonClientFactory, TritonStage1Runner, TritonStage2Runner
from lung_nodule.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="localhost:8001")
    parser.add_argument("--stage1-model", default="stage1_swinunetr")
    parser.add_argument("--stage2-model", default="stage2_fp_reducer")
    parser.add_argument("--patch-size", default="96,96,96")
    parser.add_argument("--crop-size", default="48,48,48")
    parser.add_argument("--stage2-batch", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = setup_logging(args.log_level, name="triton_smoke", force=True)
    patch = tuple(int(v) for v in args.patch_size.split(","))
    crop = tuple(int(v) for v in args.crop_size.split(","))

    factory = TritonClientFactory(url=args.url, client_timeout_s=60)
    try:
        status = factory.ensure_ready([args.stage1_model, args.stage2_model])
    except Exception as exc:
        logger.error("server not ready: %s", exc)
        return 2

    logger.info("server ready: %s", status["server"])
    stage1 = TritonStage1Runner(factory, model_name=args.stage1_model, patch_size=patch, max_batch=1)
    stage2 = TritonStage2Runner(factory, model_name=args.stage2_model, max_batch=args.stage2_batch)

    volume = np.random.default_rng(0).random(patch, dtype=np.float32)

    stage1_latencies: list[float] = []
    for repeat in range(max(1, args.repeat)):
        started = time.time()
        probabilities = stage1.predict_volume(volume)
        stage1_latencies.append(time.time() - started)
        for name, array in probabilities.items():
            if array.shape != patch:
                logger.error("stage-1 output %s has shape %s, expected %s", name, array.shape, patch)
                return 1
        logger.info(
            "stage-1 run %d: mask max %.3f, centre max %.3f (%.2fs)",
            repeat,
            float(probabilities["mask_prob"].max()),
            float(probabilities["center_prob"].max()),
            stage1_latencies[-1],
        )

    crops = np.random.default_rng(1).random((args.stage2_batch, 2, *crop), dtype=np.float32)
    started = time.time()
    probabilities, diameters = stage2.score_crops(crops)
    stage2_latency = time.time() - started
    if probabilities.shape != (args.stage2_batch,):
        logger.error("stage-2 returned %s probabilities, expected %d", probabilities.shape, args.stage2_batch)
        return 1
    logger.info(
        "stage-2: batch %d, mean prob %.3f, mean diameter %.2f mm (%.2fs)",
        args.stage2_batch,
        float(probabilities.mean()),
        float(diameters.mean()),
        stage2_latency,
    )

    logger.info(
        "OK - stage1 p50 %.2fs over %d runs, stage2 %.2fs per batch",
        sorted(stage1_latencies)[len(stage1_latencies) // 2],
        len(stage1_latencies),
        stage2_latency,
    )
    factory.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
