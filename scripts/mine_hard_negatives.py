#!/usr/bin/env python
"""Mine stage-1 false positives for stage-2 training.

Takes a predictions CSV produced on the *training* split (scoring your own
training data on purpose) and keeps the top-scoring false positives per scan.

    python scripts/mine_hard_negatives.py \
        --predictions artifacts/predictions/train.csv \
        --annotations data/raw/luna16/annotations/annotations.csv \
        --per-scan 20 --min-score 0.05 \
        --output artifacts/stage1/hard_negatives.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lung_nodule.data.candidates import mine_hard_negatives, save_hard_negatives
from lung_nodule.data.luna16 import load_split, read_annotations, read_predictions
from lung_nodule.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-scan", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--split", default=None, help="Restrict to the scans of one split key")
    parser.add_argument("--split-key", default="train", choices=("train", "val"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = setup_logging(args.log_level, name="mine_hard_negatives", force=True)
    predictions = read_predictions(args.predictions)
    annotations = read_annotations(args.annotations)

    if args.split:
        allowed = set(load_split(args.split)[args.split_key])
        predictions = predictions.loc[predictions["seriesuid"].isin(allowed)]
        annotations = annotations.loc[annotations["seriesuid"].isin(allowed)]

    payload = mine_hard_negatives(predictions, annotations, per_scan=args.per_scan, min_score=args.min_score)
    save_hard_negatives(payload, args.output)
    logger.info(
        "mined %d hard negatives across %d scans -> %s",
        payload["n_candidates"],
        len(payload["scans"]),
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
