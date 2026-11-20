#!/usr/bin/env python
"""Create a train/val split JSON from a LUNA16 annotations file.

Example::

    python scripts/make_split.py --annotations data/raw/luna16/annotations/annotations.csv \
        --output configs/splits/luna16_fold0.json --val-fraction 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lung_nodule.data.luna16 import read_annotations
from lung_nodule.utils.io import ensure_dir, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0, help="Recorded in the output for traceability")
    parser.add_argument("--prefix", default="luna16")
    args = parser.parse_args()

    series = sorted(read_annotations(args.annotations)["seriesuid"].unique().tolist())
    if len(series) < 2:
        print("need at least two series to build a split", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    shuffled = list(rng.permutation(np.asarray(series, dtype=object)))
    n_val = max(1, round(len(shuffled) * float(args.val_fraction)))
    val = sorted(str(uid) for uid in shuffled[:n_val])
    train = sorted(str(uid) for uid in shuffled[n_val:])

    payload = {
        "description": f"{args.prefix} split generated from {Path(args.annotations).name}",
        "fold": int(args.fold),
        "seed": int(args.seed),
        "val_fraction": float(args.val_fraction),
        "train": train,
        "val": val,
    }
    target = Path(args.output)
    ensure_dir(target.parent)
    write_json(target, payload)
    print(json.dumps({"train": len(train), "val": len(val), "output": str(target)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
