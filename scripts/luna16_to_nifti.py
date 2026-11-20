#!/usr/bin/env python
"""Convert LUNA16 ``.mhd``/``.raw`` scans to ``.nii.gz`` (geometry preserved).

    python scripts/luna16_to_nifti.py --images-dir data/raw/luna16/subset0 \
        --output data/raw/luna16_nifti --limit 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import SimpleITK as sitk

    source = Path(args.images - dir)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)

    scans = sorted(source.glob("*.mhd"))
    if args.limit:
        scans = scans[: args.limit]
    if not scans:
        print(f"no .mhd files found in {source}", file=sys.stderr)
        return 1

    converted = 0
    for path in tqdm(scans, desc="nifti"):
        destination = target / f"{path.stem}.nii.gz"
        if destination.exists() and not args.overwrite:
            continue
        image = sitk.ReadImage(str(path))
        sitk.WriteImage(image, str(destination), useCompression=True)
        converted += 1

    print(f"converted {converted}/{len(scans)} scans -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
