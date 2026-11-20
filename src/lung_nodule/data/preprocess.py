"""LUNA16 -> 1 mm cached arrays preprocessing.

Run once per fold::

    python -m lung_nodule.data.preprocess \
        --images-dir data/raw/luna16/subset0 \
        --annotations data/raw/luna16/annotations/annotations.csv \
        --output data/preprocessed/luna16_1mm

Each scan becomes ``<series_uid>.npz`` (image / mask / heatmap / geometry). The
resampled images are what both training stages consume, so voxel spacing is
identical everywhere and physical distances never need to be recomputed.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lung_nodule.data.luna16 import (
    VolumeGeometry,
    annotations_for,
    find_series_file,
    rasterize_annotations,
    read_annotations,
)
from lung_nodule.utils.arrays import normalize_hu
from lung_nodule.utils.io import ensure_dir, utc_now_iso, write_json
from lung_nodule.utils.logging import setup_logging


def preprocess_scan(
    series_uid: str,
    images_dir: str | Path,
    annotations: pd.DataFrame,
    output_dir: str | Path,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    hu_window: tuple[float, float] = (-1000.0, 400.0),
    heatmap_sigma_scale: float = 0.35,
    overwrite: bool = False,
) -> dict:
    """Resample one scan, rasterise its nodule targets and cache the arrays."""
    import SimpleITK as sitk

    output_dir = Path(output_dir)
    target = output_dir / f"{series_uid}.npz"
    if target.exists() and not overwrite:
        return {"seriesuid": series_uid, "status": "skipped", "path": str(target)}

    image = sitk.ReadImage(str(find_series_file(images_dir, series_uid)))
    resampled = resample_image(image, spacing)
    volume = sitk.GetArrayFromImage(resampled).astype(np.float32)
    geometry = VolumeGeometry.from_sitk(resampled)
    image_norm = normalize_hu(volume, hu_min=hu_window[0], hu_max=hu_window[1])

    scan_annotations = annotations_for(annotations, series_uid)
    mask, heatmap = rasterize_annotations(
        shape=tuple(int(s) for s in image_norm.shape),  # type: ignore[arg-type]
        geometry=geometry,
        annotations=scan_annotations,
        heatmap_sigma_scale=heatmap_sigma_scale,
    )

    np.savez_compressed(
        target,
        image=image_norm.astype(np.float16),
        mask=mask.astype(np.uint8),
        heatmap=heatmap.astype(np.float16),
        meta=np.asarray(json.dumps(geometry.to_dict())),
    )
    return {
        "seriesuid": series_uid,
        "status": "ok",
        "path": str(target),
        "shape": list(geometry.shape),
        "spacing": list(geometry.spacing),
        "n_nodules": len(scan_annotations),
        "mask_voxels": int(mask.sum()),
    }


def resample_image(image, spacing: tuple[float, float, float]):
    """Linear resample to an isotropic grid, preserving the world extent."""
    import SimpleITK as sitk

    original_spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    original_size = np.asarray(image.GetSize(), dtype=np.float64)
    target_size = np.round(original_size * original_spacing / np.asarray(spacing, dtype=np.float64)).astype(
        int
    )
    target_size = np.maximum(target_size, 1)
    return sitk.Resample(
        image,
        [int(s) for s in target_size],
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        [float(s) for s in spacing],
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def _worker(payload: dict) -> dict:
    return preprocess_scan(**payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images-dir", required=True, help="Directory containing .mhd/.nii.gz scans")
    parser.add_argument("--annotations", required=True, help="LUNA16 annotations.csv")
    parser.add_argument("--output", required=True, help="Output directory for the npz cache")
    parser.add_argument("--spacing", default="1.0,1.0,1.0", help="Target isotropic spacing in mm")
    parser.add_argument("--hu-window", default="-1000,400", help="HU clipping window, e.g. -1000,400")
    parser.add_argument("--heatmap-sigma-scale", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N scans")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging(args.log_level)
    overall = setup_logging(args.log_level, name="lung_nodule.preprocess", force=True)

    annotations = read_annotations(args.annotations)
    series_uids = sorted(annotations["seriesuid"].unique().tolist())
    if args.limit:
        series_uids = series_uids[: args.limit]
    if not series_uids:
        logger.error("no series found in %s", args.annotations)
        return 1

    spacing = tuple(float(v) for v in args.spacing.split(","))
    hu_window = tuple(float(v) for v in args.hu_window.split(","))
    output_dir = ensure_dir(args.output)

    payloads = [
        {
            "series_uid": uid,
            "images_dir": args.images_dir,
            "annotations": annotations,
            "output_dir": output_dir,
            "spacing": spacing,
            "hu_window": hu_window,
            "heatmap_sigma_scale": args.heatmap_sigma_scale,
            "overwrite": args.overwrite,
        }
        for uid in series_uids
    ]

    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_worker, payload): payload["series_uid"] for payload in payloads}
        for future in tqdm(as_completed(futures), total=len(futures), desc="preprocess"):
            uid = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                overall.exception("failed to preprocess %s: %s", uid, exc)
                records.append({"seriesuid": uid, "status": "failed", "error": str(exc)})

    manifest = {
        "created_at": utc_now_iso(),
        "source": str(Path(args.annotations).resolve()),
        "images_dir": str(Path(args.images_dir).resolve()),
        "spacing": list(spacing),
        "hu_window": list(hu_window),
        "n_series": len(records),
        "n_ok": sum(record["status"] in {"ok", "skipped"} for record in records),
        "n_failed": sum(record["status"] == "failed" for record in records),
        "records": sorted(records, key=lambda record: record["seriesuid"]),
    }
    write_json(Path(output_dir) / "manifest.json", manifest)
    logger.info(
        "preprocessed %d/%d scans into %s",
        manifest["n_ok"],
        manifest["n_series"],
        output_dir,
    )
    return 0 if manifest["n_failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
