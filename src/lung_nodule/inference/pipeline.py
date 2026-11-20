"""End-to-end two-stage inference pipeline (backend agnostic).

Stage 1 produces probability maps, candidates are extracted in world coordinates,
stage 2 rescores fixed-size crops around them, and the final score is written in
the LUNA16 submission format (``seriesuid, coordX, coordY, coordZ, score``).

The same class drives the local PyTorch runners and the Triton runners, so a
deployment cannot silently diverge from the validated training/inference path.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from lung_nodule.data.luna16 import (
    VolumeGeometry,
    find_series_file,
    load_split,
    load_volume,
    read_predictions,
    write_predictions,
)
from lung_nodule.inference.postprocess import (
    CANDIDATE_COLUMNS,
    CandidateParams,
    distance_nms,
    empty_candidates,
    postprocess_candidates,
)
from lung_nodule.utils.arrays import crop_3d
from lung_nodule.utils.config import load_config
from lung_nodule.utils.io import ensure_dir, git_sha, utc_now_iso, write_json
from lung_nodule.utils.logging import setup_logging

COMBINE_MODES = ("multiply", "stage2_only", "mean")
DIAMETER_SOURCES = ("mask", "stage2", "mean")


@dataclass
class PipelineSettings:
    candidates: CandidateParams
    crop_size: tuple[int, int, int] = (48, 48, 48)
    use_mask_channel: bool = True
    combine_mode: str = "multiply"
    stage2_floor: float = 1.0e-3
    diameter_source: str = "mask"
    min_stage2_score: float = 0.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> PipelineSettings:
        stage2 = cfg.get("stage2", {}) if isinstance(cfg, Mapping) else {}
        combine = cfg.get("combine", {}) if isinstance(cfg, Mapping) else {}
        settings = cls(
            candidates=CandidateParams.from_mapping(cfg.get("candidates")),
            crop_size=tuple(int(v) for v in stage2.get("crop_size", (48, 48, 48))),  # type: ignore[arg-type]
            use_mask_channel=bool(stage2.get("use_mask_channel", True)),
            combine_mode=str(combine.get("mode", "multiply")),
            stage2_floor=float(combine.get("stage2_floor", 1.0e-3)),
            diameter_source=str(combine.get("diameter_source", "mask")),
            min_stage2_score=float(combine.get("min_stage2_score", 0.0)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.combine_mode not in COMBINE_MODES:
            raise ValueError(f"combine.mode must be one of {COMBINE_MODES}, got {self.combine_mode!r}")
        if self.diameter_source not in DIAMETER_SOURCES:
            raise ValueError(f"combine.diameter_source must be one of {DIAMETER_SOURCES}")
        if len(self.crop_size) != 3:
            raise ValueError(f"stage2.crop_size must have 3 entries, got {self.crop_size}")


class NodulePipeline:
    """Two-stage detector driven by a stage-1 runner and an optional stage-2 runner."""

    def __init__(
        self,
        stage1: Any,
        stage2: Any | None,
        settings: PipelineSettings,
        logger: Any | None = None,
    ) -> None:
        self.stage1 = stage1
        self.stage2 = stage2
        self.settings = settings
        self.logger = logger or setup_logging(name="lung_nodule.pipeline")

    # ------------------------------------------------------------------ inference
    def predict_volume(
        self,
        volume: np.ndarray,
        geometry: VolumeGeometry,
        series_uid: str = "",
    ) -> pd.DataFrame:
        probabilities = self.stage1.predict_volume(volume)
        candidates = postprocess_candidates(
            probabilities["mask_prob"],
            probabilities["center_prob"],
            geometry,
            self.settings.candidates,
            series_uid=series_uid,
        )
        if candidates.empty:
            return candidates
        if self.stage2 is None:
            return self._finalize(candidates)

        crops = self.build_crops(volume, probabilities["mask_prob"], candidates)
        probabilities_stage2, diameters_stage2 = self.stage2.score_crops(crops)

        candidates = candidates.copy()
        candidates["score_stage1"] = candidates["score"].to_numpy(dtype=np.float32)
        candidates["score_stage2"] = probabilities_stage2.astype(np.float32)
        candidates["diameter_stage2"] = diameters_stage2.astype(np.float32)
        candidates["score"] = self._combine_scores(candidates).astype(np.float32)
        candidates["diameter_mm"] = self._combine_diameters(candidates).astype(np.float32)

        if self.settings.min_stage2_score > 0:
            candidates = candidates.loc[candidates["score"] >= self.settings.min_stage2_score]
        if candidates.empty:
            return empty_candidates()

        # Re-running NMS on the fused score keeps the best candidate of each cluster.
        candidates = distance_nms(candidates, self.settings.candidates.nms_distance_mm)
        return self._finalize(candidates)

    def predict_scan(self, image_path: str | Path) -> pd.DataFrame:
        volume, geometry = load_volume(image_path)
        return self.predict_volume(volume, geometry, series_uid=Path(image_path).stem)

    def build_crops(
        self,
        volume: np.ndarray,
        mask_prob: np.ndarray,
        candidates: pd.DataFrame,
    ) -> np.ndarray:
        size = self.settings.crop_size
        crops: list[np.ndarray] = []
        for row in candidates.itertuples(index=False):
            center = (float(row.voxel_z), float(row.voxel_y), float(row.voxel_x))
            channels = [crop_3d(volume, center, size)]
            if self.settings.use_mask_channel:
                channels.append(crop_3d(mask_prob, center, size))
            crops.append(np.stack(channels, axis=0))
        return np.asarray(crops, dtype=np.float32)

    # ------------------------------------------------------------------ cohort
    def predict_cohort(
        self,
        series_uids: Sequence[str],
        images_dir: str | Path,
        output_csv: str | Path,
        resume: bool = False,
        flush_every: int = 20,
        timings: list[float] | None = None,
    ) -> pd.DataFrame:
        output_csv = Path(output_csv)
        frames: list[pd.DataFrame] = []
        done: set[str] = set()

        if resume and output_csv.exists():
            existing = read_predictions(output_csv)
            frames.append(existing)
            done = set(existing["seriesuid"].astype(str))
            self.logger.info("resuming: %d scans already predicted", len(done))

        pending = [uid for uid in series_uids if uid not in done]
        started = time.time()
        for index, uid in enumerate(tqdm(pending, desc="stage1+2"), start=1):
            scan_started = time.time()
            try:
                path = find_series_file(images_dir, uid)
                frame = self.predict_scan(path)
                frame["seriesuid"] = uid
                frames.append(frame)
            except Exception:
                self.logger.exception("inference failed for %s", uid)
            if timings is not None:
                timings.append(time.time() - scan_started)
            if index % flush_every == 0:
                write_predictions(self._concat(frames), output_csv)

        predictions = self._concat(frames)
        write_predictions(predictions, output_csv)
        elapsed = time.time() - started
        self.logger.info(
            "predicted %d scans in %.1fs (%.2fs/scan), %d candidates -> %s",
            len(pending),
            elapsed,
            elapsed / max(1, len(pending)),
            len(predictions),
            output_csv,
        )
        return predictions

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        device: str | None = None,
        with_stage2: bool = True,
        logger: Any | None = None,
    ) -> NodulePipeline:
        """Load both checkpoints and build the local PyTorch runners."""
        from lung_nodule.inference.torch_runners import TorchStage1Runner, TorchStage2Runner
        from lung_nodule.models.stage1_detector import load_stage1
        from lung_nodule.models.stage2_classifier import load_stage2
        from lung_nodule.training.engine import resolve_device

        logging = logger or setup_logging(name="lung_nodule.pipeline")
        torch_device = resolve_device(device or cfg.get("device", "cuda"))
        settings = PipelineSettings.from_config(cfg)

        patch_size = tuple(int(v) for v in cfg.get("stage1", {}).get("patch_size", (96, 96, 96)))
        stage1_model = load_stage1(cfg["stage1"]["checkpoint"], device=torch_device, patch_size=patch_size)
        stage1_runner = TorchStage1Runner(
            stage1_model,
            patch_size=patch_size,
            device=torch_device,
            sw_batch_size=int(cfg.get("stage1", {}).get("sw_batch_size", 4)),
            overlap=float(cfg.get("stage1", {}).get("overlap", 0.5)),
            amp=bool(cfg.get("stage1", {}).get("amp", True)),
        )
        logging.info("loaded stage-1 checkpoint: %s", cfg["stage1"]["checkpoint"])

        stage2_runner = None
        if with_stage2:
            stage2_model = load_stage2(
                cfg["stage2"]["checkpoint"], device=torch_device, crop_size=settings.crop_size
            )
            stage2_runner = TorchStage2Runner(
                stage2_model,
                device=torch_device,
                batch_size=int(cfg.get("stage2", {}).get("batch_size", 64)),
                amp=bool(cfg.get("stage2", {}).get("amp", True)),
            )
            logging.info("loaded stage-2 checkpoint: %s", cfg["stage2"]["checkpoint"])

        return cls(stage1_runner, stage2_runner, settings, logger=logging)

    # ------------------------------------------------------------------ helpers
    def _combine_scores(self, candidates: pd.DataFrame) -> np.ndarray:
        stage1_score = candidates["score_stage1"].to_numpy(dtype=np.float64)
        stage2_score = np.clip(candidates["score_stage2"].to_numpy(dtype=np.float64), 0.0, 1.0)
        floor = float(self.settings.stage2_floor)
        if self.settings.combine_mode == "stage2_only":
            return stage2_score
        if self.settings.combine_mode == "mean":
            return 0.5 * (stage1_score + stage2_score)
        return stage1_score * (floor + (1.0 - floor) * stage2_score)

    def _combine_diameters(self, candidates: pd.DataFrame) -> np.ndarray:
        from_mask = candidates["diameter_mm"].to_numpy(dtype=np.float64)
        from_stage2 = candidates["diameter_stage2"].to_numpy(dtype=np.float64)
        params = self.settings.candidates
        source = self.settings.diameter_source
        if source == "stage2":
            values = from_stage2
        elif source == "mean" and (from_stage2 > 0).all():
            values = 0.5 * (from_mask + from_stage2)
        else:
            values = from_mask
        return np.clip(values, params.min_diameter_mm, params.max_diameter_mm)

    def _finalize(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return candidates
        frame = candidates.sort_values("score", ascending=False).reset_index(drop=True)
        for column in CANDIDATE_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame.loc[:, list(CANDIDATE_COLUMNS)]

    @staticmethod
    def _concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
        usable = [frame for frame in frames if frame is not None and not frame.empty]
        if not usable:
            return empty_candidates()
        return pd.concat(usable, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="pipeline.yaml")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--annotations", default=None, help="Optional annotations.csv to restrict the cohort")
    parser.add_argument("--split", default=None, help="Optional split JSON (train/val)")
    parser.add_argument("--split-key", default="val", choices=("train", "val"))
    parser.add_argument("--series-uid", action="append", default=[], help="Explicit series UID (repeatable)")
    parser.add_argument("--output", required=True, help="Predictions CSV")
    parser.add_argument("--device", default=None, help="cuda | cpu (defaults to config/auto)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip scans already present in --output")
    parser.add_argument("--no-stage2", action="store_true", help="Evaluate stage 1 alone")
    parser.add_argument("--override", dest="overrides", action="append", default=[])
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging(args.log_level, name="lung_nodule.pipeline", force=True)
    cfg = load_config(args.config, overrides=args.overrides)

    pipeline = NodulePipeline.from_config(
        cfg,
        device=args.device,
        with_stage2=not args.no_stage2,
        logger=logger,
    )

    if args.series_uid:
        series_uids = list(args.series_uid)
    elif args.split:
        series_uids = load_split(args.split)[args.split_key]
    elif args.annotations:
        from lung_nodule.data.luna16 import read_annotations

        series_uids = sorted(read_annotations(args.annotations)["seriesuid"].unique().tolist())
    else:
        series_uids = sorted({path.stem for path in Path(args.images_dir).glob("*.mhd")})
    if args.limit:
        series_uids = series_uids[: args.limit]
    if not series_uids:
        logger.error("empty cohort: pass --split, --annotations or --series-uid")
        return 1

    started = time.time()
    predictions = pipeline.predict_cohort(
        series_uids,
        args.images_dir,
        args.output,
        resume=args.resume,
    )

    summary = {
        "created_at": utc_now_iso(),
        "git_sha": git_sha(),
        "config": str(args.config),
        "config_fingerprint": cfg.fingerprint(),
        "series": len(series_uids),
        "candidates": len(predictions),
        "stage2": not args.no_stage2,
        "seconds": round(time.time() - started, 2),
        "predictions_csv": str(Path(args.output).resolve()),
    }
    summary_path = Path(args.output).with_name(Path(args.output).stem + ".summary.json")
    write_json(summary_path, summary)
    ensure_dir(summary_path.parent)
    logger.info("wrote %d candidates to %s", len(predictions), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
