"""FROC / CPM evaluation following the LUNA16 protocol.

The competition criterion is one prediction per nodule at most, matched when the
predicted centre falls inside the annotated nodule sphere (radius =
``diameter_mm / 2``). Sensitivity is the fraction of annotations matched;
false positives are predictions that matched nothing, reported per scan.

All thresholds are swept in a single pass: predictions are sorted by descending
score and matched greedily, which yields the same matches as re-running the
matcher for every threshold because a higher-scoring prediction always claims an
annotation first.

``CPM`` (Competition Performance Metric) is the mean sensitivity at the seven
LUNA16 operating points ``{1/8, 1/4, 1/2, 1, 2, 4, 8}`` false positives/scan.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lung_nodule.data.luna16 import read_annotations, read_predictions
from lung_nodule.utils.io import ensure_dir, write_json
from lung_nodule.utils.logging import setup_logging

FP_LEVELS: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
_ORIGIN_THRESHOLD = 1e12  # stands in for "threshold above every prediction"


@dataclass
class FrocResult:
    thresholds: np.ndarray
    fps_per_scan: np.ndarray
    sensitivity: np.ndarray
    cpm: float
    sensitivity_at: dict[str, float]
    operating_points: dict[str, dict[str, float]]
    fp_at_sensitivity: float | None
    n_scans: int
    n_annotations: int
    n_predictions: int
    n_false_positives: int
    matching: dict[str, float] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        metrics: dict[str, float] = {"cpm": float(self.cpm)}
        metrics.update({f"sens@{key}fp": float(value) for key, value in self.sensitivity_at.items()})
        if self.fp_at_sensitivity is not None:
            metrics["fp_at_sens0.90"] = float(self.fp_at_sensitivity)

        curve = {
            "threshold": _downsample(self.thresholds),
            "fp_per_scan": _downsample(self.fps_per_scan),
            "sensitivity": _downsample(self.sensitivity),
        }
        return {
            "schema_version": 1,
            "metrics": metrics,
            "counts": {
                "n_scans": int(self.n_scans),
                "n_annotations": int(self.n_annotations),
                "n_predictions": int(self.n_predictions),
                "n_false_positives": int(self.n_false_positives),
            },
            "operating_points": self.operating_points,
            "matching": dict(self.matching),
            "curve": curve,
        }


def match_predictions(
    predictions: pd.DataFrame,
    annotations: pd.DataFrame,
    radius_scale: float = 0.5,
    min_radius_mm: float = 0.0,
) -> pd.DataFrame:
    """Add an ``is_tp`` column to ``predictions`` using the sphere criterion."""
    frame = predictions.copy()
    frame["is_tp"] = False
    if frame.empty:
        return frame

    annotation_groups = {str(uid): group for uid, group in annotations.groupby("seriesuid", sort=False)}
    for uid, group in frame.groupby("seriesuid", sort=False):
        target = annotation_groups.get(str(uid))
        if target is None or target.empty:
            continue

        order = np.argsort(-group["score"].to_numpy(dtype=np.float64), kind="mergesort")
        points = group[["coordX", "coordY", "coordZ"]].to_numpy(dtype=np.float64)[order]
        target_points = target[["coordX", "coordY", "coordZ"]].to_numpy(dtype=np.float64)
        radii = np.maximum(
            target["diameter_mm"].to_numpy(dtype=np.float64) * float(radius_scale),
            float(min_radius_mm),
        )

        used = np.zeros(len(target), dtype=bool)
        hits = np.zeros(len(order), dtype=bool)
        for position in range(len(order)):
            distances = np.linalg.norm(target_points - points[position], axis=1)
            distances[used] = np.inf
            nearest = int(np.argmin(distances))
            if distances[nearest] <= radii[nearest]:
                used[nearest] = True
                hits[position] = True

        frame.loc[group.index[order][hits], "is_tp"] = True

    return frame


def compute_froc(
    predictions: pd.DataFrame,
    annotations: pd.DataFrame,
    fp_levels: tuple[float, ...] = FP_LEVELS,
    radius_scale: float = 0.5,
    min_radius_mm: float = 0.0,
    n_scans: int | None = None,
    sensitivity_target: float = 0.90,
) -> FrocResult:
    if annotations.empty:
        raise ValueError("cannot compute FROC without annotations")
    if "score" not in predictions.columns:
        raise ValueError("predictions need a 'score' column")

    evaluated_scans = int(n_scans or annotations["seriesuid"].nunique())
    if evaluated_scans <= 0:
        raise ValueError("n_scans must be positive")
    n_annotations = len(annotations)

    matched = match_predictions(predictions, annotations, radius_scale, min_radius_mm)
    n_predictions = len(matched)
    n_false_positives = int((~matched["is_tp"]).sum()) if n_predictions else 0

    if n_predictions == 0:
        thresholds = np.asarray([_ORIGIN_THRESHOLD], dtype=np.float64)
        fps = np.zeros(1, dtype=np.float64)
        sensitivity = np.zeros(1, dtype=np.float64)
    else:
        ordered = matched.sort_values("score", ascending=False, kind="mergesort")
        grouped = ordered.groupby("score", sort=False)
        true_positives = grouped["is_tp"].sum().cumsum().to_numpy(dtype=np.float64)
        false_positives = (grouped.size() - grouped["is_tp"].sum()).cumsum().to_numpy(dtype=np.float64)
        thresholds = np.concatenate([[_ORIGIN_THRESHOLD], grouped["score"].max().to_numpy(dtype=np.float64)])
        fps = np.concatenate([[0.0], false_positives / evaluated_scans])
        sensitivity = np.concatenate([[0.0], true_positives / max(1, n_annotations)])

    sensitivity = np.maximum.accumulate(np.clip(sensitivity, 0.0, 1.0))
    fps = np.maximum.accumulate(np.clip(fps, 0.0, None))

    sensitivities_at: dict[str, float] = {}
    operating_points: dict[str, dict[str, float]] = {}
    for level in fp_levels:
        key = _format_fp(level)
        value = step_sensitivity(fps, sensitivity, level)
        sensitivities_at[key] = value
        operating_points[key] = operating_point(fps, sensitivity, thresholds, level)

    cpm = float(np.mean([sensitivities_at[_format_fp(level)] for level in fp_levels]))
    fp_hit = fp_at_sensitivity(fps, sensitivity, target=sensitivity_target)

    return FrocResult(
        thresholds=thresholds,
        fps_per_scan=fps,
        sensitivity=sensitivity,
        cpm=cpm,
        sensitivity_at=sensitivities_at,
        operating_points=operating_points,
        fp_at_sensitivity=fp_hit,
        n_scans=evaluated_scans,
        n_annotations=n_annotations,
        n_predictions=n_predictions,
        n_false_positives=n_false_positives,
        matching={"radius_scale": float(radius_scale), "min_radius_mm": float(min_radius_mm)},
    )


def step_sensitivity(fps_per_scan: np.ndarray, sensitivity: np.ndarray, target_fp: float) -> float:
    """Best sensitivity reachable without exceeding ``target_fp`` false positives/scan."""
    mask = fps_per_scan <= float(target_fp) + 1e-12
    if not mask.any():
        return 0.0
    return float(sensitivity[mask].max())


def fp_at_sensitivity(fps_per_scan: np.ndarray, sensitivity: np.ndarray, target: float = 0.9) -> float | None:
    """False positives/scan needed to reach ``target`` sensitivity (``None`` if unreachable)."""
    mask = sensitivity >= float(target) - 1e-12
    if not mask.any():
        return None
    return float(fps_per_scan[mask].min())


def operating_point(
    fps_per_scan: np.ndarray,
    sensitivity: np.ndarray,
    thresholds: np.ndarray,
    target_fp: float,
) -> dict[str, float]:
    mask = fps_per_scan <= float(target_fp) + 1e-12
    if not mask.any():
        return {"threshold": float(_ORIGIN_THRESHOLD), "fp_per_scan": 0.0, "sensitivity": 0.0}
    index = int(np.flatnonzero(mask)[-1])
    threshold = thresholds[index]
    return {
        "threshold": float("inf") if threshold >= _ORIGIN_THRESHOLD else float(threshold),
        "fp_per_scan": float(fps_per_scan[index]),
        "sensitivity": float(sensitivity[index]),
    }


def bootstrap_cpm(
    predictions: pd.DataFrame,
    annotations: pd.DataFrame,
    n_boot: int = 200,
    seed: int = 0,
    fp_levels: tuple[float, ...] = FP_LEVELS,
    radius_scale: float = 0.5,
    min_radius_mm: float = 0.0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Scan-level bootstrap confidence interval for the CPM (cheap fold-level uncertainty)."""
    scans = sorted(set(annotations["seriesuid"].astype(str)) | set(predictions["seriesuid"].astype(str)))
    if not scans:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}

    prediction_groups = {str(uid): group for uid, group in predictions.groupby("seriesuid", sort=False)}
    annotation_groups = {str(uid): group for uid, group in annotations.groupby("seriesuid", sort=False)}
    rng = np.random.default_rng(seed)

    values: list[float] = []
    for _ in range(int(n_boot)):
        sample = rng.choice(np.asarray(scans, dtype=object), size=len(scans), replace=True)
        prediction_parts: list[pd.DataFrame] = []
        annotation_parts: list[pd.DataFrame] = []
        for draw, uid in enumerate(sample):
            alias = f"boot{draw}"
            if uid in prediction_groups:
                prediction_parts.append(prediction_groups[uid].assign(seriesuid=alias))
            if uid in annotation_groups:
                annotation_parts.append(annotation_groups[uid].assign(seriesuid=alias))
        if not annotation_parts:
            continue
        resampled_predictions = (
            pd.concat(prediction_parts, ignore_index=True) if prediction_parts else predictions.iloc[0:0]
        )
        result = compute_froc(
            resampled_predictions,
            pd.concat(annotation_parts, ignore_index=True),
            fp_levels=fp_levels,
            radius_scale=radius_scale,
            min_radius_mm=min_radius_mm,
            n_scans=len(sample),
        )
        values.append(result.cpm)

    if not values:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "lo": float(np.quantile(array, alpha / 2)),
        "hi": float(np.quantile(array, 1 - alpha / 2)),
        "n_boot": int(array.size),
    }


def render_markdown(result: FrocResult, title: str = "FROC report") -> str:
    lines = [
        f"# {title}",
        "",
        f"- scans: {result.n_scans}",
        f"- annotations: {result.n_annotations}",
        f"- predictions: {result.n_predictions} (false positives: {result.n_false_positives})",
        f"- matching: centre-in-sphere, radius = {result.matching.get('radius_scale', 0.5)} x diameter",
        "",
        "| operating point | sensitivity | threshold |",
        "| --- | --- | --- |",
    ]
    for key, point in result.operating_points.items():
        threshold = point["threshold"]
        lines.append(
            f"| {key} FP/scan | {point['sensitivity']:.4f} | {'inf' if threshold == float('inf') else f'{threshold:.4f}'} |"
        )
    lines.extend(["", f"**CPM: {result.cpm:.4f}**", ""])
    if result.fp_at_sensitivity is not None:
        lines.append(f"False positives/scan at 0.90 sensitivity: {result.fp_at_sensitivity:.4f}")
    return "\n".join(lines) + "\n"


def save_froc_report(
    result: FrocResult,
    output_dir: str | Path,
    name: str = "report",
    extra: dict[str, Any] | None = None,
    plot: bool = False,
) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    report = result.to_report()
    if extra:
        report["metadata"] = extra
    paths = {"report": write_json(output_dir / f"{name}.json", report)}

    markdown = render_markdown(result)
    markdown_path = output_dir / f"{name}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    paths["markdown"] = markdown_path

    curve_path = output_dir / f"{name}_curve.csv"
    pd.DataFrame(
        {
            "threshold": result.thresholds,
            "fp_per_scan": result.fps_per_scan,
            "sensitivity": result.sensitivity,
        }
    ).to_csv(curve_path, index=False)
    paths["curve"] = curve_path

    if plot:
        plot_path = plot_froc(result, output_dir / f"{name}.png")
        if plot_path is not None:
            paths["plot"] = plot_path
    return paths


def plot_froc(result: FrocResult, path: str | Path):
    """Log-scale FROC curve; returns ``None`` when matplotlib is not installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        return None

    figure, axes = plt.subplots(figsize=(6.0, 4.5))
    fps = np.where(result.fps_per_scan <= 0, 1e-3, result.fps_per_scan)
    axes.plot(fps, result.sensitivity, marker="o", linewidth=1.5)
    for level in FP_LEVELS:
        axes.axvline(level, color="0.85", linewidth=0.7, zorder=0)
    axes.set_xscale("log", base=2)
    axes.set_xlabel("false positives per scan")
    axes.set_ylabel("sensitivity")
    axes.set_title(f"FROC (CPM = {result.cpm:.4f})")
    axes.set_ylim(0.0, 1.02)
    axes.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return Path(path)


def _format_fp(level: float) -> str:
    """``0.125 -> '0.125'``, ``1.0 -> '1'`` (used in metric keys)."""
    return f"{float(level):g}"


def _downsample(values: np.ndarray, max_points: int = 512) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size > max_points:
        index = np.linspace(0, values.size - 1, max_points).astype(int)
        values = values[index]
    return [float(value) for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", required=True, help="Predictions CSV (seriesuid, coordX..Z, score)")
    parser.add_argument("--annotations", required=True, help="LUNA16 annotations.csv")
    parser.add_argument("--output-dir", default="artifacts/froc")
    parser.add_argument("--name", default="report")
    parser.add_argument("--radius-scale", type=float, default=0.5)
    parser.add_argument("--min-radius-mm", type=float, default=0.0)
    parser.add_argument("--n-scans", type=int, default=None, help="Override the denominator of FP/scan")
    parser.add_argument("--bootstrap", action="store_true", help="Add a bootstrap CPM confidence interval")
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging(args.log_level, name="lung_nodule.froc", force=True)

    predictions = read_predictions(args.predictions)
    annotations = read_annotations(args.annotations)
    logger.info(
        "loaded %d predictions for %d scans and %d annotations",
        len(predictions),
        predictions["seriesuid"].nunique(),
        len(annotations),
    )

    result = compute_froc(
        predictions,
        annotations,
        radius_scale=args.radius_scale,
        min_radius_mm=args.min_radius_mm,
        n_scans=args.n_scans,
    )

    extra: dict[str, Any] = {
        "predictions": str(Path(args.predictions).resolve()),
        "annotations": str(Path(args.annotations).resolve()),
    }
    if args.bootstrap:
        interval = bootstrap_cpm(
            predictions,
            annotations,
            n_boot=args.n_boot,
            seed=args.seed,
            radius_scale=args.radius_scale,
            min_radius_mm=args.min_radius_mm,
        )
        extra["cpm_bootstrap"] = interval
        logger.info(
            "CPM 95%% CI: %.4f [%.4f, %.4f] over %d resamples",
            interval["mean"],
            interval["lo"],
            interval["hi"],
            interval["n_boot"],
        )

    paths = save_froc_report(result, args.output_dir, name=args.name, extra=extra, plot=args.plot)
    for key, path in paths.items():
        logger.info("%s -> %s", key, path)
    logger.info("CPM = %.4f (%d scans)", result.cpm, result.n_scans)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "FP_LEVELS",
    "FrocResult",
    "bootstrap_cpm",
    "compute_froc",
    "fp_at_sensitivity",
    "match_predictions",
    "operating_point",
    "plot_froc",
    "render_markdown",
    "save_froc_report",
    "step_sensitivity",
]
