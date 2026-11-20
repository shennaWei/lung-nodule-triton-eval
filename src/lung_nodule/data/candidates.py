"""Candidate bookkeeping for the stage-2 false-positive reducer.

Stage-2 training samples come from two sources:

1. the LUNA16 ``candidates.csv`` (pre-computed, balanced-ish candidate list), or
2. hard negatives mined from stage-1 predictions, mixed with the annotations.

Mining keeps the FP reducer focused on the mistakes the detector actually makes,
which is the whole point of a two-stage design.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lung_nodule.data.luna16 import read_annotations
from lung_nodule.utils.io import read_json, utc_now_iso, write_json

HARD_NEGATIVE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CandidateSample:
    """One stage-2 training example in world coordinates."""

    series_uid: str
    coord: tuple[float, float, float]
    label: int
    diameter_mm: float
    score_stage1: float = 0.0


def attach_diameters(candidates: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Give each candidate the diameter of its nearest annotation (6 mm fallback)."""
    frame = candidates.copy()
    diameters = np.full(len(frame), 6.0, dtype=np.float32)

    # ``indices`` yields positional indices, which is what the array assignment needs
    # (the frame index may have gaps after filtering).
    for uid, positions in frame.groupby("seriesuid", sort=False).indices.items():
        points = frame.iloc[positions][["coordX", "coordY", "coordZ"]].to_numpy(dtype=np.float64)
        target = annotations.loc[annotations["seriesuid"] == uid]
        if target.empty:
            continue
        target_points = target[["coordX", "coordY", "coordZ"]].to_numpy(dtype=np.float64)
        target_diameters = target["diameter_mm"].to_numpy(dtype=np.float32)
        distances = np.linalg.norm(points[:, None, :] - target_points[None, :, :], axis=2)
        nearest = distances.argmin(axis=1)
        diameters[np.asarray(positions)] = target_diameters[nearest]

    frame["diameter_mm"] = diameters
    return frame


def samples_from_candidates(
    candidates: pd.DataFrame,
    annotations: pd.DataFrame,
    series_uids: list[str] | None = None,
    max_positives: int | None = None,
    max_negatives: int | None = None,
    seed: int = 42,
) -> list[CandidateSample]:
    """Convert ``candidates.csv`` rows into :class:`CandidateSample` objects."""
    frame = candidates
    if series_uids is not None:
        frame = frame.loc[frame["seriesuid"].isin(series_uids)]
    if frame.empty:
        return []

    frame = attach_diameters(frame, annotations)
    positives = frame.loc[frame["class"] == 1]
    negatives = frame.loc[frame["class"] != 1]
    rng = np.random.default_rng(seed)
    if max_positives is not None and len(positives) > max_positives:
        positives = positives.iloc[rng.choice(len(positives), size=max_positives, replace=False)]
    if max_negatives is not None and len(negatives) > max_negatives:
        negatives = negatives.iloc[rng.choice(len(negatives), size=max_negatives, replace=False)]

    return [_record_to_sample(record) for record in pd.concat([positives, negatives]).to_dict("records")]


def samples_from_annotations(
    annotations: pd.DataFrame, series_uids: list[str] | None = None
) -> list[CandidateSample]:
    """Every annotation becomes a positive candidate (used together with mined negatives)."""
    frame = annotations
    if series_uids is not None:
        frame = frame.loc[frame["seriesuid"].isin(series_uids)]
    return [
        CandidateSample(
            series_uid=str(row.seriesuid),
            coord=(float(row.coordX), float(row.coordY), float(row.coordZ)),
            label=1,
            diameter_mm=float(row.diameter_mm),
        )
        for row in frame.itertuples(index=False)
    ]


def mine_hard_negatives(
    predictions: pd.DataFrame,
    annotations: pd.DataFrame,
    per_scan: int = 20,
    min_score: float = 0.05,
) -> dict[str, Any]:
    """Keep the highest-scoring false positives of each scan for stage-2 training.

    Matching reuses the FROC criterion so that a mined negative is exactly a
    prediction that would have hurt sensitivity at the evaluation operating point.
    """
    from lung_nodule.eval.froc import match_predictions

    if predictions.empty:
        return {
            "schema_version": HARD_NEGATIVE_SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "per_scan": per_scan,
            "n_candidates": 0,
            "scans": {},
        }

    matched = match_predictions(predictions, annotations)
    false_positives = matched.loc[(~matched["is_tp"]) & (matched["score"] >= min_score)]

    scans: dict[str, list[dict[str, float]]] = {}
    for uid, group in false_positives.groupby("seriesuid", sort=False):
        top = group.sort_values("score", ascending=False).head(per_scan)
        scans[str(uid)] = [
            {
                "coordX": float(row.coordX),
                "coordY": float(row.coordY),
                "coordZ": float(row.coordZ),
                "score": float(row.score),
                "diameter_mm": float(getattr(row, "diameter_mm", 6.0)),
            }
            for row in top.itertuples(index=False)
        ]

    return {
        "schema_version": HARD_NEGATIVE_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "per_scan": per_scan,
        "min_score": min_score,
        "n_candidates": int(sum(len(v) for v in scans.values())),
        "scans": scans,
    }


def save_hard_negatives(payload: dict[str, Any], path: str | Path) -> Path:
    return write_json(path, payload)


def load_hard_negatives(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    if int(payload.get("schema_version", 0)) != HARD_NEGATIVE_SCHEMA_VERSION:
        raise ValueError(f"unsupported hard-negative schema in {path}: {payload.get('schema_version')}")
    return payload


def samples_from_hard_negatives(
    payload: dict[str, Any],
    annotations: pd.DataFrame,
    series_uids: list[str] | None = None,
) -> list[CandidateSample]:
    """Positives from the annotation table, negatives from the mined JSON."""
    allowed = set(series_uids) if series_uids is not None else None
    samples = samples_from_annotations(annotations, series_uids)

    for uid, entries in payload.get("scans", {}).items():
        if allowed is not None and uid not in allowed:
            continue
        for entry in entries:
            samples.append(
                CandidateSample(
                    series_uid=str(uid),
                    coord=(float(entry["coordX"]), float(entry["coordY"]), float(entry["coordZ"])),
                    label=0,
                    diameter_mm=float(entry.get("diameter_mm", 6.0)),
                    score_stage1=float(entry.get("score", 0.0)),
                )
            )
    return samples


def balance_samples(
    samples: list[CandidateSample],
    neg_pos_ratio: float = 2.0,
    seed: int = 42,
) -> list[CandidateSample]:
    """Down-sample negatives so the FP reducer does not collapse to "all negative"."""
    positives = [s for s in samples if s.label == 1]
    negatives = [s for s in samples if s.label != 1]
    if not positives or not negatives:
        return samples
    keep = int(min(len(negatives), max(1, round(len(positives) * neg_pos_ratio))))
    rng = np.random.default_rng(seed)
    if keep < len(negatives):
        index = rng.choice(len(negatives), size=keep, replace=False)
        negatives = [negatives[i] for i in sorted(index)]
    return positives + negatives


def load_training_samples(
    source: str,
    candidates_csv: str | Path | None = None,
    hard_negatives_json: str | Path | None = None,
    annotations_csv: str | Path | None = None,
    series_uids: list[str] | None = None,
    neg_pos_ratio: float = 2.0,
    seed: int = 42,
) -> list[CandidateSample]:
    """Build the stage-2 sample list from either candidates.csv or mined negatives."""
    annotations = read_annotations(annotations_csv) if annotations_csv else pd.DataFrame()

    if source == "candidates":
        if candidates_csv is None:
            raise ValueError("source='candidates' requires candidates_csv")
        from lung_nodule.data.luna16 import read_candidates

        samples = samples_from_candidates(
            read_candidates(candidates_csv),
            annotations,
            series_uids=series_uids,
            seed=seed,
        )
    elif source == "hard_negatives":
        if hard_negatives_json is None:
            raise ValueError("source='hard_negatives' requires hard_negatives_json")
        samples = samples_from_hard_negatives(
            load_hard_negatives(hard_negatives_json), annotations, series_uids
        )
    else:
        raise ValueError(f"unknown sample source: {source!r}")

    return balance_samples(samples, neg_pos_ratio=neg_pos_ratio, seed=seed)


def _record_to_sample(record: dict[str, Any]) -> CandidateSample:
    return CandidateSample(
        series_uid=str(record["seriesuid"]),
        coord=(float(record["coordX"]), float(record["coordY"]), float(record["coordZ"])),
        label=int(record.get("class", 0) != 0),
        diameter_mm=float(record.get("diameter_mm", 6.0)),
    )


__all__ = [
    "CandidateSample",
    "attach_diameters",
    "balance_samples",
    "load_hard_negatives",
    "load_training_samples",
    "mine_hard_negatives",
    "samples_from_annotations",
    "samples_from_candidates",
    "samples_from_hard_negatives",
    "save_hard_negatives",
]
