"""NumPy-only metrics used outside the FROC computation."""

from __future__ import annotations

import numpy as np


def binary_auroc(scores: np.ndarray | list[float], labels: np.ndarray | list[float]) -> float:
    """Rank-based AUROC with average ranks for ties (Mann-Whitney U)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel() >= 0.5
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    average_rank = cumulative - (counts - 1) / 2.0
    ranks = average_rank[inverse]

    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion_at_threshold(
    scores: np.ndarray | list[float],
    labels: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, int]:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel() >= 0.5
    predicted = scores >= float(threshold)
    return {
        "tp": int(np.sum(predicted & labels)),
        "fp": int(np.sum(predicted & ~labels)),
        "tn": int(np.sum(~predicted & ~labels)),
        "fn": int(np.sum(~predicted & labels)),
    }


def sensitivity_specificity(
    scores: np.ndarray | list[float],
    labels: np.ndarray | list[float],
    threshold: float = 0.5,
) -> tuple[float, float]:
    counts = confusion_at_threshold(scores, labels, threshold)
    sensitivity = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    specificity = counts["tn"] / max(1, counts["tn"] + counts["fp"])
    return float(sensitivity), float(specificity)


__all__ = ["binary_auroc", "confusion_at_threshold", "sensitivity_specificity"]
