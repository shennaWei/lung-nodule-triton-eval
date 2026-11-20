"""Run history and leaderboard for FROC tracking.

Every evaluation appends one JSON line to ``artifacts/froc/history.jsonl``:

    python -m lung_nodule.eval.tracking --append --report artifacts/froc/report.json
    python -m lung_nodule.eval.tracking --history artifacts/froc/history.jsonl \
        --output artifacts/froc/LEADERBOARD.md
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lung_nodule.utils.io import (
    append_jsonl,
    git_branch,
    git_dirty,
    git_sha,
    read_json,
    read_jsonl,
    stable_hash,
    utc_now_iso,
    write_json,
)
from lung_nodule.utils.logging import setup_logging

HISTORY_SCHEMA_VERSION = 1


@dataclass
class RunRecord:
    run_id: str
    created_at: str
    git_sha: str
    git_branch: str
    git_dirty: bool
    backend: str
    dataset: str
    metrics: dict[str, float]
    config_fingerprint: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = HISTORY_SCHEMA_VERSION
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunRecord:
        return cls(
            run_id=str(payload.get("run_id", "unknown")),
            created_at=str(payload.get("created_at", "")),
            git_sha=str(payload.get("git_sha", "unknown")),
            git_branch=str(payload.get("git_branch", "unknown")),
            git_dirty=bool(payload.get("git_dirty", False)),
            backend=str(payload.get("backend", "unknown")),
            dataset=str(payload.get("dataset", "unknown")),
            metrics={str(k): float(v) for k, v in dict(payload.get("metrics", {})).items()},
            config_fingerprint=payload.get("config_fingerprint"),
            tags={str(k): str(v) for k, v in dict(payload.get("tags", {})).items()},
            artifacts={str(k): str(v) for k, v in dict(payload.get("artifacts", {})).items()},
        )


def make_run_record(
    metrics: Mapping[str, float],
    dataset: str,
    backend: str = "triton",
    config_fingerprint: str | None = None,
    tags: Mapping[str, str] | None = None,
    artifacts: Mapping[str, str] | None = None,
    run_id: str | None = None,
) -> RunRecord:
    id_source = {
        "git_sha": git_sha(),
        "dataset": dataset,
        "backend": backend,
        "config": config_fingerprint,
        "metrics": {str(k): round(float(v), 6) for k, v in metrics.items()},
    }
    return RunRecord(
        run_id=run_id or stable_hash(id_source, length=12),
        created_at=utc_now_iso(),
        git_sha=git_sha(),
        git_branch=git_branch(),
        git_dirty=git_dirty(),
        backend=backend,
        dataset=dataset,
        metrics={str(k): float(v) for k, v in metrics.items()},
        config_fingerprint=config_fingerprint,
        tags={str(k): str(v) for k, v in dict(tags or {}).items()},
        artifacts={str(k): str(v) for k, v in dict(artifacts or {}).items()},
    )


def append_run(record: RunRecord, history_path: str | Path) -> Path:
    return append_jsonl(history_path, record.to_dict())


def load_history(path: str | Path) -> list[RunRecord]:
    return [RunRecord.from_dict(payload) for payload in read_jsonl(path)]


def render_leaderboard(
    records: Sequence[RunRecord],
    metric: str = "cpm",
    top_k: int = 10,
    extra_metrics: Sequence[str] = ("sens@1fp", "sens@4fp", "fp_at_sens0.90"),
) -> str:
    ranked = sorted(
        (record for record in records if metric in record.metrics),
        key=lambda record: record.metrics[metric],
        reverse=True,
    )[: max(1, int(top_k))]

    header = ["rank", "run", "commit", "backend", metric, *extra_metrics, "config", "created"]
    lines = [
        f"# FROC leaderboard ({metric})",
        "",
        f"Runs tracked: {len(records)}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for rank, record in enumerate(ranked, start=1):
        row = [
            str(rank),
            f"`{record.run_id}`",
            f"`{record.git_sha[:8]}`" + ("*" if record.git_dirty else ""),
            record.backend,
            f"{record.metrics[metric]:.4f}",
        ]
        for name in extra_metrics:
            row.append(f"{record.metrics[name]:.4f}" if name in record.metrics else "-")
        row.extend([record.config_fingerprint or "-", record.created_at])
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("*Denotes a dirty working tree at evaluation time.*")
    return "\n".join(lines) + "\n"


def measure_trend(records: Sequence[RunRecord], metric: str = "cpm", window: int = 5) -> dict[str, Any]:
    """Compare the last ``window`` runs against the preceding window."""
    values = [record.metrics[metric] for record in records if metric in record.metrics]
    if len(values) < 2:
        return {"metric": metric, "n": len(values), "status": "insufficient_data"}

    recent = np.asarray(values[-window:], dtype=np.float64)
    previous = np.asarray(values[-2 * window : -window], dtype=np.float64)
    delta = float(recent.mean() - previous.mean()) if previous.size else 0.0
    slope = float(np.polyfit(np.arange(len(values)), np.asarray(values), 1)[0])
    threshold = 0.002  # CPM points; below this the series is treated as flat
    status = "improving" if delta > threshold else ("degrading" if delta < -threshold else "stable")
    return {
        "metric": metric,
        "n": len(values),
        "recent_mean": float(recent.mean()),
        "previous_mean": float(previous.mean()) if previous.size else None,
        "delta": delta,
        "slope_per_run": slope,
        "status": status,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--history", default="artifacts/froc/history.jsonl")
    parser.add_argument("--append", action="store_true", help="Append --report to the history")
    parser.add_argument("--report", default=None, help="FROC report JSON (with --append)")
    parser.add_argument("--backend", default="triton", help="Backend tag for the run (triton | torch | onnx)")
    parser.add_argument("--dataset", default=None, help="Dataset tag (defaults to the report baseline name)")
    parser.add_argument("--tag", action="append", default=[], help="extra=tag pairs")
    parser.add_argument("--metric", default="cpm")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default=None, help="Write the leaderboard markdown here")
    parser.add_argument("--json", dest="json_output", default=None, help="Write the trend summary here")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging(args.log_level, name="lung_nodule.tracking", force=True)
    history_path = Path(args.history)

    if args.append:
        if not args.report:
            logger.error("--append requires --report")
            return 2
        report = read_json(args.report)
        metrics = dict(report.get("metrics", {}))
        if not metrics:
            logger.error("%s contains no metrics", args.report)
            return 2
        tags = dict(item.split("=", 1) for item in args.tag if "=" in item)
        record = make_run_record(
            metrics,
            dataset=args.dataset or str(report.get("metadata", {}).get("dataset", "unknown")),
            backend=args.backend,
            config_fingerprint=report.get("metadata", {}).get("config_fingerprint"),
            tags=tags,
            artifacts={"report": str(Path(args.report).resolve())},
        )
        append_run(record, history_path)
        logger.info("appended run %s to %s", record.run_id, history_path)

    records = load_history(history_path)
    if not records:
        logger.warning("no runs in %s yet", history_path)
        return 0

    leaderboard = render_leaderboard(records, metric=args.metric, top_k=args.top_k)
    trend = measure_trend(records, metric=args.metric)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(leaderboard, encoding="utf-8")
        logger.info("leaderboard -> %s", output)
    if args.json_output:
        write_json(args.json_output, trend)
    if not args.output:
        print(leaderboard)
    logger.info("trend: %s", trend)
    return 0


if __name__ == "__main__":
    sys.exit(main())

__all__ = [
    "HISTORY_SCHEMA_VERSION",
    "RunRecord",
    "append_run",
    "load_history",
    "main",
    "make_run_record",
    "measure_trend",
    "render_leaderboard",
]
