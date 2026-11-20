"""FROC regression gate.

Compares the metrics of a fresh FROC report against a committed baseline and
returns a non-zero exit code when any tracked metric drops beyond its tolerance.
This is what the ``froc-regression`` GitHub workflow runs on every change to the
model, the post-processing or the exported graphs.

Baseline schema (``eval/baselines/*.json``)::

    {
      "schema_version": 1,
      "name": "luna16-fold0-val",
      "dataset": {"annotations": "...", "split": "...", "n_scans": 89},
      "metrics": {"cpm": 0.8659, "sens@1fp": 0.889, "fp_at_sens0.90": 2.31},
      "tolerance": {"default": 0.01, "cpm": 0.005},
      "direction": {"fp_at_sens0.90": "lower_is_better"},
      "require": ["cpm"]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lung_nodule.utils.io import ensure_dir, git_sha, read_json, utc_now_iso, write_json
from lung_nodule.utils.logging import setup_logging

BASELINE_SCHEMA_VERSION = 1
DEFAULT_TOLERANCE = 0.01
LOWER_IS_BETTER = "lower_is_better"
HIGHER_IS_BETTER = "higher_is_better"


@dataclass
class MetricCheck:
    name: str
    baseline: float | None
    current: float | None
    delta: float | None
    tolerance: float
    direction: str
    status: str
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.name,
            "baseline": self.baseline,
            "current": self.current,
            "delta": self.delta,
            "tolerance": self.tolerance,
            "direction": self.direction,
            "status": self.status,
            "passed": self.passed,
        }


@dataclass
class RegressionReport:
    passed: bool
    checks: list[MetricCheck]
    baseline_name: str
    baseline_path: str
    report_path: str
    created_at: str
    git_sha: str
    dataset: dict[str, Any] = field(default_factory=dict)
    config_fingerprint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "passed": self.passed,
            "baseline": {"name": self.baseline_name, "path": self.baseline_path},
            "report": self.report_path,
            "created_at": self.created_at,
            "git_sha": self.git_sha,
            "dataset": self.dataset,
            "config_fingerprint": self.config_fingerprint,
            "metadata": self.metadata,
            "checks": [check.to_dict() for check in self.checks],
            "summary": {
                "n_checks": len(self.checks),
                "n_failed": sum(not check.passed for check in self.checks),
                "n_regressed": sum(check.status == "regressed" for check in self.checks),
            },
        }

    def render_markdown(self, title: str = "FROC regression check") -> str:
        status_icon = {
            "ok": "OK",
            "improved": "IMPROVED",
            "regressed": "REGRESSED",
            "missing": "MISSING",
            "new": "NEW",
        }
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"# {title}",
            "",
            f"**Verdict: {verdict}** ({sum(not c.passed for c in self.checks)} failing checks)",
            "",
            f"- baseline: `{self.baseline_name}` ({self.baseline_path})",
            f"- report: `{self.report_path}`",
            f"- commit: `{self.git_sha}`",
            f"- generated: {self.created_at}",
            "",
            "| metric | baseline | current | delta | tolerance | status |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for check in self.checks:
            baseline = _fmt(check.baseline)
            current = _fmt(check.current)
            delta = _fmt(check.delta, signed=True)
            lines.append(
                f"| `{check.name}` | {baseline} | {current} | {delta} | ±{check.tolerance:g} | "
                f"{status_icon.get(check.status, check.status)} |"
            )
        failing = [check for check in self.checks if not check.passed]
        if failing:
            lines.extend(["", "### Failing checks", ""])
            for check in failing:
                lines.append(
                    f"- `{check.name}`: {_fmt(check.baseline)} -> {_fmt(check.current)} "
                    f"(tolerance ±{check.tolerance:g}, direction {check.direction})"
                )
        return "\n".join(lines) + "\n"


def load_baseline(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    version = int(payload.get("schema_version", 0))
    if version != BASELINE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported baseline schema {version} in {path}; expected {BASELINE_SCHEMA_VERSION}"
        )
    if "metrics" not in payload:
        raise ValueError(f"{path} has no 'metrics' section")
    return payload


def evaluate_regression(
    report: Mapping[str, Any],
    baseline: Mapping[str, Any],
    default_tolerance: float = DEFAULT_TOLERANCE,
    allow_missing: bool = False,
    baseline_path: str = "",
    report_path: str = "",
) -> RegressionReport:
    """Compare ``report['metrics']`` against the baseline, metric by metric."""
    current_metrics = dict(report.get("metrics", {}))
    baseline_metrics = dict(baseline.get("metrics", {}))
    tolerances = dict(baseline.get("tolerance", {}))
    directions = dict(baseline.get("direction", {}))
    required = set(baseline.get("require", []))

    checks: list[MetricCheck] = []

    for name, baseline_value in baseline_metrics.items():
        tolerance = float(tolerances.get(name, tolerances.get("default", default_tolerance)))
        direction = str(directions.get(name, HIGHER_IS_BETTER))
        current_value = current_metrics.get(name)

        if current_value is None:
            passed = allow_missing and name not in required
            checks.append(
                MetricCheck(
                    name=name,
                    baseline=_as_float(baseline_value),
                    current=None,
                    delta=None,
                    tolerance=tolerance,
                    direction=direction,
                    status="missing",
                    passed=passed,
                )
            )
            continue

        current_float = float(current_value)
        baseline_float = float(baseline_value)
        delta = current_float - baseline_float
        if direction == LOWER_IS_BETTER:
            regressed = current_float > baseline_float + tolerance
            improved = current_float < baseline_float - tolerance
        else:
            regressed = current_float < baseline_float - tolerance
            improved = current_float > baseline_float + tolerance

        status = "regressed" if regressed else ("improved" if improved else "ok")
        checks.append(
            MetricCheck(
                name=name,
                baseline=baseline_float,
                current=current_float,
                delta=delta,
                tolerance=tolerance,
                direction=direction,
                status=status,
                passed=not regressed,
            )
        )

    for name, value in current_metrics.items():
        if name in baseline_metrics:
            continue
        checks.append(
            MetricCheck(
                name=name,
                baseline=None,
                current=float(value),
                delta=None,
                tolerance=float(tolerances.get("default", default_tolerance)),
                direction=str(directions.get(name, HIGHER_IS_BETTER)),
                status="new",
                passed=True,
            )
        )

    dataset = dict(baseline.get("dataset", {}))
    if "counts" in report:
        dataset.setdefault("counts", report["counts"])

    return RegressionReport(
        passed=all(check.passed for check in checks),
        checks=checks,
        baseline_name=str(baseline.get("name", Path(baseline_path).stem)),
        baseline_path=str(baseline_path),
        report_path=str(report_path),
        created_at=utc_now_iso(),
        git_sha=git_sha(),
        dataset=dataset,
        config_fingerprint=report.get("metadata", {}).get("config_fingerprint")
        if isinstance(report.get("metadata"), Mapping)
        else None,
        metadata={"report_metadata": report.get("metadata", {})},
    )


def write_regression_report(
    result: RegressionReport,
    output_dir: str | Path,
    name: str = "regression",
    title: str = "FROC regression check",
) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    paths = {
        "json": write_json(output_dir / f"{name}.json", result.to_dict()),
    }
    markdown_path = output_dir / f"{name}.md"
    markdown_path.write_text(result.render_markdown(title), encoding="utf-8")
    paths["markdown"] = markdown_path
    return paths


def update_baseline(
    baseline_path: str | Path, report_path: str | Path, output_path: str | Path | None = None
) -> Path:
    """Promote the metrics of ``report_path`` into the baseline (keeps tolerances)."""
    baseline = load_baseline(baseline_path)
    report = read_json(report_path)
    baseline["metrics"] = dict(report.get("metrics", {}))
    baseline["updated_at"] = utc_now_iso()
    baseline["updated_from"] = str(Path(report_path).resolve())
    baseline["git_sha"] = git_sha()

    target = Path(output_path or baseline_path)
    if output_path is not None:
        ensure_dir(Path(output_path).parent)
    return write_json(target, baseline)


def append_github_summary(markdown: str, path: str | None = None) -> Path | None:
    """Append the markdown report to ``$GITHUB_STEP_SUMMARY`` when running in Actions."""
    target = path or os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return None
    summary_path = Path(target)
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(markdown)
        handle.write("\n")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--report", required=True, help="FROC report JSON (lndet-eval-froc output)")
    parser.add_argument("--baseline", required=True, help="Committed baseline JSON")
    parser.add_argument("--output-dir", default="artifacts/froc")
    parser.add_argument("--name", default="regression")
    parser.add_argument("--title", default="FROC regression check")
    parser.add_argument("--default-tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--allow-missing", action="store_true", help="Do not fail when a baseline metric is absent"
    )
    parser.add_argument("--no-fail", action="store_true", help="Exit 0 even when a metric regressed")
    parser.add_argument(
        "--update-baseline", action="store_true", help="Write the current metrics back to the baseline"
    )
    parser.add_argument("--baseline-output", default=None, help="Where to write the updated baseline")
    parser.add_argument("--summary", action="store_true", help="Append the report to $GITHUB_STEP_SUMMARY")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging(args.log_level, name="lung_nodule.regression", force=True)

    try:
        report = read_json(args.report)
        baseline = load_baseline(args.baseline)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("cannot load inputs: %s", exc)
        return 2

    result = evaluate_regression(
        report,
        baseline,
        default_tolerance=args.default_tolerance,
        allow_missing=args.allow_missing,
        baseline_path=args.baseline,
        report_path=args.report,
    )
    paths = write_regression_report(result, args.output_dir, name=args.name, title=args.title)

    markdown = paths["markdown"].read_text(encoding="utf-8")
    if args.summary:
        written = append_github_summary(markdown)
        if written is not None:
            logger.info("appended report to %s", written)
    for line in markdown.splitlines():
        logger.info("%s", line)

    if args.update_baseline:
        target = update_baseline(args.baseline, args.report, args.baseline_output)
        logger.info("baseline updated -> %s", target)
        shutil.copy2(paths["markdown"], Path(args.output_dir) / "regression_applied.md")

    if not result.passed and not args.no_fail:
        logger.error(
            "FROC regression detected: %d failing metric(s)", sum(not c.passed for c in result.checks)
        )
        return 1
    logger.info("FROC regression check passed")
    return 0


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "-"
    if signed:
        return f"{value:+.5g}"
    return f"{value:.5g}"


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "MetricCheck",
    "RegressionReport",
    "append_github_summary",
    "evaluate_regression",
    "load_baseline",
    "main",
    "update_baseline",
    "write_regression_report",
]
