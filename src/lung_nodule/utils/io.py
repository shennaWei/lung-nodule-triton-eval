"""Filesystem and provenance helpers (atomic JSON writes, git metadata, hashing)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now_iso() -> str:
    """Second-resolution UTC timestamp, e.g. ``2025-02-11T09:31:07Z``."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: str | os.PathLike[str], payload: Any, indent: int = 2) -> Path:
    """Atomically write ``payload`` as JSON (never leaves a half-written report)."""
    target = Path(path)
    ensure_dir(target.parent)
    text = json.dumps(payload, indent=indent, sort_keys=True, default=_json_default)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def read_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: str | os.PathLike[str], record: Mapping[str, Any]) -> Path:
    """Append one JSON object per line; used for the FROC run history."""
    target = Path(path)
    ensure_dir(target.parent)
    line = json.dumps(record, sort_keys=True, default=_json_default)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return target


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{target}:{lineno} is not valid JSON: {exc}") from exc
    return records


def stable_hash(payload: Any, length: int = 16) -> str:
    """Deterministic short hash of any JSON-serialisable payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(cwd: str | os.PathLike[str] | None = None) -> str:
    """Current commit SHA, or ``"unknown"`` outside a git checkout."""
    return _git(["rev-parse", "HEAD"], cwd) or "unknown"


def git_short_sha(cwd: str | os.PathLike[str] | None = None) -> str:
    return _git(["rev-parse", "--short", "HEAD"], cwd) or "unknown"


def git_dirty(cwd: str | os.PathLike[str] | None = None) -> bool:
    """True when the working tree has uncommitted changes."""
    status = _git(["status", "--porcelain"], cwd)
    return bool(status)


def git_branch(cwd: str | os.PathLike[str] | None = None) -> str:
    return _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd) or "unknown"


def copy_artifact(src: str | os.PathLike[str], dst_dir: str | os.PathLike[str]) -> Path:
    src = Path(src)
    dst = ensure_dir(dst_dir) / src.name
    shutil.copy2(src, dst)
    return dst


def _git(args: list[str], cwd: str | os.PathLike[str] | None = None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):  # numpy scalars / arrays
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


__all__ = [
    "append_jsonl",
    "copy_artifact",
    "ensure_dir",
    "git_branch",
    "git_dirty",
    "git_sha",
    "git_short_sha",
    "read_json",
    "read_jsonl",
    "sha256_file",
    "stable_hash",
    "utc_now_iso",
    "write_json",
]
