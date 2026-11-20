"""YAML configuration loading with dotted-path overrides.

Configs are plain nested dicts. :class:`Config` only adds convenience accessors
(``cfg.model.feature_size`` and ``cfg.get_path("model.feature_size")``) plus a
stable fingerprint used to tag FROC runs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

import yaml

_MISSING = object()


def to_plain(value: Any) -> Any:
    """Recursively convert a :class:`Config` tree into plain builtins."""
    if isinstance(value, Mapping):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


class Config(dict):
    """Nested dict with attribute access, dotted lookups and a fingerprint."""

    def __init__(self, mapping: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        payload: dict[str, Any] = {}
        if mapping is not None:
            payload.update(dict(mapping))
        payload.update(kwargs)
        for key, value in payload.items():
            self[str(key)] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, Config):
            return value
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    # Keep ``copy``/``get`` behaving like a normal dict; only unknown attributes
    # are resolved against the mapping.
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(f"{type(self).__name__!s} has no key {name!r}") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = _MISSING) -> Any:
        """Return the value stored at ``a.b.c`` or ``default`` when missing."""
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                if default is _MISSING:
                    raise KeyError(f"missing configuration key: {dotted}")
                return default
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        """Set ``a.b.c = value``, creating intermediate mappings as needed."""
        parts = dotted.split(".")
        node: MutableMapping = self
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, MutableMapping):
                child = Config()
                node[part] = child
            node = child
        node[parts[-1]] = value

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    def fingerprint(self) -> str:
        """Short stable hash of the whole config, used to tag evaluation runs."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def copy(self) -> Config:  # type: ignore[override]
        return Config(to_plain(self))

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        return cls(_read_yaml(Path(path)))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Config:
        return cls(to_plain(mapping))


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (``override`` wins)."""
    merged: dict[str, Any] = {str(k): to_plain(v) for k, v in base.items()}
    for key, value in override.items():
        key = str(key)
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = to_plain(value)
    return merged


def load_config(*paths: str | Path | None, overrides: Iterable[str] = ()) -> Config:
    """Load and merge YAML files left to right, then apply ``a.b=value`` overrides."""
    merged: dict[str, Any] = {}
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        merged = deep_merge(merged, _read_yaml(path))

    cfg = Config(merged)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key.subkey=value, got {item!r}")
        key, raw = item.split("=", 1)
        cfg.set_path(key.strip(), yaml.safe_load(raw))
    return cfg


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a YAML mapping at the top level")
    return payload


__all__ = ["Config", "deep_merge", "load_config", "to_plain"]
