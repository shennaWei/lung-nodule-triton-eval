"""PEP 562 lazy exports.

Heavy subpackages (``torch``, ``monai``, ``tritonclient``) should only be imported
when the caller actually touches a symbol that needs them. The evaluation CLI has
to run on a CPU-only runner with just NumPy/Pandas/SciPy installed, so
``import lung_nodule.data.luna16`` must not drag in PyTorch through
``lung_nodule/data/__init__.py``.
"""

from __future__ import annotations

import importlib
from typing import Any


def make_lazy_getattr(package: str, exports: dict[str, tuple[str, str]]):
    """Build a module-level ``__getattr__`` for a mapping ``name -> (module, attr)``."""

    def __getattr__(name: str) -> Any:
        try:
            module_name, attribute = exports[name]
        except KeyError as exc:
            raise AttributeError(f"module {package!r} has no attribute {name!r}") from exc
        value = getattr(importlib.import_module(module_name), attribute)
        globals()[name] = value
        return value

    return __getattr__


def make_dir(namespace: dict[str, Any], names: list[str]):
    """Build a module-level ``__dir__`` from the caller's ``globals()``."""

    def __dir__() -> list[str]:
        return sorted(set(namespace) | set(names))

    return __dir__


__all__ = ["make_dir", "make_lazy_getattr"]
