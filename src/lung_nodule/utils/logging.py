"""Console logging setup shared by every CLI entry point."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str | int = "INFO",
    log_file: str | os.PathLike[str] | None = None,
    name: str = "lung_nodule",
    force: bool = False,
) -> logging.Logger:
    """Configure a stream handler (and optional file handler) on ``name``.

    Repeated calls are idempotent unless ``force`` is set, which keeps the CLIs
    quiet when they are imported from a notebook.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers and not force:
        return logger
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        target = Path(log_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "lung_nodule") -> logging.Logger:
    return logging.getLogger(name)


__all__ = ["get_logger", "setup_logging"]
