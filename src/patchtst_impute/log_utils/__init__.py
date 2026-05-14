"""Logging bootstrap (console + append-only file). Avoids naming the package `logging`."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_dir: Path, level: str, log_name: str) -> None:
    """
    Configure root logger with stdout and UTF-8 file handler.
    File opens in append mode so existing log lines are preserved.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    lvl = getattr(logging, level.upper(), None)
    if not isinstance(lvl, int):
        lvl = logging.INFO

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(lvl)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


__all__ = ["setup_logging"]
