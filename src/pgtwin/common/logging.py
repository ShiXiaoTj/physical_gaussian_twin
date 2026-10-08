"""Project-wide logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"
_configured = False


def configure(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure the root ``pgtwin`` logger once (console + optional file)."""
    global _configured
    root = logging.getLogger("pgtwin")
    if _configured:
        root.setLevel(level)
        return
    root.setLevel(level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    root.addHandler(console)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        root.addHandler(fh)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the ``pgtwin`` namespace."""
    configure()
    if not name.startswith("pgtwin"):
        name = f"pgtwin.{name}"
    return logging.getLogger(name)
