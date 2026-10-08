"""YAML / JSON / CSV helpers with numpy-safe serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _to_builtin(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_builtin(data), f, sort_keys=False, allow_unicode=True)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str | Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(data), f, indent=indent, ensure_ascii=False)


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict:
    """Load an experiment YAML, resolving a ``defaults: <relative path>``
    key by deep-merging the current file over its base config(s)."""
    path = Path(path)
    cfg = load_yaml(path)
    base_ref = cfg.pop("defaults", None)
    if base_ref:
        refs = base_ref if isinstance(base_ref, list) else [base_ref]
        merged: dict = {}
        for ref in refs:
            merged = deep_update(merged, load_config((path.parent / ref).resolve()))
        cfg = deep_update(merged, cfg)
    return cfg
