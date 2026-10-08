"""Searchable physics parameter spaces."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class ParameterSpec:
    """One searchable physics parameter with bounds and grid resolution."""

    name: str
    min: float
    max: float
    step: float | None = None  # grid step; if None use num
    num: int | None = None  # number of grid points; used if step is None
    log_scale: bool = False
    initial: float | None = None  # initial guess / baseline value

    def __post_init__(self) -> None:
        if self.max <= self.min:
            raise ValueError(f"{self.name}: max ({self.max}) must be > min ({self.min})")
        if self.step is None and self.num is None:
            self.num = 20
        if self.initial is None:
            self.initial = 0.5 * (self.min + self.max)
        self.initial = float(np.clip(self.initial, self.min, self.max))

    def grid(self) -> np.ndarray:
        """Grid values (inclusive of both bounds)."""
        if self.log_scale:
            n = self.num or max(2, int(round(np.log(self.max / self.min) / self.step)) + 1)
            return np.geomspace(self.min, self.max, n)
        if self.step is not None:
            vals = np.arange(self.min, self.max + 0.5 * self.step, self.step)
            return np.clip(vals, self.min, self.max)
        return np.linspace(self.min, self.max, self.num)

    def sample(self, rng: np.random.Generator) -> float:
        if self.log_scale:
            return float(np.exp(rng.uniform(np.log(self.min), np.log(self.max))))
        return float(rng.uniform(self.min, self.max))

    def clip(self, value: float) -> float:
        return float(np.clip(value, self.min, self.max))

    @classmethod
    def from_config(cls, name: str, cfg: dict) -> "ParameterSpec":
        return cls(
            name=name,
            min=float(cfg["min"]),
            max=float(cfg["max"]),
            step=float(cfg["step"]) if "step" in cfg else None,
            num=int(cfg["num"]) if "num" in cfg else None,
            log_scale=bool(cfg.get("log_scale", False)),
            initial=float(cfg["initial"]) if "initial" in cfg else None,
        )


class ParameterSpace:
    """Ordered collection of :class:`ParameterSpec` defining the search space."""

    def __init__(self, specs: list[ParameterSpec]):
        if not specs:
            raise ValueError("parameter space must contain at least one parameter")
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names: {names}")
        self.specs = list(specs)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, name: str) -> ParameterSpec:
        for s in self.specs:
            if s.name == name:
                return s
        raise KeyError(name)

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def initial_params(self) -> dict[str, float]:
        return {s.name: float(s.initial) for s in self.specs}

    def grid(self) -> list[dict[str, float]]:
        """Cartesian product of per-parameter grids as parameter dicts."""
        axes = [s.grid() for s in self.specs]
        return [
            {name: float(v) for name, v in zip(self.names, combo)}
            for combo in itertools.product(*axes)
        ]

    def grid_size(self) -> int:
        n = 1
        for s in self.specs:
            n *= len(s.grid())
        return n

    def sample(self, rng: np.random.Generator) -> dict[str, float]:
        return {s.name: s.sample(rng) for s in self.specs}

    def clip(self, params: dict[str, float]) -> dict[str, float]:
        return {s.name: s.clip(params[s.name]) for s in self.specs}

    def subspace(self, names: list[str]) -> "ParameterSpace":
        """Restrict to a subset of parameters (used by calibration policies)."""
        return ParameterSpace([self[n] for n in names if n in self])

    @classmethod
    def from_config(cls, cfg: dict) -> "ParameterSpace":
        """Build from a config mapping ``{name: {min, max, step, ...}}``."""
        return cls([ParameterSpec.from_config(name, c) for name, c in cfg.items()])
