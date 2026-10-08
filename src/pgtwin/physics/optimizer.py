"""Black-box parameter optimizers over :class:`ParameterSpace`.

An optimizer minimizes ``objective(params: dict) -> float`` and records the
full evaluation history for reproducibility and plotting.

Implemented: grid search (Phase 1 default), random search.
Interface-reserved: Bayesian optimization, differentiable optimization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pgtwin.common.logging import get_logger
from pgtwin.physics.parameter_space import ParameterSpace

logger = get_logger("physics.optimizer")

Objective = Callable[[dict[str, float]], float]


@dataclass
class Evaluation:
    """One objective evaluation."""

    params: dict[str, float]
    loss: float
    index: int


@dataclass
class OptimizationResult:
    best_params: dict[str, float]
    best_loss: float
    history: list[Evaluation] = field(default_factory=list)

    @property
    def n_evaluations(self) -> int:
        return len(self.history)

    def history_arrays(self) -> dict[str, np.ndarray]:
        """Columns: each parameter + loss, ordered by evaluation index."""
        out: dict[str, list[float]] = {"loss": []}
        for ev in self.history:
            for k, v in ev.params.items():
                out.setdefault(k, []).append(v)
            out["loss"].append(ev.loss)
        return {k: np.asarray(v) for k, v in out.items()}


class BaseOptimizer(ABC):
    """Minimize an objective over a parameter space."""

    name: str = "base"

    @abstractmethod
    def optimize(self, objective: Objective, space: ParameterSpace) -> OptimizationResult:
        ...

    @staticmethod
    def _run_batch(
        objective: Objective, candidates: list[dict[str, float]]
    ) -> OptimizationResult:
        history: list[Evaluation] = []
        best_params, best_loss = None, np.inf
        for i, params in enumerate(candidates):
            loss = float(objective(params))
            history.append(Evaluation(params=dict(params), loss=loss, index=i))
            if np.isfinite(loss) and loss < best_loss:
                best_loss, best_params = loss, dict(params)
            logger.debug("eval %d/%d: %s -> %.6f", i + 1, len(candidates), params, loss)
        if best_params is None:
            raise RuntimeError("all objective evaluations returned non-finite loss")
        return OptimizationResult(best_params, best_loss, history)


class GridSearchOptimizer(BaseOptimizer):
    """Exhaustive grid search — the Phase 1 default.

    Deterministic, trivially reproducible, and gives the full
    loss-vs-parameter curve for free (identifiability diagnostics).
    """

    name = "grid_search"

    def optimize(self, objective: Objective, space: ParameterSpace) -> OptimizationResult:
        candidates = space.grid()
        logger.info(
            "grid search: %d candidates over %s", len(candidates), space.names
        )
        return self._run_batch(objective, candidates)


class RandomSearchOptimizer(BaseOptimizer):
    """Uniform random search with a fixed seed (fallback / high-dim spaces)."""

    name = "random_search"

    def __init__(self, n_samples: int = 50, seed: int = 0):
        self.n_samples = n_samples
        self.seed = seed

    def optimize(self, objective: Objective, space: ParameterSpace) -> OptimizationResult:
        rng = np.random.default_rng(self.seed)
        candidates = [space.sample(rng) for _ in range(self.n_samples)]
        logger.info(
            "random search: %d samples over %s (seed=%d)",
            self.n_samples,
            space.names,
            self.seed,
        )
        return self._run_batch(objective, candidates)


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization (Phase 3+; requires scikit-optimize)."""

    name = "bayesian"

    def __init__(self, n_calls: int = 30, seed: int = 0):
        self.n_calls = n_calls
        self.seed = seed

    def optimize(self, objective: Objective, space: ParameterSpace) -> OptimizationResult:
        try:
            from skopt import gp_minimize  # type: ignore
            from skopt.space import Real  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise NotImplementedError(
                "BayesianOptimizer requires scikit-optimize: pip install scikit-optimize"
            ) from exc

        dims = [Real(s.min, s.max, name=s.name) for s in space.specs]
        history: list[Evaluation] = []

        def wrapped(x: list[float]) -> float:
            params = {name: float(v) for name, v in zip(space.names, x)}
            loss = float(objective(params))
            history.append(Evaluation(params, loss, len(history)))
            return loss

        res = gp_minimize(
            wrapped, dims, n_calls=self.n_calls, random_state=self.seed
        )  # pragma: no cover
        best = {name: float(v) for name, v in zip(space.names, res.x)}
        return OptimizationResult(best, float(res.fun), history)  # pragma: no cover


class DifferentiableOptimizer(BaseOptimizer):
    """Gradient-based calibration through a differentiable simulator.

    Interface reserved for Phase 3+ (e.g. Warp / Brax / DiffTaichi
    backends).  Black-box optimizers cover the PoC.
    """

    name = "differentiable"

    def optimize(self, objective: Objective, space: ParameterSpace) -> OptimizationResult:
        raise NotImplementedError(
            "Differentiable calibration requires a differentiable simulator "
            "backend; planned for a later phase."
        )


def create_optimizer(cfg: dict) -> BaseOptimizer:
    """Factory from a ``calibration:`` config section.

    Accepts ``method:`` or ``optimizer:`` with short or long names
    (``grid``/``grid_search``, ``random``/``random_search``, ...).
    """
    method = str(cfg.get("method", cfg.get("optimizer", "grid_search")))
    aliases = {"grid": "grid_search", "random": "random_search"}
    method = aliases.get(method, method)
    if method == "grid_search":
        return GridSearchOptimizer()
    if method == "random_search":
        return RandomSearchOptimizer(
            n_samples=int(cfg.get("n_samples", 50)), seed=int(cfg.get("seed", 0))
        )
    if method == "bayesian":
        return BayesianOptimizer(
            n_calls=int(cfg.get("n_calls", 30)), seed=int(cfg.get("seed", 0))
        )
    if method == "differentiable":
        return DifferentiableOptimizer()
    raise ValueError(f"unknown optimization method '{method}'")
