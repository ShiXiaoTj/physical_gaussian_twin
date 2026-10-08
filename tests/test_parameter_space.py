"""Parameter space and optimizer (grid / random) tests."""

import numpy as np
import pytest

from pgtwin.physics.optimizer import (
    DifferentiableOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    create_optimizer,
)
from pgtwin.physics.parameter_space import ParameterSpace, ParameterSpec


class TestParameterSpace:
    def test_grid_values_inclusive(self):
        spec = ParameterSpec("friction", 0.1, 0.5, step=0.1)
        np.testing.assert_allclose(spec.grid(), [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_clip(self):
        spec = ParameterSpec("friction", 0.05, 1.5)
        assert spec.clip(3.0) == 1.5
        assert spec.clip(-1.0) == 0.05

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            ParameterSpec("friction", 1.0, 0.5)

    def test_from_config(self):
        space = ParameterSpace.from_config(
            {"friction": {"min": 0.05, "max": 1.5, "step": 0.05, "initial": 0.2}}
        )
        assert space.names == ["friction"]
        assert space.initial_params()["friction"] == pytest.approx(0.2)

    def test_grid_is_cartesian_product(self):
        space = ParameterSpace(
            [
                ParameterSpec("friction", 0.1, 0.3, step=0.1),
                ParameterSpec("mass", 1.0, 2.0, step=0.5),
            ]
        )
        assert len(space.grid()) == 9
        assert space.grid_size() == 9

    def test_default_initial_is_midpoint(self):
        space = ParameterSpace([ParameterSpec("friction", 0.0, 1.0)])
        assert space.initial_params()["friction"] == pytest.approx(0.5)

    def test_sample_within_bounds(self):
        space = ParameterSpace.from_config({"friction": {"min": 0.2, "max": 0.9}})
        rng = np.random.default_rng(1)
        for _ in range(50):
            params = space.sample(rng)
            assert 0.2 <= params["friction"] <= 0.9

    def test_subspace(self):
        space = ParameterSpace(
            [ParameterSpec("friction", 0.1, 1.0), ParameterSpec("mass", 0.1, 2.0)]
        )
        sub = space.subspace(["friction"])
        assert sub.names == ["friction"]

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError):
            ParameterSpace(
                [ParameterSpec("friction", 0.1, 1.0), ParameterSpec("friction", 0.2, 0.9)]
            )


class TestOptimizers:
    """Optimizers must recover the minimum of a known 1-D objective."""

    @staticmethod
    def quadratic(params: dict) -> float:
        return (params["friction"] - 0.8) ** 2

    def test_grid_search_finds_minimum(self):
        space = ParameterSpace.from_config(
            {"friction": {"min": 0.05, "max": 1.5, "step": 0.05}}
        )
        result = GridSearchOptimizer().optimize(self.quadratic, space)
        assert result.best_params["friction"] == pytest.approx(0.8, abs=0.051)
        assert result.n_evaluations == len(space.grid())

    def test_history_best_matches_best_loss(self):
        space = ParameterSpace.from_config(
            {"friction": {"min": 0.05, "max": 1.5, "step": 0.1}}
        )
        result = GridSearchOptimizer().optimize(self.quadratic, space)
        assert min(ev.loss for ev in result.history) == pytest.approx(result.best_loss)

    def test_history_arrays(self):
        space = ParameterSpace.from_config(
            {"friction": {"min": 0.1, "max": 0.5, "step": 0.1}}
        )
        result = GridSearchOptimizer().optimize(self.quadratic, space)
        arrays = result.history_arrays()
        assert set(arrays) == {"friction", "loss"}
        assert len(arrays["loss"]) == result.n_evaluations

    def test_random_search_finds_minimum(self):
        space = ParameterSpace.from_config({"friction": {"min": 0.05, "max": 1.5}})
        result = RandomSearchOptimizer(n_samples=200, seed=0).optimize(
            self.quadratic, space
        )
        assert result.best_params["friction"] == pytest.approx(0.8, abs=0.05)

    def test_random_search_is_seed_deterministic(self):
        space = ParameterSpace.from_config({"friction": {"min": 0.05, "max": 1.5}})
        a = RandomSearchOptimizer(n_samples=25, seed=7).optimize(self.quadratic, space)
        b = RandomSearchOptimizer(n_samples=25, seed=7).optimize(self.quadratic, space)
        assert a.best_params == b.best_params

    def test_factory_accepts_short_names(self):
        assert create_optimizer({"optimizer": "grid"}).name == "grid_search"
        assert (
            create_optimizer({"method": "random_search", "n_samples": 5}).name
            == "random_search"
        )

    def test_factory_rejects_unknown(self):
        with pytest.raises(ValueError):
            create_optimizer({"optimizer": "nope"})

    def test_differentiable_is_interface_only(self):
        space = ParameterSpace.from_config({"friction": {"min": 0.1, "max": 1.0}})
        with pytest.raises(NotImplementedError):
            DifferentiableOptimizer().optimize(self.quadratic, space)
