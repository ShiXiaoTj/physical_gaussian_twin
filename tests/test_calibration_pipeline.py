"""System test: synthetic friction recovery (Phase 2 milestone).

Ground-truth friction 0.8, initial guess 0.2. The calibrator must recover
0.75..0.85 and reduce trajectory RMSE substantially. This is the gate for
proceeding to a real simulator backend.
"""

import numpy as np
import pytest

from pgtwin.physics.calibrator import HierarchicalCalibrator, PhysicsCalibrator
from pgtwin.physics.losses import TrajectoryLoss
from pgtwin.physics.optimizer import GridSearchOptimizer, RandomSearchOptimizer
from pgtwin.physics.parameter_space import ParameterSpace
from pgtwin.physics.sensitivity import (
    parameters_for_action,
    validate_identifiability,
)
from pgtwin.real_data.episode import ActionType
from pgtwin.real_data.synthetic import SyntheticEpisodeGenerator, SyntheticPushConfig
from pgtwin.simulator.mock_adapter import MockSimulator

FRICTION_SPACE = {"friction": {"min": 0.05, "max": 1.5, "step": 0.05, "initial": 0.2}}


def run_calibration(episode, obj, space_cfg=FRICTION_SPACE, optimizer=None):
    sim = MockSimulator(dt=0.002)
    calibrator = PhysicsCalibrator(sim, loss=TrajectoryLoss(), settle_time=0.5)
    result = calibrator.calibrate(
        episode,
        obj,
        ParameterSpace.from_config(space_cfg),
        optimizer or GridSearchOptimizer(),
    )
    sim.close()
    return result


class TestSyntheticFrictionRecovery:
    """The headline milestone test."""

    def test_recovers_ground_truth_friction(self, synthetic_push):
        episode, obj = synthetic_push
        result = run_calibration(episode, obj)
        assert 0.75 <= result.optimized_params["friction"] <= 0.85, (
            f"recovered {result.optimized_params['friction']}, expected ~0.8"
        )

    def test_reduces_trajectory_error(self, synthetic_push):
        episode, obj = synthetic_push
        result = run_calibration(episode, obj)
        assert result.final_loss < result.initial_loss
        assert result.final_metrics["rmse_position_cm"] < (
            result.initial_metrics["rmse_position_cm"]
        )
        assert result.improvement["rmse_position_cm"] > 50.0

    def test_calibrated_error_is_small(self, synthetic_push):
        """Noise-free synthetic data -> near-exact match after calibration."""
        episode, obj = synthetic_push
        result = run_calibration(episode, obj)
        assert result.final_metrics["rmse_position_cm"] < 0.5

    def test_ee_replay_is_faithful(self, synthetic_push):
        """Guards against attributing replay error to physics."""
        episode, obj = synthetic_push
        result = run_calibration(episode, obj)
        assert result.ee_replay_rmse_m < 1e-3

    def test_result_is_serializable(self, synthetic_push, tmp_path):
        episode, obj = synthetic_push
        result = run_calibration(episode, obj)
        data = result.to_dict()
        for key in ("initial_params", "optimized_params", "initial_metrics",
                    "final_metrics", "improvement"):
            assert key in data
        import json

        json.loads(json.dumps(data))  # must be JSON-clean

    def test_history_recorded(self, synthetic_push):
        episode, obj = synthetic_push
        result = run_calibration(episode, obj)
        assert result.optimization.n_evaluations == 30
        losses = [ev.loss for ev in result.optimization.history]
        assert min(losses) == pytest.approx(result.final_loss)

    def test_random_search_also_recovers(self, synthetic_push):
        episode, obj = synthetic_push
        result = run_calibration(
            episode, obj, optimizer=RandomSearchOptimizer(n_samples=40, seed=0)
        )
        assert 0.7 <= result.optimized_params["friction"] <= 0.9


class TestRobustness:
    def test_recovery_with_measurement_noise(self):
        """1 mm noise must not break recovery (widened tolerance)."""
        gen = SyntheticEpisodeGenerator(
            SyntheticPushConfig(
                episode_id="noisy", true_friction=0.8, noise_std_pos=0.001, seed=3
            )
        )
        episode, obj = gen.generate()
        result = run_calibration(episode, obj)
        assert 0.7 <= result.optimized_params["friction"] <= 0.9

    @pytest.mark.parametrize("true_friction", [0.3, 0.6, 1.0])
    def test_recovery_across_friction_values(self, true_friction):
        gen = SyntheticEpisodeGenerator(
            SyntheticPushConfig(
                episode_id=f"mu_{true_friction}",
                true_friction=true_friction,
                noise_std_pos=0.0,
            )
        )
        episode, obj = gen.generate()
        result = run_calibration(episode, obj)
        assert result.optimized_params["friction"] == pytest.approx(
            true_friction, abs=0.08
        )

    def test_repeated_runs_are_deterministic(self, synthetic_push):
        episode, obj = synthetic_push
        a = run_calibration(episode, obj)
        b = run_calibration(episode, obj)
        assert a.optimized_params == b.optimized_params
        assert a.final_loss == pytest.approx(b.final_loss)


class TestSensitivityPolicy:
    def test_push_selects_friction_first(self):
        assert parameters_for_action(ActionType.PUSH)[0] == "friction"

    def test_lift_selects_mass(self):
        assert parameters_for_action(ActionType.LIFT) == ["mass"]

    def test_collision_selects_restitution(self):
        assert "restitution" in parameters_for_action(ActionType.COLLISION)

    def test_push_rejects_joint_mass_friction(self):
        """Identifiability guard: mass+friction from one push is ambiguous."""
        with pytest.raises(ValueError, match="identifiab"):
            validate_identifiability(ActionType.PUSH, ["friction", "mass"])

    def test_push_accepts_friction_only(self):
        validate_identifiability(ActionType.PUSH, ["friction"])

    def test_warns_on_insensitive_parameter(self):
        with pytest.raises(ValueError, match="not sensitive"):
            validate_identifiability(ActionType.PUSH, ["young_modulus"])


class TestHierarchicalCalibrator:
    def test_stage_order_respects_sensitivity(self, synthetic_push):
        episode, obj = synthetic_push
        sim = MockSimulator(dt=0.002)
        hc = HierarchicalCalibrator(sim, settle_time=0.5)
        results = hc.calibrate([(episode, FRICTION_SPACE)], obj)
        sim.close()
        assert len(results) == 1
        assert 0.75 <= results[0].optimized_params["friction"] <= 0.85
