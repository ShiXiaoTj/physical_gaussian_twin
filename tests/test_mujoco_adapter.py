"""MuJoCo adapter validation (Phase 3).

Skipped when mujoco is not installed. Covers: BaseSimulator contract,
compliant-pusher replay, friction identifiability, same-backend synthetic
recovery (the Phase 3 gate), and the cross-backend actuation guardrail.
"""

import numpy as np
import pytest

from pgtwin.common.se3 import Pose
from pgtwin.physics.calibrator import PhysicsCalibrator
from pgtwin.physics.losses import TrajectoryLoss
from pgtwin.physics.optimizer import GridSearchOptimizer
from pgtwin.physics.parameter_space import ParameterSpace
from pgtwin.physics.physical_object import PhysicalObject
from pgtwin.real_data.synthetic import SyntheticEpisodeGenerator, SyntheticPushConfig
from pgtwin.simulator import create_simulator
from pgtwin.simulator.mujoco_adapter import MUJOCO_AVAILABLE

pytestmark = pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="mujoco not installed")

SIM_CFG = {"backend": "mujoco", "dt": 0.001}
SPACE = ParameterSpace.from_config(
    {"friction": {"min": 0.05, "max": 1.5, "step": 0.05, "initial": 0.2}}
)


def make_object(friction: float = 0.5) -> PhysicalObject:
    return PhysicalObject.from_config(
        {
            "object_id": "box_01",
            "shape": "box",
            "size": [0.08, 0.08, 0.04],  # squat: stays in sliding regime
            "physics": {"mass": 0.42, "friction_dynamic": friction},
        }
    )


def make_sim(**overrides):
    sim = create_simulator({**SIM_CFG, **overrides})
    sim.load_scene({"table_height": 0.0})
    sim.load_robot({"ee_radius": 0.012})
    return sim


def straight_push(sim, obj, mu, speed=0.5, settle=1.5):
    sim.reset()
    sim.set_object_pose(obj.object_id, Pose(position=[0.5, 0.0, 0.02]))
    sim.set_physics_parameters(obj.object_id, {"friction": mu})
    ts = np.arange(0.0, 0.5, 0.005)
    ee = [Pose(position=[0.38 + speed * t, 0.0, 0.02]) for t in ts]
    return sim.apply_ee_trajectory(ts, ee, object_id=obj.object_id, settle_time=settle)


class TestContract:
    def test_is_available(self):
        from pgtwin.simulator.mujoco_adapter import MuJoCoSimulator

        assert MuJoCoSimulator.is_available()

    def test_pose_roundtrip(self):
        sim = make_sim()
        obj = make_object()
        sim.load_object(obj, Pose(position=[0.5, 0.1, 0.02]))
        state = sim.get_object_state("box_01")
        np.testing.assert_allclose(state.pose.position, [0.5, 0.1, 0.02], atol=1e-9)
        sim.set_object_pose("box_01", Pose.from_xyz_yaw(0.4, -0.1, 0.02, 0.3))
        state = sim.get_object_state("box_01")
        np.testing.assert_allclose(state.pose.position, [0.4, -0.1, 0.02], atol=1e-9)
        assert state.pose.yaw == pytest.approx(0.3, abs=1e-6)
        sim.close()

    def test_reset_restores_pose_and_time(self):
        sim = make_sim()
        obj = make_object()
        sim.load_object(obj, Pose(position=[0.5, 0.0, 0.02]))
        straight_push(sim, obj, 0.5, settle=0.2)
        sim.reset()
        assert sim.time == pytest.approx(0.0)
        np.testing.assert_allclose(
            sim.get_object_state("box_01").pose.position, [0.5, 0.0, 0.02], atol=1e-9
        )
        sim.close()

    def test_deterministic_across_resets(self):
        sim = make_sim()
        obj = make_object()
        sim.load_object(obj, Pose(position=[0.5, 0.0, 0.02]))
        a = straight_push(sim, obj, 0.7, settle=0.5).object_trajectory.positions[-1]
        b = straight_push(sim, obj, 0.7, settle=0.5).object_trajectory.positions[-1]
        np.testing.assert_allclose(a, b, atol=1e-12)
        sim.close()

    def test_mass_and_restitution_mapping(self):
        sim = make_sim()
        obj = make_object()
        sim.load_object(obj, Pose(position=[0.5, 0.0, 0.02]))
        sim.set_physics_parameters("box_01", {"mass": 1.0, "restitution": 0.4})
        body_id = sim.model.body("box_01").id
        assert sim.model.body_mass[body_id] == pytest.approx(1.0)
        sim.set_physics_parameters("box_01", {"mass": 0.5})
        assert sim.model.body_mass[body_id] == pytest.approx(0.5)
        with pytest.raises(KeyError):
            sim.set_physics_parameters("box_01", {"warp": 1.0})
        sim.close()


class TestCompliantPusher:
    def test_tip_tracks_target_with_bounded_lag(self):
        """The compliant tip must follow the commanded path with a small,
        physically meaningful lag (reported as EE replay error)."""
        sim = make_sim()
        obj = make_object()
        sim.load_object(obj, Pose(position=[0.5, 0.0, 0.02]))
        result = straight_push(sim, obj, 0.6, settle=0.5)
        lag = result.ee_replay_rmse()
        assert 0.0 < lag < 0.03  # nonzero (compliance) but < 3 cm
        sim.close()

    def test_first_target_teleports_tip(self):
        """The tip must start on the first target, not fly in from the
        model-default position through the scene."""
        sim = make_sim()
        obj = make_object()
        sim.load_object(obj, Pose(position=[0.5, 0.0, 0.02]))
        sim.reset()
        sim.set_ee_pose(Pose(position=[0.3, 0.0, 0.02]))
        tip = sim.get_robot_state().ee_pose.position
        np.testing.assert_allclose(tip, [0.3, 0.0, 0.02], atol=1e-9)
        sim.close()

    def test_object_not_hit_by_teleport(self):
        sim = make_sim()
        obj = make_object()
        sim.load_object(obj, Pose(position=[0.5, 0.0, 0.02]))
        sim.reset()
        sim.set_ee_pose(Pose(position=[0.3, 0.0, 0.02]))
        sim.step(100)
        np.testing.assert_allclose(
            sim.get_object_state("box_01").pose.position[:2], [0.5, 0.0], atol=1e-6
        )
        sim.close()


class TestFrictionIdentifiability:
    def test_friction_changes_displacement_midrange(self):
        sim = make_sim()
        obj = make_object()
        sim.load_object(obj, Pose(position=[0.5, 0.0, 0.02]))
        finals = {
            mu: straight_push(sim, obj, mu).object_trajectory.positions[-1, 0]
            for mu in (0.3, 0.6, 1.0)
        }
        assert finals[0.3] > finals[0.6] > finals[1.0]
        sim.close()


class TestSyntheticRecovery:
    """Phase 3 gate: MuJoCo-generated episode -> MuJoCo grid calibration."""

    @pytest.fixture(scope="class")
    def mj_episode(self):
        gen = SyntheticEpisodeGenerator(
            SyntheticPushConfig(
                episode_id="mj_test",
                true_friction=0.8,
                backend="mujoco",
                sim_dt=0.001,
                hold_time=1.5,
                noise_std_pos=0.0,
            )
        )
        return gen.generate()

    def test_noise_free_recovery_is_exact(self, mj_episode):
        episode, obj = mj_episode
        sim = create_simulator(SIM_CFG)
        cal = PhysicsCalibrator(sim, loss=TrajectoryLoss(), settle_time=0.5)
        res = cal.calibrate(episode, obj, SPACE, GridSearchOptimizer())
        sim.close()
        assert res.optimized_params["friction"] == pytest.approx(0.8, abs=1e-9)
        assert res.final_loss == pytest.approx(0.0, abs=1e-9)

    def test_noisy_recovery_within_band(self):
        gen = SyntheticEpisodeGenerator(
            SyntheticPushConfig(
                episode_id="mj_noisy",
                true_friction=0.8,
                backend="mujoco",
                sim_dt=0.001,
                hold_time=1.5,
                noise_std_pos=0.001,
                seed=0,
            )
        )
        episode, obj = gen.generate()
        sim = create_simulator(SIM_CFG)
        cal = PhysicsCalibrator(sim, loss=TrajectoryLoss(), settle_time=0.5)
        res = cal.calibrate(episode, obj, SPACE, GridSearchOptimizer())
        sim.close()
        assert 0.65 <= res.optimized_params["friction"] <= 0.95
        assert res.improvement["rmse_position_cm"] > 80.0


class TestCrossBackendGuardrail:
    def test_actuation_mismatch_is_flagged_by_ee_replay_error(self):
        """A mock-recorded EE trajectory replayed through the compliant
        MuJoCo pusher must report a large EE replay error — the framework's
        defense against mis-attributing actuation gaps to physics."""
        gen = SyntheticEpisodeGenerator(
            SyntheticPushConfig(
                episode_id="xb", true_friction=0.8, backend="mock",
                sim_dt=0.002, hold_time=1.0,
            )
        )
        episode, obj = gen.generate()
        sim = create_simulator(SIM_CFG)
        cal = PhysicsCalibrator(sim, loss=TrajectoryLoss(), settle_time=0.5)
        cal._setup_scene(episode, obj)
        rollout = cal.simulate(episode, obj, {"friction": 0.8})
        ee_err = rollout.ee_replay_rmse()
        sim.close()
        assert ee_err > 0.005  # >> same-backend kinematic replay (~1e-17)
