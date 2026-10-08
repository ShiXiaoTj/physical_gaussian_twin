"""MockSimulator behaviour: determinism, friction monotonicity, contract.

The friction-monotonicity test is the physical basis of the whole PoC:
a push episode identifies friction only if higher friction -> shorter
post-release slide, strictly.
"""

import numpy as np
import pytest

from pgtwin.common.se3 import Pose
from pgtwin.physics.physical_object import PhysicalObject, PhysicsParameters
from pgtwin.simulator import create_simulator
from pgtwin.simulator.mock_adapter import MockSimulator


def make_object(friction: float = 0.5) -> PhysicalObject:
    return PhysicalObject.from_config(
        {
            "object_id": "box_01",
            "shape": "box",
            "size": [0.06, 0.06, 0.06],
            "physics": {"mass": 0.42, "friction_dynamic": friction},
        }
    )


def push_sim(friction: float, speed: float = 0.25, dt: float = 0.002):
    """Straight +x push then 1.5 s free slide; returns (final_x, ReplayResult)."""
    sim = MockSimulator(dt=dt)
    sim.load_scene({"table_height": 0.0})
    sim.load_robot({"name": "franka", "ee_radius": 0.012})
    sim.load_object(make_object(friction))
    sim.set_object_pose("box_01", Pose(position=[0.5, 0.0, 0.03]))

    ts = np.arange(0.0, 2.0 + dt, dt)
    ee = [Pose(position=[0.40 + speed * t, 0.0, 0.03]) for t in ts]
    result = sim.apply_ee_trajectory(ts, ee, object_id="box_01", settle_time=1.5)
    sim.close()
    return float(result.object_trajectory.positions[-1, 0]), result


class TestSimulatorContract:
    def test_factory_defaults_to_mock(self):
        sim = create_simulator({"backend": "mock", "dt": 0.002})
        assert isinstance(sim, MockSimulator)
        assert sim.dt == pytest.approx(0.002)
        sim.close()

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="unknown simulator backend"):
            create_simulator({"backend": "nope"})

    def test_is_available(self):
        assert MockSimulator.is_available() is True

    def test_constructor_overrides(self):
        sim = MockSimulator(dt=0.01, ee_radius=0.02)
        assert sim.config.dt == pytest.approx(0.01)
        assert sim.config.ee_radius == pytest.approx(0.02)

    def test_unknown_override_rejected(self):
        with pytest.raises(TypeError):
            MockSimulator(unknown_option=1)

    def test_object_state_after_set_pose(self):
        sim = MockSimulator()
        sim.load_scene({})
        sim.load_object(make_object())
        sim.set_object_pose("box_01", Pose(position=[0.5, 0.1, 0.03]))
        state = sim.get_object_state("box_01")
        np.testing.assert_allclose(state.pose.position, [0.5, 0.1, 0.03], atol=1e-12)

    def test_unknown_object_raises(self):
        sim = MockSimulator()
        with pytest.raises(KeyError):
            sim.get_object_state("ghost")

    def test_reset_restores_initial_pose_and_time(self):
        sim = MockSimulator()
        sim.load_scene({})
        sim.load_object(make_object(), Pose(position=[0.5, 0.0, 0.03]))
        sim.set_ee_pose(Pose(position=[0.46, 0.0, 0.03]))
        sim.step(100)
        sim.reset()
        assert sim.time == pytest.approx(0.0)
        np.testing.assert_allclose(
            sim.get_object_state("box_01").pose.position, [0.5, 0.0, 0.03], atol=1e-12
        )


class TestPushPhysics:
    def test_object_moves_along_push_direction(self):
        x_final, _ = push_sim(0.5)
        assert x_final > 0.5  # started at x = 0.5

    def test_higher_friction_moves_object_less(self):
        """Strictly monotone friction -> displacement response: the
        identifiability requirement for friction-from-push."""
        xs = [push_sim(mu)[0] for mu in (0.2, 0.5, 0.9, 1.3)]
        assert all(a > b for a, b in zip(xs, xs[1:])), xs

    def test_deterministic_across_runs(self):
        a, _ = push_sim(0.7)
        b, _ = push_sim(0.7)
        assert a == pytest.approx(b, abs=1e-12)

    def test_ee_replay_error_is_small(self):
        """Kinematic replay must track the commanded EE path, otherwise
        object mismatch cannot be attributed to physics."""
        _, result = push_sim(0.6)
        assert result.ee_replay_rmse() < 1e-6

    def test_object_stays_on_table(self):
        _, result = push_sim(0.6)
        z = result.object_trajectory.positions[:, 2]
        np.testing.assert_allclose(z, z[0], atol=1e-9)

    def test_object_untouched_when_ee_far_away(self):
        sim = MockSimulator()
        sim.load_scene({})
        sim.load_robot({"name": "franka"})
        sim.load_object(make_object())
        sim.set_object_pose("box_01", Pose(position=[0.5, 0.0, 0.03]))
        ts = np.arange(0.0, 1.0, 0.002)
        ee = [Pose(position=[0.0, 0.5, 0.3]) for _ in ts]
        result = sim.apply_ee_trajectory(ts, ee, object_id="box_01")
        np.testing.assert_allclose(
            result.object_trajectory.positions[-1], [0.5, 0.0, 0.03], atol=1e-9
        )
        sim.close()

    def test_no_contact_when_ee_above_object(self):
        """EE passing over the object (outside the vertical contact band)
        must not move it."""
        sim = MockSimulator()
        sim.load_scene({})
        sim.load_object(make_object())
        sim.set_object_pose("box_01", Pose(position=[0.5, 0.0, 0.03]))
        ts = np.arange(0.0, 1.0, 0.002)
        ee = [Pose(position=[0.40 + 0.25 * t, 0.0, 0.30]) for t in ts]
        result = sim.apply_ee_trajectory(ts, ee, object_id="box_01")
        np.testing.assert_allclose(
            result.object_trajectory.positions[-1], [0.5, 0.0, 0.03], atol=1e-9
        )

    def test_set_physics_parameters_changes_outcome(self):
        finals = []
        for mu in (0.25, 1.0):
            sim = MockSimulator(dt=0.002)
            sim.load_scene({})
            sim.load_robot({"name": "franka"})
            sim.load_object(make_object(0.5))
            sim.set_physics_parameters("box_01", {"friction": mu})
            sim.set_object_pose("box_01", Pose(position=[0.5, 0.0, 0.03]))
            ts = np.arange(0.0, 1.0, 0.002)
            ee = [Pose(position=[0.40 + 0.25 * t, 0.0, 0.03]) for t in ts]
            result = sim.apply_ee_trajectory(
                ts, ee, object_id="box_01", settle_time=1.5
            )
            finals.append(float(result.object_trajectory.positions[-1, 0]))
            sim.close()
        assert finals[0] > finals[1]

    def test_unknown_parameter_rejected(self):
        sim = MockSimulator()
        sim.load_object(make_object())
        with pytest.raises(KeyError):
            sim.set_physics_parameters("box_01", {"warp_factor": 9.0})

    def test_off_center_push_induces_rotation(self):
        sim = MockSimulator(dt=0.002)
        sim.load_scene({})
        sim.load_object(make_object(0.4))
        sim.set_object_pose("box_01", Pose(position=[0.5, 0.0, 0.03]))
        ts = np.arange(0.0, 1.2, 0.002)
        # push line offset 2 cm from the object center
        ee = [Pose(position=[0.40 + 0.2 * t, 0.02, 0.03]) for t in ts]
        result = sim.apply_ee_trajectory(ts, ee, object_id="box_01", settle_time=0.5)
        final_yaw = result.object_trajectory.pose(-1).yaw
        assert abs(final_yaw) > 1e-3


def test_physics_parameters_update_is_immutable():
    p = PhysicsParameters(mass=0.4, friction_dynamic=0.5)
    q = p.update({"friction": 0.9})
    assert p.friction_dynamic == pytest.approx(0.5)
    assert q.friction_dynamic == pytest.approx(0.9)
    assert q.mass == pytest.approx(0.4)
