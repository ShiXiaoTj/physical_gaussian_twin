"""Phase 8: mass calibration (LIFT) and hierarchical staging tests."""

import numpy as np
import pytest

from pgtwin.physics.calibrator import HierarchicalCalibrator, PhysicsCalibrator
from pgtwin.physics.estimators import estimate_mass_from_lift
from pgtwin.physics.losses import ForceLoss
from pgtwin.physics.parameter_space import ParameterSpace
from pgtwin.physics.sensitivity import observable_for_action
from pgtwin.real_data.episode import ActionType
from pgtwin.real_data.synthetic import (
    SyntheticEpisodeGenerator,
    SyntheticLiftConfig,
    SyntheticLiftGenerator,
    SyntheticPushConfig,
)
from pgtwin.real_data.trajectory_loader import EpisodeLoader, save_episode
from pgtwin.simulator.mock_adapter import MockSimulator

MASS_SPACE = {"mass": {"min": 0.1, "max": 2.0, "step": 0.05, "initial": 1.0}}
FRICTION_SPACE = {"friction": {"min": 0.05, "max": 1.5, "step": 0.05, "initial": 0.2}}


@pytest.fixture(scope="module")
def lift_episode():
    gen = SyntheticLiftGenerator(
        SyntheticLiftConfig(
            episode_id="lift_test", true_mass=0.42, true_friction=0.8,
            noise_std_force=0.05, seed=0,
        )
    )
    return gen.generate()


class TestObservableDispatch:
    def test_lift_observable_is_force(self):
        assert observable_for_action(ActionType.LIFT) == "ee_force"

    def test_push_observable_is_pose(self):
        assert observable_for_action(ActionType.PUSH) == "object_pose"


class TestGraspMechanics:
    def test_attached_object_follows_ee(self, lift_episode):
        episode, obj = lift_episode
        sim = MockSimulator(dt=0.002)
        sim.load_scene({})
        sim.load_object(obj, episode.initial_object_pose())
        sim.set_ee_pose(episode.ee_trajectory().pose(0))
        sim.attach_object(obj.object_id)
        result = sim.apply_ee_trajectory(
            episode.ee_trajectory(), object_id=obj.object_id
        )
        # object z must track the EE rise
        rise = result.object_trajectory.positions[-1, 2] - \
            result.object_trajectory.positions[0, 2]
        assert rise == pytest.approx(0.10, abs=0.005)
        sim.close()

    def test_hold_force_equals_mg(self, lift_episode):
        episode, obj = lift_episode
        fz = np.array([s.ee_force[2] for s in episode.robot_states])
        # last 0.5 s of the hold: F_z ~ m g (noise 0.05 N averages out)
        assert np.mean(fz[-40:]) == pytest.approx(0.42 * 9.81, abs=0.05)

    def test_no_grasp_no_force(self):
        sim = MockSimulator()
        state = sim.get_robot_state()
        assert state.ee_force is None
        sim.close()

    def test_detach_releases(self, lift_episode):
        episode, obj = lift_episode
        sim = MockSimulator(dt=0.002)
        sim.load_scene({})
        sim.load_object(obj, episode.initial_object_pose())
        sim.attach_object(obj.object_id)
        sim.detach_object(obj.object_id)
        assert sim.get_robot_state().ee_force is None
        sim.close()


class TestDirectMassEstimator:
    def test_recovers_mass_from_hold(self, lift_episode):
        episode, _ = lift_episode
        est = estimate_mass_from_lift(episode)
        assert est.mass_kg == pytest.approx(0.42, abs=0.02)
        assert est.n_samples >= 5

    def test_requires_force_data(self):
        gen = SyntheticEpisodeGenerator(SyntheticPushConfig(episode_id="p"))
        push_ep, _ = gen.generate()
        with pytest.raises(ValueError, match="ee_force"):
            estimate_mass_from_lift(push_ep)

    def test_roundtrip_via_disk(self, lift_episode, tmp_path):
        """Force columns must survive the CSV round-trip."""
        episode, _ = lift_episode
        save_episode(episode, tmp_path / "lift")
        loaded = EpisodeLoader.load(tmp_path / "lift")
        est = estimate_mass_from_lift(loaded)
        assert est.mass_kg == pytest.approx(0.42, abs=0.02)


class TestForceLoss:
    def test_zero_for_identical(self):
        ts = np.linspace(0, 1, 50)
        f = np.column_stack([np.zeros(50), np.zeros(50), np.full(50, 4.0)])
        assert ForceLoss()(ts, f, ts, f) == pytest.approx(0.0)

    def test_mass_scaling_units(self):
        """0.1 kg mass error at rest -> ~0.981 N mean discrepancy."""
        ts = np.linspace(0, 1, 50)
        f1 = np.column_stack([np.zeros(50), np.zeros(50), np.full(50, 0.42 * 9.81)])
        f2 = np.column_stack([np.zeros(50), np.zeros(50), np.full(50, 0.52 * 9.81)])
        assert ForceLoss()(ts, f1, ts, f2) == pytest.approx(0.981, abs=1e-6)

    def test_no_overlap_raises(self):
        f = np.zeros((10, 3))
        with pytest.raises(ValueError, match="overlap"):
            ForceLoss()(np.linspace(0, 1, 10), f, np.linspace(5, 6, 10), f)


class TestMassCalibration:
    def test_grid_recovers_mass(self, lift_episode):
        episode, obj = lift_episode
        sim = MockSimulator(dt=0.002)
        cal = PhysicsCalibrator(sim, settle_time=0.0)
        result = cal.calibrate(
            episode, obj, ParameterSpace.from_config(MASS_SPACE)
        )
        sim.close()
        # grid step 0.05 around truth 0.42 -> 0.40 or 0.45
        assert result.optimized_params["mass"] == pytest.approx(0.42, abs=0.05)
        assert result.final_loss < result.initial_loss * 0.2

    def test_signal_dispatch_is_force(self, lift_episode):
        episode, obj = lift_episode
        sim = MockSimulator(dt=0.002)
        cal = PhysicsCalibrator(sim, settle_time=0.0)
        assert cal._signal_for(episode) == "ee_force"
        result = cal.calibrate(episode, obj, ParameterSpace.from_config(MASS_SPACE))
        assert "force_mae_n" in result.final_metrics
        sim.close()

    def test_lift_trajectory_alone_cannot_identify_mass(self, lift_episode):
        """Physics check: under kinematic grasp the object trajectory is
        mass-independent, so forcing signal='object_pose' must yield a flat
        landscape (max/min loss ratio ~ 1)."""
        episode, obj = lift_episode
        sim = MockSimulator(dt=0.002)
        cal = PhysicsCalibrator(sim, settle_time=0.0, signal="object_pose")
        losses = []
        cal._setup_scene(episode, obj)
        real = episode.object_trajectory()
        for mass in (0.2, 0.6, 1.2, 1.8):
            rollout = cal.simulate(episode, obj, {"mass": mass})
            losses.append(cal.loss(real, rollout.object_trajectory))
        sim.close()
        spread = (max(losses) - min(losses)) / max(max(losses), 1e-12)
        assert spread < 0.05, f"trajectory should be mass-blind, losses={losses}"


class TestHierarchical:
    @pytest.fixture(scope="class")
    def episode_pair(self):
        lift_gen = SyntheticLiftGenerator(SyntheticLiftConfig(
            episode_id="hier_lift", true_mass=0.42, true_friction=0.8,
            noise_std_force=0.05, seed=1,
        ))
        lift_ep, obj = lift_gen.generate()
        push_gen = SyntheticEpisodeGenerator(SyntheticPushConfig(
            episode_id="hier_push", true_mass=0.42, true_friction=0.8,
            noise_std_pos=0.001, seed=1, hold_time=1.5,
        ))
        push_ep, _ = push_gen.generate()
        return lift_ep, push_ep, obj

    def test_two_stage_recovery(self, episode_pair):
        lift_ep, push_ep, obj = episode_pair
        sim = MockSimulator(dt=0.002)
        hc = HierarchicalCalibrator(sim, settle_time=0.5)
        space = ParameterSpace.from_config({**MASS_SPACE, **FRICTION_SPACE})
        results = hc.calibrate_auto([lift_ep, push_ep], obj, space)
        sim.close()

        assert len(results) == 2
        assert results[0].action_type == "lift"
        assert results[1].action_type == "push"
        assert results[0].optimized_params["mass"] == pytest.approx(0.42, abs=0.05)
        assert results[1].optimized_params["friction"] == pytest.approx(0.8, abs=0.06)

    def test_stage_threading(self, episode_pair):
        """Stage 2 must run with stage 1's calibrated mass, not the guess."""
        lift_ep, push_ep, obj = episode_pair
        sim = MockSimulator(dt=0.002)
        hc = HierarchicalCalibrator(sim, settle_time=0.5)
        space = ParameterSpace.from_config({**MASS_SPACE, **FRICTION_SPACE})
        results = hc.calibrate_auto([lift_ep, push_ep], obj, space)
        sim.close()
        stage2_input = results[1].extra["stage_input_physics"]
        assert stage2_input["mass"] == pytest.approx(
            results[0].optimized_params["mass"]
        )

    def test_stage_order_lift_before_push(self, episode_pair):
        """suggest_stages must schedule LIFT (mass) before PUSH (friction)
        regardless of the episode list order."""
        lift_ep, push_ep, obj = episode_pair
        sim = MockSimulator(dt=0.002)
        hc = HierarchicalCalibrator(sim, settle_time=0.5)
        space = ParameterSpace.from_config({**MASS_SPACE, **FRICTION_SPACE})
        results = hc.calibrate_auto([push_ep, lift_ep], obj, space)  # reversed
        sim.close()
        assert [r.action_type for r in results] == ["lift", "push"]
