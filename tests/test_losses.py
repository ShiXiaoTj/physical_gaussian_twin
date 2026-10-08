"""Loss function tests: units, zero-cases, weighting."""

import numpy as np
import pytest

from pgtwin.common.se3 import yaw_to_quat
from pgtwin.physics.losses import (
    LossWeights,
    PositionLoss,
    RotationLoss,
    TrajectoryLoss,
    VelocityLoss,
)
from pgtwin.real_data.pose_utils import PoseTrajectory, make_timestamps


def make_traj(offset=0.0, yaw=0.0, n=20) -> PoseTrajectory:
    ts = np.linspace(0.0, 1.0, n)
    positions = np.column_stack([ts + offset, np.zeros(n), np.zeros(n)])
    quats = np.tile(yaw_to_quat(yaw), (n, 1))
    return PoseTrajectory(ts, positions, quats)


class TestPositionLoss:
    def test_zero_for_identical(self):
        t = make_traj()
        assert PositionLoss()(t, t.copy()) == pytest.approx(0.0)

    def test_constant_offset_in_meters(self):
        assert PositionLoss()(make_traj(), make_traj(offset=0.05)) == pytest.approx(0.05)

    def test_misaligned_raises(self):
        with pytest.raises(ValueError, match="not aligned"):
            PositionLoss()(make_traj(n=20), make_traj(n=21))


class TestRotationLoss:
    def test_zero_for_identical(self):
        t = make_traj(yaw=0.4)
        assert RotationLoss()(t, t.copy()) == pytest.approx(0.0, abs=1e-9)

    def test_constant_yaw_error_in_radians(self):
        loss = RotationLoss()(make_traj(yaw=0.0), make_traj(yaw=0.3))
        assert loss == pytest.approx(0.3, abs=1e-9)


class TestVelocityLoss:
    def test_zero_for_same_motion(self):
        assert VelocityLoss()(make_traj(), make_traj(offset=1.0)) == pytest.approx(
            0.0, abs=1e-9
        )


class TestTrajectoryLoss:
    def test_default_weights_position_only(self):
        loss = TrajectoryLoss()
        assert loss.active_terms == ["position"]

    def test_weighted_total(self):
        loss = TrajectoryLoss(LossWeights(position=1.0, rotation=2.0))
        real = make_traj(yaw=0.0)
        sim = make_traj(offset=0.1, yaw=0.2)
        breakdown = loss.breakdown(real, sim, align=False)
        assert breakdown["position"] == pytest.approx(0.1, abs=1e-9)
        assert breakdown["rotation"] == pytest.approx(0.2, abs=1e-9)
        assert breakdown["total"] == pytest.approx(0.1 + 2.0 * 0.2, abs=1e-8)

    def test_internal_alignment_different_rates(self):
        real = make_traj(n=31)
        ts = make_timestamps(0.0, 1.0, 200.0)
        sim = PoseTrajectory(
            ts,
            np.column_stack([ts + 0.02, np.zeros_like(ts), np.zeros_like(ts)]),
            np.tile([0, 0, 0, 1.0], (len(ts), 1)),
        )
        assert TrajectoryLoss()(real, sim) == pytest.approx(0.02, abs=1e-4)
