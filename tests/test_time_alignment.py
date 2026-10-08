"""Trajectory interpolation and real-sim time alignment tests."""

import numpy as np
import pytest

from pgtwin.common.se3 import Pose, yaw_to_quat
from pgtwin.real_data.pose_utils import PoseTrajectory, make_timestamps
from pgtwin.real_data.time_alignment import (
    align_trajectories,
    common_timeline,
    dtw_align,
    resample_to,
)


def linear_traj(t0: float, t1: float, rate: float, speed: float = 1.0) -> PoseTrajectory:
    ts = make_timestamps(t0, t1, rate)
    positions = np.column_stack([speed * ts, np.zeros_like(ts), np.zeros_like(ts)])
    quats = np.stack([yaw_to_quat(0.1 * t) for t in ts])
    return PoseTrajectory(ts, positions, quats)


class TestPoseTrajectory:
    def test_interpolation_linear_position(self):
        traj = linear_traj(0.0, 2.0, 10.0)
        p = traj.pose_at(0.55)
        assert p.position[0] == pytest.approx(0.55, abs=1e-9)

    def test_interpolation_slerp_rotation(self):
        traj = linear_traj(0.0, 2.0, 10.0)
        assert traj.pose_at(1.25).yaw == pytest.approx(0.125, abs=1e-9)

    def test_clamping_outside_range(self):
        traj = linear_traj(0.0, 1.0, 10.0)
        assert traj.pose_at(-5.0).position[0] == pytest.approx(0.0)
        assert traj.pose_at(99.0).position[0] == pytest.approx(traj.positions[-1, 0])

    def test_resample_preserves_signal(self):
        traj = linear_traj(0.0, 2.0, 100.0)
        low = traj.resample(make_timestamps(0.0, 2.0, 7.0))
        for t, p in zip(low.timestamps, low.positions):
            assert p[0] == pytest.approx(t, abs=1e-9)

    def test_velocities(self):
        traj = linear_traj(0.0, 2.0, 50.0, speed=0.3)
        v = traj.linear_velocities()
        np.testing.assert_allclose(v[:, 0], 0.3, atol=1e-6)

    def test_rejects_non_monotonic_timestamps(self):
        with pytest.raises(ValueError):
            PoseTrajectory(
                np.array([0.0, 0.1, 0.1]),
                np.zeros((3, 3)),
                np.tile([0, 0, 0, 1.0], (3, 1)),
            )

    def test_transform_frame_change(self):
        traj = linear_traj(0.0, 1.0, 10.0)
        T = Pose.from_xyz_yaw(0.0, 1.0, 0.0, np.pi / 2)
        out = traj.transform(T)
        # x-axis motion becomes y-axis motion under 90 deg yaw
        np.testing.assert_allclose(out.positions[:, 1], 1.0 + traj.positions[:, 0], atol=1e-9)


class TestTimeAlignment:
    def test_common_timeline_overlap(self):
        a = linear_traj(0.0, 2.0, 30.0)
        b = linear_traj(0.5, 3.0, 100.0)
        ts = common_timeline(a, b, 30.0)
        assert ts[0] >= 0.5 - 1e-9
        assert ts[-1] <= 2.0 + 1e-9

    def test_no_overlap_raises(self):
        a = linear_traj(0.0, 1.0, 30.0)
        b = linear_traj(2.0, 3.0, 30.0)
        with pytest.raises(ValueError, match="overlap"):
            common_timeline(a, b, 30.0)

    def test_align_same_signal_different_rates(self):
        """A 30 Hz and a 500 Hz sampling of the same motion must align to
        near-zero discrepancy — the core real-vs-sim rate mismatch case."""
        a = linear_traj(0.0, 2.0, 30.0, speed=0.4)
        b = linear_traj(0.0, 2.0, 500.0, speed=0.4)
        a2, b2, ts = align_trajectories(a, b, 60.0)
        assert len(a2) == len(b2) == len(ts)
        np.testing.assert_allclose(a2.positions, b2.positions, atol=1e-6)

    def test_resample_to_rate(self):
        traj = linear_traj(0.0, 1.0, 100.0)
        out = resample_to(traj, 10.0)
        dt = np.diff(out.timestamps)
        np.testing.assert_allclose(dt, 0.1, atol=1e-9)

    def test_dtw_reserved(self):
        a = linear_traj(0.0, 1.0, 10.0)
        with pytest.raises(NotImplementedError):
            dtw_align(a, a)
