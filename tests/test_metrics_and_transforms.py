"""Metrics units and FrameRegistry transform-graph tests."""

import numpy as np
import pytest

from pgtwin.common.se3 import Pose, yaw_to_quat
from pgtwin.common.transforms import FrameRegistry
from pgtwin.evaluation.metrics import (
    ate_position_cm,
    final_displacement_error_cm,
    improvement_percent,
    rmse_position_cm,
    rotation_error_deg,
    trajectory_metrics,
)
from pgtwin.real_data.pose_utils import PoseTrajectory


def traj(offset=0.0, yaw=0.0, n=10) -> PoseTrajectory:
    ts = np.linspace(0, 1, n)
    pos = np.column_stack([ts + offset, np.zeros(n), np.zeros(n)])
    return PoseTrajectory(ts, pos, np.tile(yaw_to_quat(yaw), (n, 1)))


class TestMetricUnits:
    def test_ate_reports_cm(self):
        assert ate_position_cm(traj(), traj(offset=0.05)) == pytest.approx(5.0)

    def test_rmse_reports_cm(self):
        assert rmse_position_cm(traj(), traj(offset=0.02)) == pytest.approx(2.0)

    def test_rotation_reports_degrees(self):
        assert rotation_error_deg(traj(), traj(yaw=np.pi / 6)) == pytest.approx(30.0)

    def test_final_displacement_cm(self):
        assert final_displacement_error_cm(traj(), traj(offset=0.1)) == pytest.approx(10.0)

    def test_bundle_keys(self):
        m = trajectory_metrics(traj(), traj(offset=0.01))
        assert {"ate_position_cm", "rmse_position_cm", "rotation_error_deg",
                "final_displacement_error_cm", "max_position_error_cm"} <= set(m)

    def test_improvement_percent(self):
        imp = improvement_percent({"rmse_position_cm": 8.0}, {"rmse_position_cm": 2.0})
        assert imp["rmse_position_cm"] == pytest.approx(75.0)

    def test_improvement_zero_initial_guarded(self):
        imp = improvement_percent({"x": 0.0}, {"x": 1.0})
        assert imp["x"] == 0.0


class TestFrameRegistry:
    def test_direct_and_inverse(self):
        reg = FrameRegistry()
        T = Pose.from_xyz_yaw(1.0, 0.0, 0.5, np.pi / 2)
        reg.add("world", "robot", T)
        got = reg.get("world", "robot")
        assert got.position_distance(T) < 1e-12
        inv = reg.get("robot", "world")
        assert inv.position_distance(T.inverse()) < 1e-12

    def test_chained_resolution(self):
        """world -> robot -> ee resolved through the graph must equal the
        direct composition (the classic T_world_ee chain)."""
        reg = FrameRegistry()
        T_wr = Pose.from_xyz_yaw(0.2, 0.0, 0.0, 0.3)
        T_re = Pose.from_xyz_yaw(0.0, 0.4, 0.3, -0.8)
        reg.add("world", "robot", T_wr)
        reg.add("robot", "ee", T_re)
        expected = T_wr.compose(T_re)
        got = reg.get("world", "ee")
        assert got.position_distance(expected) < 1e-10
        assert got.rotation_distance(expected) < 1e-9

    def test_identity_same_frame(self):
        reg = FrameRegistry()
        ident = reg.get("world", "world")
        assert np.linalg.norm(ident.position) == 0.0

    def test_missing_path_raises(self):
        reg = FrameRegistry()
        reg.add("world", "robot", Pose.identity())
        with pytest.raises(KeyError):
            reg.get("world", "camera")
