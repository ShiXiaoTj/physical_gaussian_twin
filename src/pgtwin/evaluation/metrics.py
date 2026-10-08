"""Real-Sim trajectory metrics.

Units are part of every metric name: positions are reported in **cm**,
rotations in **degrees** (internal computation stays in SI).
"""

from __future__ import annotations

import numpy as np

from pgtwin.common.se3 import quat_angular_distance
from pgtwin.real_data.pose_utils import PoseTrajectory
from pgtwin.real_data.time_alignment import align_trajectories

M_TO_CM = 100.0


def _aligned(
    real: PoseTrajectory, sim: PoseTrajectory, rate: float
) -> tuple[PoseTrajectory, PoseTrajectory]:
    real2, sim2, _ = align_trajectories(real, sim, rate)
    return real2, sim2


def position_errors_m(real: PoseTrajectory, sim: PoseTrajectory) -> np.ndarray:
    """Per-sample Euclidean position errors (m); assumes aligned inputs."""
    return np.linalg.norm(real.positions - sim.positions, axis=1)


def rotation_errors_rad(real: PoseTrajectory, sim: PoseTrajectory) -> np.ndarray:
    return np.array(
        [
            quat_angular_distance(real.quaternions[i], sim.quaternions[i])
            for i in range(len(real))
        ]
    )


def ate_position_cm(real: PoseTrajectory, sim: PoseTrajectory) -> float:
    """Absolute trajectory error: mean position error (cm)."""
    return float(np.mean(position_errors_m(real, sim))) * M_TO_CM


def rmse_position_cm(real: PoseTrajectory, sim: PoseTrajectory) -> float:
    """Root-mean-square position error (cm)."""
    return float(np.sqrt(np.mean(position_errors_m(real, sim) ** 2))) * M_TO_CM


def rotation_error_deg(real: PoseTrajectory, sim: PoseTrajectory) -> float:
    """Mean geodesic rotation error (degrees)."""
    return float(np.degrees(np.mean(rotation_errors_rad(real, sim))))


def final_displacement_error_cm(real: PoseTrajectory, sim: PoseTrajectory) -> float:
    """Position error at the final common timestamp (cm)."""
    return float(np.linalg.norm(real.positions[-1] - sim.positions[-1])) * M_TO_CM


def trajectory_metrics(
    real: PoseTrajectory,
    sim: PoseTrajectory,
    resample_rate: float = 30.0,
    align: bool = True,
) -> dict[str, float]:
    """Standard metric bundle between a real and a simulated trajectory."""
    if align:
        real, sim = _aligned(real, sim, resample_rate)
    return {
        "ate_position_cm": ate_position_cm(real, sim),
        "rmse_position_cm": rmse_position_cm(real, sim),
        "rotation_error_deg": rotation_error_deg(real, sim),
        "final_displacement_error_cm": final_displacement_error_cm(real, sim),
        "max_position_error_cm": float(np.max(position_errors_m(real, sim))) * M_TO_CM,
    }


def improvement_percent(
    initial_metrics: dict[str, float], final_metrics: dict[str, float]
) -> dict[str, float]:
    """Relative improvement per metric: ``(initial - final) / initial * 100``.

    Positive = calibration reduced the error. Metrics with ~zero initial
    value are reported as 0 to avoid division blow-ups.
    """
    out: dict[str, float] = {}
    for key, initial in initial_metrics.items():
        final = final_metrics.get(key)
        if final is None:
            continue
        if abs(initial) < 1e-12:
            out[key] = 0.0
        else:
            out[key] = float((initial - final) / initial * 100.0)
    return out
