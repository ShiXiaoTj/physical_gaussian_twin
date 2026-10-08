"""Real-Sim time alignment.

Real captures (e.g. 30 Hz mocap) and simulator rollouts (e.g. 500 Hz)
never share a timeline.  Before any loss is computed, both trajectories
are resampled onto a **common uniform timeline** covering their temporal
overlap (linear position interpolation + quaternion SLERP).

Optional DTW alignment is reserved for Phase 3+ (non-rigid time warps,
e.g. trigger latency); the Phase 1 contract is: episodes are hardware-
synchronized, so a shared clock + resampling is sufficient.
"""

from __future__ import annotations

import numpy as np

from pgtwin.real_data.pose_utils import PoseTrajectory, make_timestamps


def common_timeline(
    a: PoseTrajectory, b: PoseTrajectory, rate: float
) -> np.ndarray:
    """Uniform timeline over the temporal overlap of ``a`` and ``b``."""
    t0 = max(a.start_time, b.start_time)
    t1 = min(a.end_time, b.end_time)
    if t1 <= t0:
        raise ValueError(
            f"trajectories do not overlap in time: [{a.start_time:.3f}, "
            f"{a.end_time:.3f}] vs [{b.start_time:.3f}, {b.end_time:.3f}]"
        )
    return make_timestamps(t0, t1, rate)


def align_trajectories(
    a: PoseTrajectory,
    b: PoseTrajectory,
    rate: float = 30.0,
) -> tuple[PoseTrajectory, PoseTrajectory, np.ndarray]:
    """Resample both trajectories onto their common timeline at ``rate`` Hz.

    Returns ``(a_aligned, b_aligned, timestamps)`` with identical timestamps,
    ready for element-wise loss computation.
    """
    ts = common_timeline(a, b, rate)
    return a.resample(ts), b.resample(ts), ts


def resample_to(traj: PoseTrajectory, rate: float) -> PoseTrajectory:
    """Resample a single trajectory to a uniform ``rate`` Hz timeline."""
    return traj.resample(make_timestamps(traj.start_time, traj.end_time, rate))


def estimate_time_offset(
    a: PoseTrajectory,
    b: PoseTrajectory,
    max_offset: float = 0.5,
    step: float = 1.0 / 120.0,
    rate: float = 60.0,
) -> float:
    """Estimate a constant clock offset between two trajectories of the
    *same* physical signal by brute-force cross-correlation of positions.

    Returns the offset ``dt`` such that ``b.shift_time(dt)`` best matches
    ``a``.  Useful when real robot and mocap clocks are not synchronized.
    """
    offsets = np.arange(-max_offset, max_offset + 1e-9, step)
    best_dt, best_err = 0.0, np.inf
    for dt in offsets:
        try:
            a2, b2, _ = align_trajectories(a, b.shift_time(dt), rate)
        except ValueError:
            continue
        err = float(np.mean(np.linalg.norm(a2.positions - b2.positions, axis=1)))
        if err < best_err:
            best_err, best_dt = err, float(dt)
    return best_dt


def dtw_align(a: PoseTrajectory, b: PoseTrajectory) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic-time-warping alignment (reserved for Phase 3+).

    Not required for the PoC: pgtwin episodes assume a shared hardware
    clock, so uniform resampling (:func:`align_trajectories`) is used.
    """
    raise NotImplementedError(
        "DTW alignment is a Phase 3+ extension; use align_trajectories() "
        "for clock-synchronized episodes."
    )
