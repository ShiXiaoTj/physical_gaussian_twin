"""Timestamped SE(3) trajectories.

:class:`PoseTrajectory` is the central exchange type between real data,
simulator replay, time alignment, losses and metrics.  Positions are in
meters, timestamps in seconds, quaternions in xyzw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pgtwin.common.se3 import (
    Pose,
    ensure_quat_continuity,
    quat_identity,
    quat_slerp,
)


@dataclass
class PoseTrajectory:
    """Time-indexed sequence of poses (single rigid body or EE)."""

    timestamps: np.ndarray  # (N,) seconds, strictly increasing
    positions: np.ndarray  # (N, 3) meters
    quaternions: np.ndarray  # (N, 4) xyzw

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=float).reshape(-1)
        n = len(self.timestamps)
        self.positions = np.asarray(self.positions, dtype=float).reshape(n, 3)
        self.quaternions = np.asarray(self.quaternions, dtype=float).reshape(n, 4)
        norms = np.linalg.norm(self.quaternions, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        self.quaternions = ensure_quat_continuity(self.quaternions / norms)
        if n > 1 and np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing")

    # -- constructors --------------------------------------------------------
    @classmethod
    def from_poses(cls, timestamps: np.ndarray, poses: list[Pose]) -> "PoseTrajectory":
        return cls(
            timestamps=np.asarray(timestamps, dtype=float),
            positions=np.stack([p.position for p in poses])
            if poses
            else np.zeros((0, 3)),
            quaternions=np.stack([p.quaternion for p in poses])
            if poses
            else np.zeros((0, 4)),
        )

    @classmethod
    def static(cls, timestamps: np.ndarray, pose: Pose) -> "PoseTrajectory":
        ts = np.asarray(timestamps, dtype=float)
        n = len(ts)
        return cls(ts, np.tile(pose.position, (n, 1)), np.tile(pose.quaternion, (n, 1)))

    # -- basic accessors -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def start_time(self) -> float:
        return float(self.timestamps[0])

    @property
    def end_time(self) -> float:
        return float(self.timestamps[-1])

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def pose(self, index: int) -> Pose:
        return Pose(self.positions[index], self.quaternions[index])

    def poses(self) -> list[Pose]:
        return [self.pose(i) for i in range(len(self))]

    # -- interpolation -------------------------------------------------------
    def pose_at(self, t: float) -> Pose:
        """Interpolated pose at time ``t`` (linear pos + SLERP; clamped)."""
        ts = self.timestamps
        if len(ts) == 0:
            raise ValueError("empty trajectory")
        if t <= ts[0]:
            return self.pose(0)
        if t >= ts[-1]:
            return self.pose(len(ts) - 1)
        i = int(np.searchsorted(ts, t, side="right")) - 1
        alpha = (t - ts[i]) / (ts[i + 1] - ts[i])
        return Pose(
            position=(1 - alpha) * self.positions[i] + alpha * self.positions[i + 1],
            quaternion=quat_slerp(self.quaternions[i], self.quaternions[i + 1], alpha),
        )

    def resample(self, new_timestamps: np.ndarray) -> "PoseTrajectory":
        """Resample onto ``new_timestamps`` (clamped at both ends)."""
        new_ts = np.asarray(new_timestamps, dtype=float)
        poses = [self.pose_at(t) for t in new_ts]
        return PoseTrajectory.from_poses(new_ts, poses)

    # -- derived quantities ---------------------------------------------------
    def linear_velocities(self) -> np.ndarray:
        """(N, 3) finite-difference linear velocities (m/s), central where possible."""
        if len(self) < 2:
            return np.zeros((len(self), 3))
        return np.gradient(self.positions, self.timestamps, axis=0)

    def path_length(self) -> float:
        if len(self) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(self.positions, axis=0), axis=1)))

    def total_displacement(self) -> float:
        if len(self) < 2:
            return 0.0
        return float(np.linalg.norm(self.positions[-1] - self.positions[0]))

    # -- transformations ------------------------------------------------------
    def shift_time(self, offset: float) -> "PoseTrajectory":
        return PoseTrajectory(
            self.timestamps + offset, self.positions.copy(), self.quaternions.copy()
        )

    def normalize_time(self) -> "PoseTrajectory":
        """Shift timestamps so the trajectory starts at t=0."""
        return self.shift_time(-self.start_time)

    def slice_time(self, t0: float, t1: float) -> "PoseTrajectory":
        mask = (self.timestamps >= t0) & (self.timestamps <= t1)
        return PoseTrajectory(
            self.timestamps[mask], self.positions[mask], self.quaternions[mask]
        )

    def transform(self, T: Pose) -> "PoseTrajectory":
        """Apply frame change ``T`` (e.g. ``T_world_sim``) to every pose."""
        poses = [T.compose(p) for p in self.poses()]
        return PoseTrajectory.from_poses(self.timestamps.copy(), poses)

    def copy(self) -> "PoseTrajectory":
        return PoseTrajectory(
            self.timestamps.copy(), self.positions.copy(), self.quaternions.copy()
        )


def make_timestamps(t0: float, t1: float, rate: float) -> np.ndarray:
    """Uniform timeline [t0, t1] at ``rate`` Hz (inclusive of both ends)."""
    n = max(2, int(np.floor((t1 - t0) * rate)) + 1)
    return t0 + np.arange(n) / rate


def identity_quats(n: int) -> np.ndarray:
    return np.tile(quat_identity(), (n, 1))
