"""Real-Sim trajectory discrepancy losses.

All losses operate on two :class:`PoseTrajectory` objects that have already
been resampled onto a common timeline (see
:mod:`pgtwin.real_data.time_alignment`).  Each loss documents its unit,
because mixing meters with radians silently reweights the objective.

Total objective:

    L = w_pos * L_pos + w_rot * L_rot + w_vel * L_vel + w_contact * L_contact

Phase 1 defaults: ``w_pos = 1.0``, all others ``0.0``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from pgtwin.common.se3 import quat_angular_distance
from pgtwin.real_data.pose_utils import PoseTrajectory
from pgtwin.real_data.time_alignment import align_trajectories


class BaseLoss(ABC):
    """A scalar discrepancy between a real and a simulated trajectory."""

    name: str = "loss"
    unit: str = ""

    @abstractmethod
    def __call__(self, real: PoseTrajectory, sim: PoseTrajectory) -> float:
        """Return the scalar loss (assumes aligned timestamps)."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(unit={self.unit})"


def _check_aligned(real: PoseTrajectory, sim: PoseTrajectory) -> None:
    if len(real) != len(sim):
        raise ValueError(
            f"trajectories not aligned: {len(real)} vs {len(sim)} samples; "
            "call align_trajectories() first"
        )
    if len(real) == 0:
        raise ValueError("empty trajectories")


class PositionLoss(BaseLoss):
    """Mean Euclidean position error. Unit: **meters**."""

    name = "position"
    unit = "m"

    def __call__(self, real: PoseTrajectory, sim: PoseTrajectory) -> float:
        _check_aligned(real, sim)
        return float(np.mean(np.linalg.norm(real.positions - sim.positions, axis=1)))


class RotationLoss(BaseLoss):
    """Mean quaternion geodesic distance. Unit: **radians**."""

    name = "rotation"
    unit = "rad"

    def __call__(self, real: PoseTrajectory, sim: PoseTrajectory) -> float:
        _check_aligned(real, sim)
        errs = [
            quat_angular_distance(real.quaternions[i], sim.quaternions[i])
            for i in range(len(real))
        ]
        return float(np.mean(errs))


class VelocityLoss(BaseLoss):
    """Mean linear-velocity error from finite differences. Unit: **m/s**."""

    name = "velocity"
    unit = "m/s"

    def __call__(self, real: PoseTrajectory, sim: PoseTrajectory) -> float:
        _check_aligned(real, sim)
        dv = real.linear_velocities() - sim.linear_velocities()
        return float(np.mean(np.linalg.norm(dv, axis=1)))


class ContactLoss(BaseLoss):
    """Discrepancy in *motion onset* time — a contact-timing proxy.

    Compares the first time each object moves beyond ``threshold`` from its
    start pose.  A real object that starts sliding later than the simulated
    one indicates wrong static friction / contact stiffness.  Unit: **seconds**.
    """

    name = "contact"
    unit = "s"

    def __init__(self, threshold: float = 2e-3):
        self.threshold = threshold

    def _onset(self, traj: PoseTrajectory) -> float:
        disp = np.linalg.norm(traj.positions - traj.positions[0], axis=1)
        moved = np.nonzero(disp > self.threshold)[0]
        return float(traj.timestamps[moved[0]]) if len(moved) else float(traj.end_time)

    def __call__(self, real: PoseTrajectory, sim: PoseTrajectory) -> float:
        _check_aligned(real, sim)
        return abs(self._onset(real) - self._onset(sim))


class ForceLoss:
    """Mean |F| discrepancy between measured and simulated EE forces.

    The LIFT observable (physics.sensitivity.ACTION_OBSERVABLE): under a
    kinematic grasp the trajectory is imposed, so mass shows up only in the
    grip force ``F = m (a + g_up)``.  Operates on force *time series*
    (timestamps + (N,3) vectors), aligned by linear interpolation onto the
    overlap window.  Unit: **newtons**.
    """

    name = "force"
    unit = "N"

    def __init__(self, resample_rate: float = 30.0, component: str = "norm"):
        if component not in ("norm", "z"):
            raise ValueError("component must be 'norm' or 'z'")
        self.resample_rate = resample_rate
        self.component = component

    def _signal(self, forces: np.ndarray) -> np.ndarray:
        forces = np.atleast_2d(np.asarray(forces, dtype=float))
        if self.component == "z":
            return forces[:, 2]
        return np.linalg.norm(forces, axis=1)

    def __call__(
        self,
        real_ts: np.ndarray,
        real_forces: np.ndarray,
        sim_ts: np.ndarray,
        sim_forces: np.ndarray,
    ) -> float:
        t0 = max(float(real_ts[0]), float(sim_ts[0]))
        t1 = min(float(real_ts[-1]), float(sim_ts[-1]))
        if t1 <= t0:
            raise ValueError("force series do not overlap in time")
        n = max(2, int((t1 - t0) * self.resample_rate) + 1)
        ts = np.linspace(t0, t1, n)
        real_i = np.interp(ts, real_ts, self._signal(real_forces))
        sim_i = np.interp(ts, sim_ts, self._signal(sim_forces))
        return float(np.mean(np.abs(real_i - sim_i)))


@dataclass
class LossWeights:
    position: float = 1.0
    rotation: float = 0.0
    velocity: float = 0.0
    contact: float = 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> "LossWeights":
        return cls(
            position=float(cfg.get("position_weight", 1.0)),
            rotation=float(cfg.get("rotation_weight", 0.0)),
            velocity=float(cfg.get("velocity_weight", 0.0)),
            contact=float(cfg.get("contact_weight", 0.0)),
        )

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "rotation": self.rotation,
            "velocity": self.velocity,
            "contact": self.contact,
        }


class TrajectoryLoss:
    """Weighted sum of component losses, with per-term breakdown.

    Handles alignment internally: pass raw real/sim trajectories with
    arbitrary sampling rates and they are resampled onto their common
    timeline at ``resample_rate`` Hz before evaluation.
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        resample_rate: float = 30.0,
        contact_threshold: float = 2e-3,
    ):
        self.weights = weights or LossWeights()
        self.resample_rate = resample_rate
        self.components: dict[str, BaseLoss] = {
            "position": PositionLoss(),
            "rotation": RotationLoss(),
            "velocity": VelocityLoss(),
            "contact": ContactLoss(contact_threshold),
        }

    @property
    def active_terms(self) -> list[str]:
        return [
            k for k in self.components if getattr(self.weights, k) != 0.0
        ]

    def breakdown(
        self, real: PoseTrajectory, sim: PoseTrajectory, align: bool = True
    ) -> dict[str, float]:
        """Per-term losses in native units plus the weighted ``total``."""
        if align:
            real, sim, _ = align_trajectories(real, sim, self.resample_rate)
        out: dict[str, float] = {}
        total = 0.0
        for key, loss in self.components.items():
            weight = getattr(self.weights, key)
            if weight == 0.0:
                continue
            value = loss(real, sim)
            out[key] = value
            total += weight * value
        out["total"] = total
        return out

    def __call__(
        self, real: PoseTrajectory, sim: PoseTrajectory, align: bool = True
    ) -> float:
        return self.breakdown(real, sim, align=align)["total"]

    @classmethod
    def from_config(cls, loss_cfg: dict, resample_rate: float = 30.0) -> "TrajectoryLoss":
        return cls(
            weights=LossWeights.from_config(loss_cfg),
            resample_rate=resample_rate,
            contact_threshold=float(loss_cfg.get("contact_threshold", 2e-3)),
        )
