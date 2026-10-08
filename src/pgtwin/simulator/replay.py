"""Replay real EE trajectories into a simulator.

:class:`PushReplayController` streams a recorded real end-effector
trajectory into the simulator as kinematic position targets and records
both the resulting object trajectory **and the actually simulated EE
trajectory**.

The latter matters: if the simulated EE deviates from the real EE (IK
failure, controller lag, collision), the object trajectory mismatch is an
*action replay* problem, not a physics-parameter problem.  Calibration
must therefore always report the EE replay error alongside the object
error (docs/architecture.md, "Replay fidelity").

Modes:
* ``kinematic`` (Phase 1 default): EE pose is set directly each step.
* ``controller``: reserved for torque/impedance replay on real backends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pgtwin.common.logging import get_logger
from pgtwin.real_data.pose_utils import PoseTrajectory
from pgtwin.simulator.base_simulator import BaseSimulator

logger = get_logger("simulator.replay")


@dataclass
class ReplayResult:
    """Everything recorded during one simulated replay."""

    object_trajectory: PoseTrajectory  # simulated object states
    ee_trajectory: PoseTrajectory  # actually simulated EE states
    ee_target_trajectory: PoseTrajectory  # commanded (real) EE trajectory
    #: (N, 3) grip force at each recorded step, or None when the backend /
    #: episode involves no grasp (LIFT observable; see physics.sensitivity).
    ee_forces: np.ndarray | None = None

    def ee_replay_rmse(self) -> float:
        """RMSE (m) between commanded and simulated EE positions.

        Near zero for kinematic replay; meaningful for controller replay.
        """
        target = self.ee_target_trajectory.resample(self.ee_trajectory.timestamps)
        err = np.linalg.norm(target.positions - self.ee_trajectory.positions, axis=1)
        return float(np.sqrt(np.mean(err**2)))


class PushReplayController:
    """Replay an EE trajectory through :class:`BaseSimulator`."""

    def __init__(self, simulator: BaseSimulator, mode: str = "kinematic"):
        if mode not in ("kinematic", "controller"):
            raise ValueError(f"unknown replay mode '{mode}'")
        if mode == "controller":
            raise NotImplementedError(
                "controller replay is reserved for physical-controller backends"
            )
        self.sim = simulator
        self.mode = mode

    def replay(
        self,
        ee_trajectory: PoseTrajectory,
        object_id: str,
        settle_time: float = 0.0,
    ) -> ReplayResult:
        """Drive the simulator along ``ee_trajectory`` and record states.

        The simulator must already be reset with the object placed at its
        initial pose.  ``settle_time`` extends the rollout after the last
        EE sample so free-sliding objects come to rest.
        """
        dt = self.sim.dt
        t0, t1 = ee_trajectory.start_time, ee_trajectory.end_time
        n_steps = int(np.ceil((t1 - t0 + settle_time) / dt))

        times: list[float] = []
        obj_pos: list[np.ndarray] = []
        obj_quat: list[np.ndarray] = []
        ee_pos: list[np.ndarray] = []
        ee_quat: list[np.ndarray] = []
        ee_forces: list[np.ndarray | None] = []

        def record() -> None:
            t = t0 + self.sim.time
            obj_state = self.sim.get_object_state(object_id)
            ee_state = self.sim.get_robot_state()
            times.append(t)
            obj_pos.append(obj_state.pose.position)
            obj_quat.append(obj_state.pose.quaternion)
            ee_pos.append(ee_state.ee_pose.position)
            ee_quat.append(ee_state.ee_pose.quaternion)
            ee_forces.append(ee_state.ee_force)

        # initial sample at t0
        self.sim.set_ee_pose(ee_trajectory.pose_at(t0))
        record()

        for i in range(n_steps):
            t_next = t0 + (i + 1) * dt
            self.sim.set_ee_pose(ee_trajectory.pose_at(min(t_next, t1)))
            self.sim.step()
            record()

        force_arr = None
        if any(f is not None for f in ee_forces):
            force_arr = np.stack(
                [f if f is not None else np.zeros(3) for f in ee_forces]
            )
        result = ReplayResult(
            object_trajectory=PoseTrajectory(
                np.asarray(times), np.stack(obj_pos), np.stack(obj_quat)
            ),
            ee_trajectory=PoseTrajectory(
                np.asarray(times), np.stack(ee_pos), np.stack(ee_quat)
            ),
            ee_target_trajectory=ee_trajectory.copy(),
            ee_forces=force_arr,
        )
        logger.debug(
            "replay done: %d steps, ee replay rmse = %.2e m",
            n_steps,
            result.ee_replay_rmse(),
        )
        return result
