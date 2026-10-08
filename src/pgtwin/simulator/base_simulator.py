"""Simulator abstraction.

Every physics backend (Mock, MuJoCo, Isaac Sim, ...) implements
:class:`BaseSimulator` so that the calibration pipeline is backend-agnostic.
The simulator world frame is assumed identical to the pgtwin world frame
(z-up, meters); adapters whose native convention differs must convert
internally (``T_sim_world``) and expose world-frame states.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pgtwin.common.se3 import Pose
from pgtwin.physics.physical_object import PhysicalObject
from pgtwin.real_data.episode import ObjectState, RobotState


@dataclass
class SimulatorConfig:
    dt: float = 0.002  # physics step (s)
    gravity: float = 9.81  # m/s^2, +z up world, gravity acts along -z
    seed: int = 0
    ee_radius: float = 0.012  # EE tip collision radius (m) for kinematic replay
    table_height: float = 0.0  # z of support surface (m)
    contact_solref: float = 0.01  # contact time constant (s), MJCF solref[0]
    contact_dampratio: float = 2.0  # overdamped contact: suppresses contact-
    # spring surging that makes push rollouts chaotic in friction
    ee_weld_solref: float = 0.04  # compliant-pusher coupling time constant (s)

    @classmethod
    def from_config(cls, cfg: dict) -> "SimulatorConfig":
        return cls(
            dt=float(cfg.get("dt", 0.002)),
            gravity=float(cfg.get("gravity", 9.81)),
            seed=int(cfg.get("seed", 0)),
            ee_radius=float(cfg.get("ee_radius", 0.012)),
            table_height=float(cfg.get("table_height", 0.0)),
            contact_solref=float(cfg.get("contact_solref", 0.01)),
            contact_dampratio=float(cfg.get("contact_dampratio", 2.0)),
            ee_weld_solref=float(cfg.get("ee_weld_solref", 0.04)),
        )


class BaseSimulator(ABC):
    """Minimal simulation interface needed by the calibration loop."""

    def __init__(self, config: SimulatorConfig | None = None, **overrides):
        cfg = config or SimulatorConfig()
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise TypeError(f"unknown simulator option '{key}'")
            setattr(cfg, key, value)
        self.config = cfg

    @classmethod
    def is_available(cls) -> bool:
        """True when the backend's dependencies are installed."""
        return True

    # -- lifecycle -----------------------------------------------------------
    @abstractmethod
    def reset(self) -> None:
        """Restore initial state (poses, velocities, sim time = 0).

        Must be deterministic: two resets followed by identical commands
        must produce identical trajectories (controlled randomness)."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""

    # -- scene setup ---------------------------------------------------------
    @abstractmethod
    def load_scene(self, scene_cfg: dict) -> None:
        """Load static environment (table / support surfaces)."""

    @abstractmethod
    def load_robot(self, robot_cfg: dict) -> None:
        """Load the robot (Phase 1: a kinematic EE proxy is sufficient)."""

    @abstractmethod
    def load_object(self, obj: PhysicalObject, initial_pose: Pose | None = None) -> None:
        """Instantiate a dynamic object from its collision proxy + physics.

        ``initial_pose`` defaults to ``obj.initial_pose``."""

    # -- state / parameter access ---------------------------------------------
    @abstractmethod
    def set_physics_parameters(self, object_id: str, params: dict[str, float]) -> None:
        """Update physics parameters (e.g. ``{"friction": 0.8}``) in place."""

    @abstractmethod
    def set_object_pose(self, object_id: str, pose: Pose) -> None:
        ...

    @abstractmethod
    def set_ee_pose(self, pose: Pose) -> None:
        """Kinematic EE target for the *next* step (position-target replay)."""

    @abstractmethod
    def get_object_state(self, object_id: str) -> ObjectState:
        ...

    @abstractmethod
    def get_robot_state(self) -> RobotState:
        ...

    def set_robot_state(self, state: RobotState) -> None:
        """Default: kinematic adapters only track the EE pose."""
        self.set_ee_pose(state.ee_pose)

    # -- grasping (LIFT support) -------------------------------------------------
    def attach_object(self, object_id: str) -> None:
        """Rigidly attach the object to the EE (kinematic grasp).

        While attached the object follows the EE and ``get_robot_state()``
        reports the grip force ``F = m (a + g_up)`` needed to impose the
        motion — the observable that identifies mass under LIFT.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support grasping (LIFT episodes)"
        )

    def detach_object(self, object_id: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support grasping (LIFT episodes)"
        )

    # -- stepping --------------------------------------------------------------
    @abstractmethod
    def step(self, n: int = 1) -> None:
        """Advance physics by ``n * dt`` seconds."""

    def apply_ee_trajectory(
        self,
        timestamps_or_traj,
        poses: "list[Pose] | None" = None,
        object_id: str | None = None,
        settle_time: float = 0.0,
    ):
        """Kinematically replay an EE trajectory; returns a ``ReplayResult``.

        Accepts either a :class:`PoseTrajectory` or ``(timestamps, poses)``.
        Convenience wrapper over :class:`PushReplayController`.
        """
        from pgtwin.real_data.pose_utils import PoseTrajectory
        from pgtwin.simulator.replay import PushReplayController

        if poses is not None and not isinstance(poses, str):
            traj = PoseTrajectory.from_poses(timestamps_or_traj, poses)
        else:
            traj = timestamps_or_traj
            if object_id is None and isinstance(poses, str):
                object_id = poses
        if object_id is None:
            raise ValueError("object_id is required")
        return PushReplayController(self).replay(traj, object_id, settle_time)

    @property
    def dt(self) -> float:
        return self.config.dt

    @property
    @abstractmethod
    def time(self) -> float:
        """Current simulation time (s since reset)."""
