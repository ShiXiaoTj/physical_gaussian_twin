"""Interaction episode data model.

An :class:`InteractionEpisode` is one real-world robot-object interaction:
synchronized robot states and object states plus metadata.  This is the
"S4: real robot interaction" record that drives calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from pgtwin.common.se3 import Pose
from pgtwin.real_data.pose_utils import PoseTrajectory


class ActionType(str, Enum):
    """Interaction primitives. Each excites different physical parameters
    (see docs/architecture.md, "Action -> Parameter Sensitivity")."""

    PUSH = "push"
    LIFT = "lift"
    SLIDE = "slide"
    COLLISION = "collision"
    PRESS = "press"

    @classmethod
    def parse(cls, value: "str | ActionType") -> "ActionType":
        if isinstance(value, ActionType):
            return value
        return cls(value.lower())


@dataclass
class RobotState:
    """Robot state sample. ``ee_pose`` is ``T_world_ee``."""

    timestamp: float
    ee_pose: Pose
    joint_position: np.ndarray | None = None
    joint_velocity: np.ndarray | None = None
    gripper_state: float = 0.0
    ee_force: np.ndarray | None = None  # (3,) N, world frame
    ee_torque: np.ndarray | None = None  # (3,) Nm, world frame


@dataclass
class ObjectState:
    """Object state sample. ``pose`` is ``T_world_object``."""

    timestamp: float
    pose: Pose
    linear_velocity: np.ndarray | None = None  # (3,) m/s
    angular_velocity: np.ndarray | None = None  # (3,) rad/s


@dataclass
class EpisodeMetadata:
    episode_id: str
    action_type: ActionType
    robot: str = "franka"
    object_id: str = "object_0"
    scene_id: str = "scene_0"
    world_frame: str = "world"
    sampling_rate: float = 30.0
    start_time: float = 0.0
    end_time: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "action_type": self.action_type.value,
            "robot": self.robot,
            "object_id": self.object_id,
            "scene_id": self.scene_id,
            "world_frame": self.world_frame,
            "sampling_rate": self.sampling_rate,
            "start_time": self.start_time,
            "end_time": self.end_time,
            **({"extra": self.extra} if self.extra else {}),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeMetadata":
        known = {
            "episode_id",
            "action_type",
            "robot",
            "object_id",
            "scene_id",
            "world_frame",
            "sampling_rate",
            "start_time",
            "end_time",
        }
        extra = dict(data.get("extra", {}))
        extra.update({k: v for k, v in data.items() if k not in known | {"extra"}})
        return cls(
            episode_id=data["episode_id"],
            action_type=ActionType.parse(data["action_type"]),
            robot=data.get("robot", "franka"),
            object_id=data.get("object_id", "object_0"),
            scene_id=data.get("scene_id", "scene_0"),
            world_frame=data.get("world_frame", "world"),
            sampling_rate=float(data.get("sampling_rate", 30.0)),
            start_time=float(data.get("start_time", 0.0)),
            end_time=float(data.get("end_time", 0.0)),
            extra=extra,
        )


@dataclass
class InteractionEpisode:
    """One recorded robot-object interaction (all poses in world frame)."""

    metadata: EpisodeMetadata
    robot_states: list[RobotState]
    object_states: list[ObjectState]

    # -- trajectory views -----------------------------------------------------
    def ee_trajectory(self) -> PoseTrajectory:
        return PoseTrajectory.from_poses(
            np.array([s.timestamp for s in self.robot_states]),
            [s.ee_pose for s in self.robot_states],
        )

    def object_trajectory(self) -> PoseTrajectory:
        return PoseTrajectory.from_poses(
            np.array([s.timestamp for s in self.object_states]),
            [s.pose for s in self.object_states],
        )

    def initial_object_pose(self) -> Pose:
        return self.object_states[0].pose.copy()

    @property
    def action_type(self) -> ActionType:
        return self.metadata.action_type

    @property
    def duration(self) -> float:
        return self.metadata.end_time - self.metadata.start_time

    # -- validation -----------------------------------------------------------
    def validate(self) -> None:
        """Raise ``ValueError`` on structural problems."""
        if not self.robot_states:
            raise ValueError("episode has no robot states")
        if not self.object_states:
            raise ValueError("episode has no object states")
        for name, states in (
            ("robot", [s.timestamp for s in self.robot_states]),
            ("object", [s.timestamp for s in self.object_states]),
        ):
            ts = np.asarray(states)
            if np.any(np.diff(ts) <= 0):
                raise ValueError(f"{name} timestamps not strictly increasing")
        r0, r1 = self.robot_states[0].timestamp, self.robot_states[-1].timestamp
        o0, o1 = self.object_states[0].timestamp, self.object_states[-1].timestamp
        if min(r1, o1) - max(r0, o0) <= 0:
            raise ValueError("robot and object streams do not overlap in time")
