"""Robot abstraction.

Phase 1-2 calibration only needs the *recorded EE trajectory*, so robot
adapters are thin: they describe kinematic metadata (EE geometry, base
transform, joint limits) and normalize vendor-specific state into
:class:`RobotState`.  Real hardware/SDK drivers plug in behind this
interface in later phases without touching calibration code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pgtwin.common.se3 import Pose
from pgtwin.real_data.episode import RobotState


class BaseRobot(ABC):
    """Metadata + state normalization for one robot type."""

    name: str = "base"
    dof: int = 0

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        base = self.config.get("base_pose", {})
        #: T_world_robot — where the robot base sits in the world frame.
        self.base_pose = Pose(
            position=base.get("position", [0.0, 0.0, 0.0]),
            quaternion=base.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]),
        )

    @property
    @abstractmethod
    def ee_radius(self) -> float:
        """Effective EE tip contact radius (m) for kinematic push replay."""

    def ee_pose_world(self, state: RobotState) -> Pose:
        """EE pose in world frame. Episodes store world-frame EE already;
        override if a data source records base-frame EE poses."""
        return state.ee_pose

    def validate_state(self, state: RobotState) -> None:
        if state.joint_position is not None and self.dof:
            if len(state.joint_position) != self.dof:
                raise ValueError(
                    f"{self.name}: expected {self.dof} joints, "
                    f"got {len(state.joint_position)}"
                )

    def to_dict(self) -> dict:
        return {"name": self.name, "dof": self.dof, "ee_radius": self.ee_radius}
