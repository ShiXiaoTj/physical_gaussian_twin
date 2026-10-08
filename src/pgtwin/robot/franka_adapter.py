"""Franka Emika Panda adapter (primary Phase 1+ robot)."""

from __future__ import annotations

from pgtwin.robot.base_robot import BaseRobot


class FrankaAdapter(BaseRobot):
    """Franka Panda: 7-DoF arm; push experiments use a closed-gripper tip
    or a cylindrical pusher tool mounted on the flange."""

    name = "franka"
    dof = 7

    # joint limits (rad), franka_description values
    JOINT_LIMITS_LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    JOINT_LIMITS_UPPER = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]

    @property
    def ee_radius(self) -> float:
        # closed Franka gripper fingertip envelope ~ 12 mm; overridable
        return float(self.config.get("ee_radius", 0.012))
