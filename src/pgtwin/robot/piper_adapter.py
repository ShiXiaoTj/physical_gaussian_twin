"""AgileX Piper adapter (secondary robot, interface parity with Franka)."""

from __future__ import annotations

from pgtwin.robot.base_robot import BaseRobot


class PiperAdapter(BaseRobot):
    """AgileX Piper: 6-DoF arm.  Same episode format as Franka; only EE
    geometry and joint metadata differ, so calibration code is unchanged."""

    name = "piper"
    dof = 6

    @property
    def ee_radius(self) -> float:
        return float(self.config.get("ee_radius", 0.010))
