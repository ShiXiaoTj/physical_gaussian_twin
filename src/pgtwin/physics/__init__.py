"""Physics calibration core: objects, parameter spaces, losses, optimizers.

``PhysicsCalibrator`` / ``CalibrationResult`` / ``HierarchicalCalibrator``
are exported lazily (PEP 562): calibrator.py imports the simulator layer,
which imports ``physics.physical_object`` — an eager import here would make
the package initialization order dependent on which module is imported
first (regression test: tests/test_imports.py).
"""

from pgtwin.physics.losses import PositionLoss, RotationLoss, TrajectoryLoss
from pgtwin.physics.parameter_space import ParameterSpace, ParameterSpec
from pgtwin.physics.physical_object import (
    CollisionProxy,
    PhysicalObject,
    PhysicsParameters,
)

_LAZY = {"CalibrationResult", "PhysicsCalibrator", "HierarchicalCalibrator",
         "ParameterCalibrationPolicy"}


def __getattr__(name: str):
    if name in _LAZY:
        from pgtwin.physics import calibrator

        return getattr(calibrator, name)
    raise AttributeError(f"module 'pgtwin.physics' has no attribute '{name}'")


__all__ = [
    "CalibrationResult",
    "PhysicsCalibrator",
    "HierarchicalCalibrator",
    "ParameterCalibrationPolicy",
    "PositionLoss",
    "RotationLoss",
    "TrajectoryLoss",
    "ParameterSpace",
    "ParameterSpec",
    "CollisionProxy",
    "PhysicalObject",
    "PhysicsParameters",
]
