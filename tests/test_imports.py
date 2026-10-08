"""Import-order robustness: every package must import cleanly first.

Regression test for a circular-import bug where importing the simulator
package before the physics package raised ImportError (physics/__init__
eagerly pulled in calibrator, which imports the simulator layer).
"""

import subprocess
import sys

import pytest

ENTRY_MODULES = [
    "pgtwin.simulator.scene_builder",
    "pgtwin.simulator",
    "pgtwin.physics",
    "pgtwin.physics.calibrator",
    "pgtwin.real_data.synthetic",
    "pgtwin.evaluation.report",
    "pgtwin.gaussian",
    "pgtwin.cli.calibrate",
]


@pytest.mark.parametrize("module", ENTRY_MODULES)
def test_fresh_import(module):
    """Each module must import in a fresh interpreter (no order luck)."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"import {module} failed:\n{result.stderr}"


def test_lazy_physics_exports():
    import pgtwin.physics as ph

    assert ph.PhysicsCalibrator is not None
    assert ph.CalibrationResult is not None
    with pytest.raises(AttributeError):
        _ = ph.does_not_exist
