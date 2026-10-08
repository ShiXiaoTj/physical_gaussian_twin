"""Experiment reporting: reproducible on-disk experiment records.

Every calibration run writes a self-contained directory:

    outputs/<experiment_id>/
    ├── config.yaml               # exact config used
    ├── result.json               # params, metrics, improvement
    ├── calibration_history.csv   # every objective evaluation
    ├── real_trajectory.csv       # aligned real object trajectory
    ├── sim_before.csv / sim_after.csv
    ├── trajectory_xy.png / trajectory_xyz.png
    ├── error_time.png / ee_replay_error.png
    └── optimization_curve.png / optimization_history.png
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pgtwin.common.io import save_json, save_yaml
from pgtwin.common.logging import get_logger
from pgtwin.evaluation import visualization as viz
from pgtwin.real_data.pose_utils import PoseTrajectory

if TYPE_CHECKING:  # avoid circular import (calibrator uses evaluation.metrics)
    from pgtwin.physics.calibrator import CalibrationResult

logger = get_logger("evaluation.report")


def trajectory_to_csv(traj: PoseTrajectory, path: str | Path) -> None:
    df = pd.DataFrame(
        {
            "timestamp": traj.timestamps,
            "x": traj.positions[:, 0],
            "y": traj.positions[:, 1],
            "z": traj.positions[:, 2],
            "qx": traj.quaternions[:, 0],
            "qy": traj.quaternions[:, 1],
            "qz": traj.quaternions[:, 2],
            "qw": traj.quaternions[:, 3],
        }
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class ExperimentReporter:
    """Writes a complete, reproducible record of a calibration run."""

    def __init__(self, output_root: str | Path, experiment_id: str):
        self.dir = Path(output_root) / experiment_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id

    def save_config(self, config: dict) -> None:
        save_yaml(config, self.dir / "config.yaml")

    def save_result(
        self, result: CalibrationResult, true_params: dict | None = None
    ) -> None:
        payload = result.to_dict()
        if true_params:
            payload["ground_truth_params"] = true_params
        save_json(payload, self.dir / "result.json")

        # full optimization history
        hist = result.optimization.history_arrays()
        pd.DataFrame(hist).to_csv(self.dir / "calibration_history.csv", index=False)

        # trajectories
        if result.real_object is not None:
            trajectory_to_csv(result.real_object, self.dir / "real_trajectory.csv")
        if result.sim_before is not None:
            trajectory_to_csv(result.sim_before, self.dir / "sim_before.csv")
        if result.sim_after is not None:
            trajectory_to_csv(result.sim_after, self.dir / "sim_after.csv")

    def save_plots(
        self,
        result: CalibrationResult,
        resample_rate: float = 30.0,
        true_params: dict | None = None,
    ) -> None:
        real, before, after = result.real_object, result.sim_before, result.sim_after
        if real is None or before is None or after is None:
            logger.warning("trajectories missing on result; skipping plots")
            return
        viz.plot_xy_trajectories(real, before, after, self.dir / "trajectory_xy.png")
        viz.plot_xyz_vs_time(real, before, after, self.dir / "trajectory_xyz.png")
        viz.plot_error_over_time(
            real, before, after, self.dir / "error_time.png", rate=resample_rate
        )
        if result.real_ee is not None and result.sim_ee is not None:
            viz.plot_ee_replay_error(
                result.real_ee,
                result.sim_ee,
                self.dir / "ee_replay_error.png",
                rate=resample_rate,
            )

        hist = result.optimization.history_arrays()
        viz.plot_optimization_history(hist["loss"], self.dir / "optimization_history.png")
        # 1-D loss landscape per calibrated parameter
        for name in result.optimized_params:
            if name in hist:
                viz.plot_param_loss_curve(
                    hist[name],
                    hist["loss"],
                    best_value=result.optimized_params[name],
                    path=self.dir / "optimization_curve.png",
                    param_name=name,
                    true_value=(true_params or {}).get(name),
                )

    def finalize(
        self,
        result: CalibrationResult,
        config: dict,
        resample_rate: float = 30.0,
        true_params: dict | None = None,
    ) -> Path:
        """Write config + result + plots; returns the experiment directory."""
        self.save_config(config)
        self.save_result(result, true_params)
        self.save_plots(result, resample_rate, true_params)
        logger.info("experiment record written to %s", self.dir)
        return self.dir
