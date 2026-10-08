"""`pgtwin evaluate`: recompute metrics/plots for a saved experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pgtwin.common.io import load_json
from pgtwin.evaluation import visualization as viz
from pgtwin.evaluation.metrics import trajectory_metrics
from pgtwin.real_data.pose_utils import PoseTrajectory


def _load_traj_csv(path: Path) -> PoseTrajectory:
    df = pd.read_csv(path)
    return PoseTrajectory(
        df["timestamp"].to_numpy(),
        df[["x", "y", "z"]].to_numpy(),
        df[["qx", "qy", "qz", "qw"]].to_numpy(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pgtwin evaluate",
        description="Re-evaluate a saved calibration experiment directory.",
    )
    parser.add_argument("--experiment", required=True, help="outputs/<experiment_id>")
    parser.add_argument("--rate", type=float, default=30.0, help="resample rate (Hz)")
    args = parser.parse_args(argv)

    exp_dir = Path(args.experiment)
    real = _load_traj_csv(exp_dir / "real_trajectory.csv")
    before = _load_traj_csv(exp_dir / "sim_before.csv")
    after = _load_traj_csv(exp_dir / "sim_after.csv")

    print(f"Experiment: {exp_dir.name}")
    if (exp_dir / "result.json").exists():
        result = load_json(exp_dir / "result.json")
        print(f"  optimized params: {result.get('optimized_params')}")

    for label, sim in (("before", before), ("after", after)):
        m = trajectory_metrics(real, sim, args.rate)
        print(f"  [{label}] " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))

    viz.plot_xy_trajectories(real, before, after, exp_dir / "trajectory_xy.png")
    viz.plot_error_over_time(real, before, after, exp_dir / "error_time.png", args.rate)
    print(f"  plots refreshed in {exp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
