"""Calibration result plots (matplotlib, headless-safe).

Every figure is written to disk; nothing is shown interactively.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pgtwin.evaluation.metrics import position_errors_m
from pgtwin.real_data.pose_utils import PoseTrajectory
from pgtwin.real_data.time_alignment import align_trajectories

_REAL_STYLE = dict(color="#1f77b4", lw=2.0, label="real")
_BEFORE_STYLE = dict(color="#d62728", lw=1.6, ls="--", label="sim (before)")
_AFTER_STYLE = dict(color="#2ca02c", lw=1.8, label="sim (after)")


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_xy_trajectories(
    real: PoseTrajectory,
    sim_before: PoseTrajectory,
    sim_after: PoseTrajectory,
    path: str | Path,
    title: str = "Object XY trajectory: real vs sim",
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(real.positions[:, 0], real.positions[:, 1], **_REAL_STYLE)
    ax.plot(sim_before.positions[:, 0], sim_before.positions[:, 1], **_BEFORE_STYLE)
    ax.plot(sim_after.positions[:, 0], sim_after.positions[:, 1], **_AFTER_STYLE)
    ax.scatter(*real.positions[0, :2], color="k", zorder=5, s=40, label="start")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def plot_xyz_vs_time(
    real: PoseTrajectory,
    sim_before: PoseTrajectory,
    sim_after: PoseTrajectory,
    path: str | Path,
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i, axis_name in enumerate("xyz"):
        ax = axes[i]
        ax.plot(real.timestamps, real.positions[:, i], **_REAL_STYLE)
        ax.plot(sim_before.timestamps, sim_before.positions[:, i], **_BEFORE_STYLE)
        ax.plot(sim_after.timestamps, sim_after.positions[:, i], **_AFTER_STYLE)
        ax.set_ylabel(f"{axis_name} (m)")
        ax.grid(alpha=0.3)
    axes[0].set_title("Object position vs time: real vs sim")
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("time (s)")
    return _save(fig, path)


def plot_param_loss_curve(
    param_values: np.ndarray,
    losses: np.ndarray,
    best_value: float,
    path: str | Path,
    param_name: str = "friction",
    true_value: float | None = None,
) -> Path:
    order = np.argsort(param_values)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(param_values[order], losses[order], "o-", color="#1f77b4", ms=4)
    ax.axvline(best_value, color="#2ca02c", ls="--", lw=1.6,
               label=f"best = {best_value:.3f}")
    if true_value is not None:
        ax.axvline(true_value, color="#d62728", ls=":", lw=1.6,
                   label=f"ground truth = {true_value:.3f}")
    ax.set_xlabel(param_name)
    ax.set_ylabel("loss")
    ax.set_title(f"Loss landscape over {param_name}")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def plot_optimization_history(
    losses: np.ndarray, path: str | Path
) -> Path:
    best_so_far = np.minimum.accumulate(losses)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(losses, "o", ms=3, alpha=0.5, color="#1f77b4", label="evaluation")
    ax.plot(best_so_far, "-", lw=2, color="#2ca02c", label="best so far")
    ax.set_xlabel("evaluation #")
    ax.set_ylabel("loss")
    ax.set_title("Optimization history")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def plot_error_over_time(
    real: PoseTrajectory,
    sim_before: PoseTrajectory,
    sim_after: PoseTrajectory,
    path: str | Path,
    rate: float = 30.0,
) -> Path:
    r1, b, _ = align_trajectories(real, sim_before, rate)
    r2, a, _ = align_trajectories(real, sim_after, rate)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(r1.timestamps, position_errors_m(r1, b) * 100, **_BEFORE_STYLE)
    ax.plot(r2.timestamps, position_errors_m(r2, a) * 100, **_AFTER_STYLE)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("object position error (cm)")
    ax.set_title("Real-Sim object position error over time")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def plot_ee_replay_error(
    real_ee: PoseTrajectory,
    sim_ee: PoseTrajectory,
    path: str | Path,
    rate: float = 30.0,
) -> Path:
    """EE replay fidelity: if this error is large, object mismatch may be
    an action-replay artifact rather than a physics-parameter issue."""
    r, s, _ = align_trajectories(real_ee, sim_ee, rate)
    err_cm = position_errors_m(r, s) * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(r.timestamps, err_cm, color="#9467bd", lw=1.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("EE position error (cm)")
    ax.set_title(
        f"EE replay error (RMSE = {np.sqrt(np.mean(err_cm ** 2)):.3f} cm)"
    )
    ax.grid(alpha=0.3)
    return _save(fig, path)
