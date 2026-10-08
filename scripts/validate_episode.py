#!/usr/bin/env python3
"""Real-data readiness validator: check an episode directory before calibrating.

    python scripts/validate_episode.py data/real/push_box_001

Catches the failure modes that silently ruin calibration (spec §33):
missing/NaN columns, non-monotonic or non-overlapping timestamps, wrong
quaternion convention, implausible units, missing force data for LIFT,
and static episodes.  Exit code 0 = ready, 1 = blocking problems.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from pgtwin.real_data.episode import ActionType  # noqa: E402
from pgtwin.real_data.trajectory_loader import EpisodeLoader  # noqa: E402


def validate(episode_dir: Path) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    for name in ("metadata.json", "robot_states.csv", "object_states.csv"):
        if not (episode_dir / name).exists():
            errors.append(f"missing file: {name}")
    if errors:
        return errors, warnings

    try:
        # loader itself validates schema, monotonicity, stream overlap
        episode = EpisodeLoader.load(episode_dir)
    except Exception as exc:
        errors.append(f"loader rejected episode: {exc}")
        return errors, warnings

    ee = episode.ee_trajectory()
    obj = episode.object_trajectory()

    # -- units plausibility (meters, tabletop scale)
    for label, traj in (("EE", ee), ("object", obj)):
        span = float(np.max(np.abs(traj.positions)))
        if span > 20.0:
            errors.append(
                f"{label} positions reach {span:.1f} — not meters? "
                "(millimeters must be converted at export time)"
            )
        elif span > 5.0:
            warnings.append(f"{label} positions reach {span:.1f} m — unusual for tabletop")

    # -- quaternion sanity (unit norm; xyzw order heuristic)
    for label, traj in (("EE", ee), ("object", obj)):
        norms = np.linalg.norm(traj.quaternions, axis=1)
        if np.any(np.abs(norms - 1.0) > 0.05):
            warnings.append(f"{label} quaternions deviate from unit norm (max "
                            f"|1-n| = {np.max(np.abs(norms - 1.0)):.3f})")
    # xyzw vs wxyz: for near-upright tabletop objects, |q[3]| (w) should
    # dominate; a column of near-zero last components suggests wxyz order.
    w_med = float(np.median(np.abs(obj.quaternions[:, 3])))
    x_med = float(np.median(np.abs(obj.quaternions[:, 0])))
    if w_med < 0.3 and x_med > 0.7:
        errors.append(
            "object quaternions look scalar-FIRST (wxyz); pgtwin requires "
            "[x, y, z, w] — reorder columns at export"
        )

    # -- sampling
    for label, traj in (("robot", ee), ("object", obj)):
        dts = np.diff(traj.timestamps)
        rate = 1.0 / np.median(dts)
        if np.max(dts) > 5 * np.median(dts):
            warnings.append(
                f"{label} stream has gaps up to {np.max(dts):.3f} s "
                f"(median dt {np.median(dts):.4f} s)"
            )
        meta_rate = episode.metadata.sampling_rate
        if meta_rate and abs(rate - meta_rate) / meta_rate > 0.2:
            warnings.append(
                f"{label} measured rate {rate:.1f} Hz vs metadata "
                f"sampling_rate {meta_rate:.1f} Hz"
            )

    # -- action-specific signal checks
    action = episode.action_type
    if action in (ActionType.PUSH, ActionType.SLIDE):
        moved = obj.total_displacement()
        if moved < 0.01:
            errors.append(
                f"object moved only {moved * 100:.1f} cm — no push signal "
                "(check tracking or the episode window)"
            )
        elif moved < 0.03:
            warnings.append(f"object moved {moved * 100:.1f} cm; friction "
                            "identifiability improves with >= 5 cm of motion")
        approach = float(np.min(np.linalg.norm(
            ee.positions - obj.positions[0], axis=1)))
        if approach > 0.15:
            warnings.append(
                f"EE never comes within {approach:.2f} m of the object start "
                "— frames misaligned?"
            )
    if action in (ActionType.LIFT, ActionType.PRESS):
        has_force = all(s.ee_force is not None for s in episode.robot_states)
        if not has_force:
            errors.append(
                f"{action.value} episode has no fx,fy,fz columns — the "
                "observable for this action is the EE force"
            )
        else:
            fz = np.array([s.ee_force[2] for s in episode.robot_states])
            if float(np.max(np.abs(fz))) < 0.2:
                warnings.append("force magnitudes < 0.2 N — load cell connected?")
    if action is ActionType.LIFT:
        rise = float(obj.positions[-1][2] - obj.positions[0][2]) if len(obj) else 0.0
        rise = float(np.max(obj.positions[:, 2]) - obj.positions[0, 2])
        if rise < 0.02:
            warnings.append(f"object rises only {rise * 100:.1f} cm during lift")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("episode", help="episode directory")
    args = parser.parse_args()

    episode_dir = Path(args.episode)
    errors, warnings = validate(episode_dir)

    print(f"Episode: {episode_dir}")
    for w in warnings:
        print(f"  ⚠ {w}")
    for e in errors:
        print(f"  ✗ {e}")
    if not errors and not warnings:
        print("  ✓ all checks passed")
    elif not errors:
        print(f"  ✓ ready ({len(warnings)} warning(s))")
    else:
        print(f"  ✗ NOT ready: {len(errors)} blocking problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
