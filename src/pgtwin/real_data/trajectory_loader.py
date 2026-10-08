"""Episode disk format loader / saver.

Episode directory layout (docs/architecture.md, "Episode format"):

    episode_0001/
    ├── metadata.json        # EpisodeMetadata
    ├── robot_states.csv     # timestamp, ee_x..ee_qw, gripper[, fx fy fz][, joint_*]
    ├── object_states.csv    # timestamp, x y z qx qy qz qw
    ├── camera/              # optional raw captures (unused in Phase 1-2)
    └── scene/               # optional gaussian scene assets

The loader validates required columns, sorts and de-duplicates timestamps,
and normalizes the time origin (optional).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pgtwin.common.logging import get_logger
from pgtwin.common.se3 import Pose
from pgtwin.real_data.episode import (
    EpisodeMetadata,
    InteractionEpisode,
    ObjectState,
    RobotState,
)
from pgtwin.common.io import load_json, save_json

logger = get_logger("real_data.loader")

ROBOT_REQUIRED = ["timestamp", "ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"]
OBJECT_REQUIRED = ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
FORCE_COLS = ["fx", "fy", "fz"]


def _clean_frame(df: pd.DataFrame, required: list[str], name: str) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")
    df = df.dropna(subset=required)
    df = df.sort_values("timestamp")
    n_before = len(df)
    df = df.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
    if len(df) < n_before:
        logger.warning("%s: dropped %d duplicate timestamps", name, n_before - len(df))
    if len(df) < 2:
        raise ValueError(f"{name}: needs >= 2 valid rows, got {len(df)}")
    return df


class EpisodeLoader:
    """Load / save :class:`InteractionEpisode` from the standard layout."""

    @staticmethod
    def load(
        episode_dir: str | Path,
        normalize_time: bool = True,
        resample_rate: float | None = None,
    ) -> InteractionEpisode:
        episode_dir = Path(episode_dir)
        meta_path = episode_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {episode_dir}")
        metadata = EpisodeMetadata.from_dict(load_json(meta_path))

        robot_df = _clean_frame(
            pd.read_csv(episode_dir / "robot_states.csv"), ROBOT_REQUIRED, "robot_states"
        )
        object_df = _clean_frame(
            pd.read_csv(episode_dir / "object_states.csv"), OBJECT_REQUIRED, "object_states"
        )

        t0 = min(robot_df["timestamp"].iloc[0], object_df["timestamp"].iloc[0])
        if normalize_time and abs(t0) > 1e-9:
            logger.info("normalizing time origin: t0 = %.6f s -> 0", t0)
            robot_df["timestamp"] -= t0
            object_df["timestamp"] -= t0

        has_force = all(c in robot_df.columns for c in FORCE_COLS)
        joint_cols = sorted(
            [c for c in robot_df.columns if c.startswith("joint_")],
            key=lambda c: int(c.split("_")[1]),
        )

        robot_states = []
        for row in robot_df.itertuples(index=False):
            d = row._asdict()
            robot_states.append(
                RobotState(
                    timestamp=float(d["timestamp"]),
                    ee_pose=Pose(
                        position=[d["ee_x"], d["ee_y"], d["ee_z"]],
                        quaternion=[d["ee_qx"], d["ee_qy"], d["ee_qz"], d["ee_qw"]],
                    ),
                    gripper_state=float(d.get("gripper", 0.0)),
                    ee_force=np.array([d[c] for c in FORCE_COLS]) if has_force else None,
                    joint_position=np.array([d[c] for c in joint_cols])
                    if joint_cols
                    else None,
                )
            )

        object_states = []
        for row in object_df.itertuples(index=False):
            d = row._asdict()
            object_states.append(
                ObjectState(
                    timestamp=float(d["timestamp"]),
                    pose=Pose(
                        position=[d["x"], d["y"], d["z"]],
                        quaternion=[d["qx"], d["qy"], d["qz"], d["qw"]],
                    ),
                )
            )

        episode = InteractionEpisode(metadata, robot_states, object_states)
        episode.validate()
        if resample_rate is not None:
            episode = _resample_episode(episode, resample_rate)
        logger.info(
            "loaded episode '%s': action=%s, %d robot samples, %d object samples, %.2fs",
            metadata.episode_id,
            metadata.action_type.value,
            len(robot_states),
            len(object_states),
            episode.object_trajectory().duration,
        )
        return episode

    @staticmethod
    def save(episode: InteractionEpisode, episode_dir: str | Path) -> Path:
        episode_dir = Path(episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "camera").mkdir(exist_ok=True)
        (episode_dir / "scene").mkdir(exist_ok=True)

        save_json(episode.metadata.to_dict(), episode_dir / "metadata.json")

        robot_rows = []
        for s in episode.robot_states:
            row = {
                "timestamp": s.timestamp,
                "ee_x": s.ee_pose.position[0],
                "ee_y": s.ee_pose.position[1],
                "ee_z": s.ee_pose.position[2],
                "ee_qx": s.ee_pose.quaternion[0],
                "ee_qy": s.ee_pose.quaternion[1],
                "ee_qz": s.ee_pose.quaternion[2],
                "ee_qw": s.ee_pose.quaternion[3],
                "gripper": s.gripper_state,
            }
            if s.ee_force is not None:
                row.update(dict(zip(FORCE_COLS, s.ee_force)))
            robot_rows.append(row)
        pd.DataFrame(robot_rows).to_csv(episode_dir / "robot_states.csv", index=False)

        object_rows = [
            {
                "timestamp": s.timestamp,
                "x": s.pose.position[0],
                "y": s.pose.position[1],
                "z": s.pose.position[2],
                "qx": s.pose.quaternion[0],
                "qy": s.pose.quaternion[1],
                "qz": s.pose.quaternion[2],
                "qw": s.pose.quaternion[3],
            }
            for s in episode.object_states
        ]
        pd.DataFrame(object_rows).to_csv(episode_dir / "object_states.csv", index=False)
        logger.info("saved episode '%s' -> %s", episode.metadata.episode_id, episode_dir)
        return episode_dir


def _resample_episode(
    episode: InteractionEpisode, rate: float
) -> InteractionEpisode:
    """Resample both state streams onto a uniform ``rate`` Hz timeline
    (linear position interpolation + quaternion SLERP)."""
    from dataclasses import replace

    from pgtwin.real_data.pose_utils import make_timestamps

    ee = episode.ee_trajectory()
    obj = episode.object_trajectory()
    t0 = max(ee.start_time, obj.start_time)
    t1 = min(ee.end_time, obj.end_time)
    ts = make_timestamps(t0, t1, rate)
    ee_rs, obj_rs = ee.resample(ts), obj.resample(ts)

    robot_states = [
        RobotState(timestamp=float(t), ee_pose=ee_rs.pose(i))
        for i, t in enumerate(ts)
    ]
    object_states = [
        ObjectState(timestamp=float(t), pose=obj_rs.pose(i))
        for i, t in enumerate(ts)
    ]
    metadata = replace(
        episode.metadata, sampling_rate=float(rate), start_time=float(t0), end_time=float(t1)
    )
    return InteractionEpisode(metadata, robot_states, object_states)


#: Module-level convenience alias.
save_episode = EpisodeLoader.save
