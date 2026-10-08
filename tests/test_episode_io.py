"""Episode save/load round-trip and validation tests."""

import json

import numpy as np
import pytest

from pgtwin.real_data.episode import ActionType, InteractionEpisode
from pgtwin.real_data.trajectory_loader import EpisodeLoader, save_episode


def test_roundtrip_preserves_trajectories(synthetic_push, tmp_path):
    episode, _ = synthetic_push
    save_episode(episode, tmp_path / "ep")
    loaded = EpisodeLoader.load(tmp_path / "ep")

    assert loaded.metadata.episode_id == episode.metadata.episode_id
    assert loaded.metadata.action_type is ActionType.PUSH
    assert len(loaded.object_states) == len(episode.object_states)
    np.testing.assert_allclose(
        loaded.object_trajectory().positions,
        episode.object_trajectory().positions,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        loaded.ee_trajectory().quaternions,
        episode.ee_trajectory().quaternions,
        atol=1e-6,
    )


def test_ground_truth_survives_roundtrip(synthetic_push, tmp_path):
    episode, _ = synthetic_push
    save_episode(episode, tmp_path / "ep")
    loaded = EpisodeLoader.load(tmp_path / "ep")
    assert loaded.metadata.extra["ground_truth"]["friction"] == pytest.approx(0.8)


def test_expected_files_written(synthetic_push, tmp_path):
    episode, _ = synthetic_push
    save_episode(episode, tmp_path / "ep")
    for name in ("metadata.json", "robot_states.csv", "object_states.csv"):
        assert (tmp_path / "ep" / name).exists()


def test_missing_column_raises(synthetic_push, tmp_path):
    episode, _ = synthetic_push
    d = tmp_path / "ep"
    save_episode(episode, d)
    text = (d / "object_states.csv").read_text().splitlines()
    (d / "object_states.csv").write_text(
        "\n".join([text[0].replace("qw", "quat_w"), *text[1:]])
    )
    with pytest.raises(ValueError, match="missing required columns"):
        EpisodeLoader.load(d)


def test_missing_metadata_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        EpisodeLoader.load(tmp_path / "empty")


def test_unsorted_timestamps_are_sorted(synthetic_push, tmp_path):
    episode, _ = synthetic_push
    d = tmp_path / "ep"
    save_episode(episode, d)
    import pandas as pd

    df = pd.read_csv(d / "object_states.csv")
    df.iloc[::-1].to_csv(d / "object_states.csv", index=False)
    loaded = EpisodeLoader.load(d)
    ts = np.array([s.timestamp for s in loaded.object_states])
    assert np.all(np.diff(ts) > 0)


def test_resample_on_load(synthetic_push, tmp_path):
    episode, _ = synthetic_push
    save_episode(episode, tmp_path / "ep")
    loaded = EpisodeLoader.load(tmp_path / "ep", resample_rate=10.0)
    ts = np.array([s.timestamp for s in loaded.object_states])
    np.testing.assert_allclose(np.diff(ts), 0.1, atol=1e-9)
    assert loaded.metadata.sampling_rate == pytest.approx(10.0)


def test_validate_rejects_disjoint_streams(synthetic_push):
    episode, _ = synthetic_push
    shifted = InteractionEpisode(
        metadata=episode.metadata,
        robot_states=episode.robot_states,
        object_states=[
            type(s)(timestamp=s.timestamp + 1e4, pose=s.pose) for s in episode.object_states
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        shifted.validate()


def test_metadata_json_is_readable(synthetic_push, tmp_path):
    episode, _ = synthetic_push
    save_episode(episode, tmp_path / "ep")
    data = json.loads((tmp_path / "ep" / "metadata.json").read_text())
    assert data["action_type"] == "push"
    assert data["robot"] == "franka"
