#!/usr/bin/env python3
"""Replay one episode's EE trajectory in a simulator and report replay
fidelity + object trajectory (no calibration).

    python scripts/replay_episode.py --episode data/generated/push_box_001 \
        --config configs/experiments/push_box.yaml [--friction 0.8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pgtwin.common.io import load_config  # noqa: E402
from pgtwin.evaluation.metrics import trajectory_metrics  # noqa: E402
from pgtwin.physics.calibrator import PhysicsCalibrator  # noqa: E402
from pgtwin.physics.physical_object import PhysicalObject  # noqa: E402
from pgtwin.real_data.trajectory_loader import EpisodeLoader  # noqa: E402
from pgtwin.simulator import create_simulator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episode", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("--friction", type=float, help="override friction for the rollout")
    args = parser.parse_args()

    config = load_config(args.config)
    episode = EpisodeLoader.load(args.episode)
    obj_cfg = dict(config.get("object", {}))
    obj_cfg.setdefault("object_id", episode.metadata.object_id)
    obj = PhysicalObject.from_config(obj_cfg)

    sim = create_simulator(config.get("simulation", {}))
    calibrator = PhysicsCalibrator(sim, scene_cfg=config.get("scene", {}))
    calibrator._setup_scene(episode, obj)

    params = {"friction": args.friction} if args.friction is not None else {
        "friction": obj.physics.friction_dynamic
    }
    result = calibrator.simulate(episode, obj, params)

    metrics = trajectory_metrics(episode.object_trajectory(), result.object_trajectory)
    print(f"replayed '{episode.metadata.episode_id}' with {params}")
    print(f"  EE replay RMSE: {result.ee_replay_rmse() * 100:.4f} cm")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
