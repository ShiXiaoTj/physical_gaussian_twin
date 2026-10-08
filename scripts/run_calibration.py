#!/usr/bin/env python3
"""One-command calibration demo / runner.

Quick start (no real data required — generates a synthetic push episode):

    python scripts/run_calibration.py --example push_box

Real / custom data:

    python scripts/run_calibration.py \
        --episode data/real/push_box_001 \
        --config configs/experiments/push_box.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pgtwin.cli.calibrate import run_pipeline  # noqa: E402
from pgtwin.common.io import load_config  # noqa: E402
from pgtwin.real_data.synthetic import SyntheticEpisodeGenerator  # noqa: E402

EXAMPLES = {
    "push_box": {
        "config": PROJECT_ROOT / "configs" / "experiments" / "push_box.yaml",
        "episode": PROJECT_ROOT / "data" / "generated" / "push_box_001",
        "synthetic": {
            "episode_id": "push_box_001",
            "true_friction": 0.8,
            "noise_std_pos": 0.001,  # 1 mm measurement noise
            "seed": 0,
        },
    },
    "push_box_mujoco": {
        "config": PROJECT_ROOT / "configs" / "experiments" / "push_box_mujoco.yaml",
        "episode": PROJECT_ROOT / "data" / "generated" / "push_box_mujoco_001",
        "synthetic": {
            "episode_id": "push_box_mujoco_001",
            "true_friction": 0.8,
            "backend": "mujoco",
            "sim_dt": 0.001,
            "noise_std_pos": 0.001,
            "seed": 0,
        },
    },
    # Push the teacher's desk instance segmented from a real 3DGS scene.
    # Requires data/gaussian/classroom_scene.npz (see the experiment config
    # header for the prepare_gaussian_scene.py command).
    # object_from_scene: the synthetic episode pushes the SAME
    # gaussian-derived collision proxy used later by calibration — a
    # geometry mismatch here trades off directly against friction
    # (docs/experiment_protocol.md §2.1, geometry-friction coupling).
    "push_desk_real_scene": {
        "config": PROJECT_ROOT / "configs" / "experiments" / "push_desk_real_scene.yaml",
        "episode": PROJECT_ROOT / "data" / "generated" / "push_desk_real_scene_001",
        "object_from_scene": True,
        "synthetic": {
            "episode_id": "push_desk_real_scene_001",
            "object_id": "desk_01",
            "true_friction": 0.5,
            "true_mass": 3.0,
            "push_speed": 0.5,
            "noise_std_pos": 0.001,
            "seed": 0,
        },
    },
    # Real 3DGS scene #2: push the red kids' chair in the Deep Blending
    # playroom (metric scale double-anchored: ceiling + desk height).
    "push_chair_playroom": {
        "config": PROJECT_ROOT / "configs" / "experiments" / "push_chair_playroom.yaml",
        "episode": PROJECT_ROOT / "data" / "generated" / "push_chair_playroom_001",
        "object_from_scene": True,
        "synthetic": {
            "episode_id": "push_chair_playroom_001",
            "object_id": "red_chair_01",
            "true_friction": 0.6,
            "true_mass": 3.0,
            "push_speed": 0.5,
            "noise_std_pos": 0.001,
            "seed": 0,
        },
    },
    # THE FULL CHAIN: photos we trained ourselves (gsplat, RTX 4090D) ->
    # aligned & double-anchored metric scene -> push the vase on the table.
    # Explicit measured cylinder proxy (flowers excluded from physics).
    "push_vase_garden": {
        "config": PROJECT_ROOT / "configs" / "experiments" / "push_vase_garden.yaml",
        "episode": PROJECT_ROOT / "data" / "generated" / "push_vase_garden_001",
        "synthetic": {
            "episode_id": "push_vase_garden_001",
            "object_id": "vase_01",
            "true_friction": 0.55,
            "true_mass": 1.2,
            "object_xy": [0.220, -0.491],      # measured vase body center
            "object_size": [0.18, 0.18, 0.26], # body cylinder as box extents
            "table_height": 0.74,
            "push_speed": 0.4,
            "push_distance": 0.10,
            "noise_std_pos": 0.001,
            "seed": 0,
        },
    },
    # Real 3DGS scene #3: vehicle-scale slide in the Tanks&Temples truck
    # scene (locked-wheel sliding; quasi-static slide is mass-independent).
    "slide_truck": {
        "config": PROJECT_ROOT / "configs" / "experiments" / "slide_truck.yaml",
        "episode": PROJECT_ROOT / "data" / "generated" / "slide_truck_001",
        "object_from_scene": True,
        "synthetic": {
            "episode_id": "slide_truck_001",
            "object_id": "truck_01",
            "action_type": "slide",
            "true_friction": 0.7,
            "true_mass": 1800.0,
            "push_speed": 0.8,
            "push_distance": 0.35,
            "retract_distance": 0.20,
            "hold_time": 2.5,
            "noise_std_pos": 0.002,
            "seed": 0,
        },
    },
}


def build_object_from_scene(config_path: Path):
    """Segment the config's gaussian instance and derive its physical object
    (collision proxy from the real gaussians). Returns (obj, center_xy)."""
    from pgtwin.gaussian.instance_manager import InstanceManager
    from pgtwin.gaussian.scene_loader import GaussianScene

    config = load_config(config_path)
    gcfg = config["gaussian"]
    scene = GaussianScene.load(PROJECT_ROOT / gcfg["scene_path"])
    bbox = gcfg["instance_bbox"]
    instance = InstanceManager(scene).segment(
        config["object"]["object_id"], bbox_min=bbox["min"], bbox_max=bbox["max"]
    )
    obj = instance.to_physical_object(
        proxy_method=config["object"].get("proxy_method", "convex_hull"),
        gaussian_asset=gcfg["scene_path"],
    )
    center = instance.center_world()
    return obj, (float(center[0]), float(center[1]))


def ensure_example_episode(name: str, force: bool = False) -> Path:
    spec = EXAMPLES[name]
    episode_dir: Path = spec["episode"]
    if force or not (episode_dir / "metadata.json").exists():
        print(f"generating synthetic episode '{name}' -> {episode_dir}")
        synthetic_cfg = dict(spec["synthetic"])
        scene_obj = None
        if spec.get("object_from_scene"):
            scene_obj, center_xy = build_object_from_scene(spec["config"])
            synthetic_cfg["object_xy"] = list(center_xy)
            print(f"    pushing gaussian-proxy object '{scene_obj.object_id}' "
                  f"at {center_xy}")
        generator = SyntheticEpisodeGenerator.from_config(synthetic_cfg)
        generator.generate(save_dir=episode_dir, obj=scene_obj)
    return episode_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--example", choices=sorted(EXAMPLES),
                        help="run a bundled example (generates synthetic data)")
    parser.add_argument("--episode", help="episode directory (real data)")
    parser.add_argument("--config", help="experiment YAML config")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--regenerate", action="store_true",
                        help="regenerate the synthetic example episode")
    args = parser.parse_args()

    if args.example:
        episode_dir = ensure_example_episode(args.example, args.regenerate)
        config_path = args.config or EXAMPLES[args.example]["config"]
    elif args.episode and args.config:
        episode_dir, config_path = Path(args.episode), Path(args.config)
    else:
        parser.error("provide --example NAME, or both --episode and --config")

    config = load_config(config_path)
    result = run_pipeline(config, episode_dir, args.output)

    print("\nCalibration completed")
    for key in result.optimized_params:
        print(f"  Initial {key}: {result.initial_params[key]:.4f}")
        print(f"  Optimized {key}: {result.optimized_params[key]:.4f}")
    print(f"  Initial RMSE: {result.initial_metrics['rmse_position_cm']:.2f} cm")
    print(f"  Final RMSE: {result.final_metrics['rmse_position_cm']:.2f} cm")
    print(f"  Improvement: {result.improvement['rmse_position_cm']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
