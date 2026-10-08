"""`pgtwin calibrate`: the end-to-end calibration pipeline.

Steps (printed as [1]..[14] at runtime):
load episode -> align -> load digital twin -> load robot -> load object
-> replay push -> baseline sim -> search parameters -> calibrated sim
-> metrics/improvement -> save calibrated Physical Gaussian Twin.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pgtwin.common.io import load_config
from pgtwin.common.logging import get_logger
from pgtwin.evaluation.report import ExperimentReporter
from pgtwin.gaussian.instance_manager import InstanceManager
from pgtwin.gaussian.scene_loader import GaussianScene
from pgtwin.physics.calibrator import CalibrationResult, PhysicsCalibrator
from pgtwin.physics.losses import TrajectoryLoss
from pgtwin.physics.optimizer import create_optimizer
from pgtwin.physics.parameter_space import ParameterSpace
from pgtwin.physics.physical_object import PhysicalObject
from pgtwin.real_data.episode import InteractionEpisode
from pgtwin.real_data.trajectory_loader import EpisodeLoader
from pgtwin.robot import create_robot
from pgtwin.simulator import create_simulator

logger = get_logger("cli.calibrate")


def _load_gaussian_object(cfg: dict, episode: InteractionEpisode):
    """Load / synthesize the Gaussian digital twin and segment the target.

    Instance selection priority:
    1. explicit ``gaussian.instance_bbox: {min: [...], max: [...]}`` (world m)
    2. bbox of ``gaussian.bbox_margin`` around the episode's initial pose

    Returns (GaussianObject | None, asset_path | None).  Never blocks the
    pipeline: without a scene the physics side still calibrates.
    """
    gcfg = cfg.get("gaussian", {})
    source = gcfg.get("scene_path")
    obj_pos = episode.initial_object_pose().position
    try:
        if source and Path(source).exists():
            scene = GaussianScene.load(source, max_points=gcfg.get("max_points"))
        elif gcfg.get("synthetic", True):
            size = cfg.get("object", {}).get("size", [0.06, 0.06, 0.06])
            table = GaussianScene.synthetic_tabletop()
            box = GaussianScene.synthetic_box(center=tuple(obj_pos), size=tuple(size))
            scene = table.merge(box)
        else:
            return None, None

        bbox_cfg = gcfg.get("instance_bbox")
        if bbox_cfg is not None:
            bbox_min = np.asarray(bbox_cfg["min"], dtype=float)
            bbox_max = np.asarray(bbox_cfg["max"], dtype=float)
        else:
            margin = float(gcfg.get("bbox_margin", 0.06))
            bbox_min, bbox_max = obj_pos - margin, obj_pos + margin

        manager = InstanceManager(scene)
        instance = manager.segment(
            episode.metadata.object_id, bbox_min=bbox_min, bbox_max=bbox_max
        )
        return instance, source
    except Exception as exc:
        logger.warning("gaussian twin unavailable (%s); continuing physics-only", exc)
        return None, None


def run_pipeline(
    config: dict,
    episode_dir: str | Path,
    output_root: str | Path = "outputs",
) -> CalibrationResult:
    """Run the full calibration pipeline from a config dict. Returns the result."""
    resample_rate = float(config.get("evaluation", {}).get("resample_rate", 30.0))

    print(f"[ 1] Loading real episode: {episode_dir}")
    episode = EpisodeLoader.load(episode_dir)
    experiment_id = config.get("experiment_id", episode.metadata.episode_id)

    print(f"[ 2] Aligning trajectories (common timeline @ {resample_rate:.0f} Hz)")
    loss = TrajectoryLoss.from_config(config.get("loss", {}), resample_rate)

    print("[ 3] Loading digital twin (Gaussian scene + instance segmentation)")
    gaussian_object, gaussian_asset = _load_gaussian_object(config, episode)
    if gaussian_object is not None:
        print(
            f"      instance '{gaussian_object.instance_id}': "
            f"{len(gaussian_object.gaussian_indices)} gaussians"
        )

    robot_name = config.get("robot", episode.metadata.robot)
    print(f"[ 4] Loading robot adapter: {robot_name}")
    robot = create_robot(robot_name if isinstance(robot_name, str) else dict(robot_name))

    print("[ 5] Loading object (collision proxy + initial physics)")
    obj_cfg = dict(config.get("object", {}))
    obj_cfg.setdefault("object_id", episode.metadata.object_id)
    if gaussian_object is not None and obj_cfg.get("proxy_from_gaussians", False):
        physical_object = gaussian_object.to_physical_object(
            proxy_method=obj_cfg.get("proxy_method", "convex_hull"),
            gaussian_asset=gaussian_asset,
        )
        physical_object.physics = PhysicalObject.from_config(obj_cfg).physics
    else:
        physical_object = PhysicalObject.from_config(obj_cfg)
        physical_object.gaussian_asset = gaussian_asset

    sim_cfg = config.get("simulation", {})
    simulator = create_simulator(sim_cfg)
    print(f"[ 6] Replaying push action in simulator (backend={sim_cfg.get('backend', 'mock')}, "
          f"dt={simulator.dt}s, kinematic EE replay)")

    space = ParameterSpace.from_config(config["calibration"]["parameters"])
    optimizer = create_optimizer(config.get("calibration", {}))
    calibrator = PhysicsCalibrator(
        simulator,
        loss=loss,
        settle_time=float(sim_cfg.get("settle_time", 0.5)),
        scene_cfg=config.get("scene", {}),
        robot_cfg=robot.to_dict(),
    )

    print("[ 7] Running initial (uncalibrated) simulation")
    result = calibrator.calibrate(episode, physical_object, space, optimizer)

    print(f"[ 8] Initial object RMSE = {result.initial_metrics['rmse_position_cm']:.2f} cm "
          f"(params: {result.initial_params})")
    print(f"[ 9] Searching parameters: {list(result.optimized_params)} "
          f"({result.optimization.n_evaluations} evaluations)")
    best_str = ", ".join(f"{k} = {v:.4f}" for k, v in result.optimized_params.items())
    print(f"[10] Best parameters: {best_str}")
    print("[11] Running calibrated simulation")
    print(f"[12] Final object RMSE = {result.final_metrics['rmse_position_cm']:.2f} cm")
    print(f"[13] Improvement = {result.improvement['rmse_position_cm']:.1f}%  "
          f"(EE replay RMSE = {result.ee_replay_rmse_m * 100:.3f} cm)")

    # -- save calibrated Physical Gaussian Twin ------------------------------
    true_params = episode.metadata.extra.get("ground_truth")
    reporter = ExperimentReporter(output_root, experiment_id)
    reporter.finalize(result, config, resample_rate, true_params)

    calibrated = PhysicalObject(
        object_id=physical_object.object_id,
        collision=physical_object.collision,
        physics=physical_object.physics.update(result.optimized_params),
        gaussian_asset=physical_object.gaussian_asset,
        initial_pose=episode.initial_object_pose(),
        calibration_info={
            "method": episode.action_type.value,
            "optimizer": optimizer.name,
            "episode": episode.metadata.episode_id,
            "trajectory_rmse_cm": result.final_metrics["rmse_position_cm"],
            "improvement_percent": result.improvement["rmse_position_cm"],
            "ee_replay_rmse_cm": result.ee_replay_rmse_m * 100,
            **(
                {
                    "gaussian_instance": {
                        "instance_id": gaussian_object.instance_id,
                        "n_gaussians": int(len(gaussian_object.gaussian_indices)),
                        "center_world": gaussian_object.center_world().tolist(),
                    }
                }
                if gaussian_object is not None
                else {}
            ),
        },
    )
    twin_path = reporter.dir / "physical_object.json"
    calibrated.save(twin_path)
    print(f"[14] Saved calibrated Physical Gaussian Twin -> {twin_path}")

    if true_params:
        gt = ", ".join(f"{k} = {v}" for k, v in true_params.items()
                       if k in result.optimized_params)
        print(f"      Ground truth (synthetic episode): {gt}")

    simulator.close()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pgtwin calibrate",
        description="Calibrate physics parameters from a real interaction episode.",
    )
    parser.add_argument("--episode", required=True, help="episode directory")
    parser.add_argument("--config", required=True, help="experiment YAML config")
    parser.add_argument("--output", default="outputs", help="output root directory")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    result = run_pipeline(config, args.episode, args.output)

    print("\nCalibration completed")
    for key, value in result.initial_params.items():
        print(f"  Initial {key}: {value:.4f}")
    for key, value in result.optimized_params.items():
        print(f"  Optimized {key}: {value:.4f}")
    print(f"  Initial RMSE: {result.initial_metrics['rmse_position_cm']:.2f} cm")
    print(f"  Final RMSE: {result.final_metrics['rmse_position_cm']:.2f} cm")
    print(f"  Improvement: {result.improvement['rmse_position_cm']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
