#!/usr/bin/env python3
"""Hierarchical (staged) physics calibration demo: LIFT -> mass, PUSH -> friction.

    python scripts/run_hierarchical.py --example lift_push
    python scripts/run_hierarchical.py --lift data/real/lift_001 \
        --push data/real/push_001 --config configs/experiments/lift_push_hierarchical.yaml

Each stage calibrates only the parameter its action identifies, using the
observable that carries it (LIFT: grip force; PUSH: object trajectory), and
threads its optimum into the next stage — never a joint black-box fit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pgtwin.common.io import load_config, save_json  # noqa: E402
from pgtwin.evaluation.report import ExperimentReporter  # noqa: E402
from pgtwin.physics.calibrator import HierarchicalCalibrator  # noqa: E402
from pgtwin.physics.estimators import estimate_mass_from_lift  # noqa: E402
from pgtwin.physics.losses import TrajectoryLoss  # noqa: E402
from pgtwin.physics.optimizer import create_optimizer  # noqa: E402
from pgtwin.physics.parameter_space import ParameterSpace  # noqa: E402
from pgtwin.physics.physical_object import PhysicalObject  # noqa: E402
from pgtwin.real_data.synthetic import (  # noqa: E402
    SyntheticEpisodeGenerator,
    SyntheticLiftGenerator,
)
from pgtwin.real_data.trajectory_loader import EpisodeLoader  # noqa: E402
from pgtwin.simulator import create_simulator  # noqa: E402

CONFIG = PROJECT_ROOT / "configs" / "experiments" / "lift_push_hierarchical.yaml"


def ensure_synthetic_episodes(config: dict, force: bool = False):
    """Generate the lift + push episode pair from the config's ground truth."""
    syn = config.get("synthetic", {})
    obj_cfg = config.get("object", {})
    common = {
        "object_id": obj_cfg.get("object_id", "box_01"),
        "object_size": obj_cfg.get("size", [0.08, 0.08, 0.04]),
        "true_mass": float(syn.get("true_mass", 0.42)),
        "true_friction": float(syn.get("true_friction", 0.8)),
    }
    lift_dir = PROJECT_ROOT / "data" / "generated" / "lift_box_001"
    push_dir = PROJECT_ROOT / "data" / "generated" / "push_box_hier_001"

    if force or not (lift_dir / "metadata.json").exists():
        gen = SyntheticLiftGenerator.from_config(
            {**common, **syn.get("lift", {}), "episode_id": "lift_box_001"}
        )
        gen.generate(save_dir=lift_dir)
    if force or not (push_dir / "metadata.json").exists():
        gen = SyntheticEpisodeGenerator.from_config(
            {**common, **syn.get("push", {}), "episode_id": "push_box_hier_001"}
        )
        gen.generate(save_dir=push_dir)
    return lift_dir, push_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--example", choices=["lift_push"],
                        help="generate synthetic lift+push episodes and run")
    parser.add_argument("--lift", help="LIFT episode directory")
    parser.add_argument("--push", help="PUSH episode directory")
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.example:
        lift_dir, push_dir = ensure_synthetic_episodes(config, args.regenerate)
    elif args.lift and args.push:
        lift_dir, push_dir = Path(args.lift), Path(args.push)
    else:
        parser.error("provide --example lift_push, or both --lift and --push")

    experiment_id = config.get("experiment_id", "hierarchical_001")
    resample = float(config.get("evaluation", {}).get("resample_rate", 30.0))

    print(f"[1] Loading episodes:\n      lift: {lift_dir}\n      push: {push_dir}")
    lift_ep = EpisodeLoader.load(lift_dir)
    push_ep = EpisodeLoader.load(push_dir)

    obj_cfg = dict(config.get("object", {}))
    obj_cfg.setdefault("object_id", lift_ep.metadata.object_id)
    obj = PhysicalObject.from_config(obj_cfg)
    print(f"[2] Initial physics guess: mass={obj.physics.mass} kg, "
          f"friction={obj.physics.friction_dynamic}")

    # direct cross-check before search-based calibration
    try:
        est = estimate_mass_from_lift(lift_ep)
        print(f"[3] Direct mass estimate (hold-window F_z/g): {est.mass_kg:.4f} kg")
    except ValueError as exc:
        est = None
        print(f"[3] Direct mass estimate unavailable: {exc}")

    sim = create_simulator(config.get("simulation", {}))
    loss = TrajectoryLoss.from_config(config.get("loss", {}), resample)
    optimizer = create_optimizer(config.get("calibration", {}))
    space = ParameterSpace.from_config(config["calibration"]["parameters"])

    hc = HierarchicalCalibrator(
        sim,
        loss=loss,
        settle_time=float(config.get("simulation", {}).get("settle_time", 0.5)),
        scene_cfg=config.get("scene", {}),
    )
    print(f"[4] Staged calibration over {space.names} "
          f"(LIFT -> mass via grip force, PUSH -> friction via trajectory)")
    results = hc.calibrate_auto([lift_ep, push_ep], obj, space, optimizer)

    calibrated_physics = obj.physics
    stage_records = []
    for i, res in enumerate(results, start=1):
        calibrated_physics = calibrated_physics.update(res.optimized_params)
        params = ", ".join(f"{k}={v:.4f}" for k, v in res.optimized_params.items())
        print(f"[{4 + i}] Stage {i} [{res.action_type}] -> {params}  "
              f"(loss {res.initial_loss:.4f} -> {res.final_loss:.4f})")
        reporter = ExperimentReporter(args.output, f"{experiment_id}/stage{i}_{res.action_type}")
        reporter.finalize(res, config, resample,
                          lift_ep.metadata.extra.get("ground_truth"))
        stage_records.append({
            "stage": i,
            "action": res.action_type,
            "episode": res.episode_id,
            "optimized_params": res.optimized_params,
            "initial_loss": res.initial_loss,
            "final_loss": res.final_loss,
            "input_physics": res.extra.get("stage_input_physics"),
        })

    truth = lift_ep.metadata.extra.get("ground_truth", {})
    calibrated = PhysicalObject(
        object_id=obj.object_id,
        collision=obj.collision,
        physics=calibrated_physics,
        gaussian_asset=obj.gaussian_asset,
        initial_pose=push_ep.initial_object_pose(),
        calibration_info={
            "method": "hierarchical",
            "stages": stage_records,
            **({"direct_mass_estimate_kg": est.mass_kg} if est else {}),
        },
    )
    out_dir = Path(args.output) / experiment_id
    calibrated.save(out_dir / "physical_object.json")
    save_json({"stages": stage_records, "ground_truth": truth},
              out_dir / "hierarchical_summary.json")

    print(f"[{5 + len(results)}] Saved calibrated Physical Gaussian Twin -> "
          f"{out_dir / 'physical_object.json'}")
    print("\nHierarchical calibration completed")
    print(f"  mass:     {obj.physics.mass:.3f} -> {calibrated_physics.mass:.4f} kg"
          + (f"  (truth {truth['mass']})" if "mass" in truth else ""))
    print(f"  friction: {obj.physics.friction_dynamic:.3f} -> "
          f"{calibrated_physics.friction_dynamic:.4f}"
          + (f"  (truth {truth['friction']})" if "friction" in truth else ""))
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
