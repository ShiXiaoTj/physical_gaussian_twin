"""Physics parameter calibration from real-sim trajectory discrepancy.

This module is the heart of pgtwin: physics parameter inversion.

    real episode -> simulated replay of the same action
    -> trajectory discrepancy -> parameter search -> calibrated params

Design principles:

1. **Identifiability first.**  Not every parameter is observable from every
   action.  :class:`ParameterCalibrationPolicy` maps each action type to
   the parameters it can excite, and the Phase-1 policy restricts PUSH
   episodes to friction only (mass fixed) to avoid friction/mass coupling.
2. **Replay fidelity is audited.**  Every calibration reports the EE
   replay error so object mismatch is never silently mis-attributed to
   physics when the action itself failed to reproduce.
3. **Staged, not joint.**  :class:`HierarchicalCalibrator` calibrates
   parameters sequentially (mass -> friction -> restitution -> stiffness),
   each stage using the action type that best identifies its parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pgtwin.common.logging import get_logger
from pgtwin.evaluation.metrics import improvement_percent, trajectory_metrics
from pgtwin.physics.losses import ForceLoss, TrajectoryLoss
from pgtwin.physics.optimizer import BaseOptimizer, GridSearchOptimizer, OptimizationResult
from pgtwin.physics.parameter_space import ParameterSpace
from pgtwin.physics.physical_object import PhysicalObject
from pgtwin.physics.sensitivity import (
    observable_for_action,
    parameters_for_action,
    suggest_stages,
    validate_identifiability,
)
from pgtwin.real_data.episode import ActionType, InteractionEpisode
from pgtwin.real_data.pose_utils import PoseTrajectory
from pgtwin.simulator.base_simulator import BaseSimulator
from pgtwin.simulator.replay import PushReplayController, ReplayResult

logger = get_logger("physics.calibrator")


class ParameterCalibrationPolicy:
    """Selects which parameters an episode's action may calibrate.

    Backed by :mod:`pgtwin.physics.sensitivity`: parameters the action is
    insensitive to are dropped (kept fixed), and coupled parameter pairs
    (e.g. friction+mass under PUSH) are rejected as non-identifiable.
    """

    def calibratable_parameters(self, action: ActionType) -> list[str]:
        return parameters_for_action(action)

    def restrict(self, action: ActionType, space: ParameterSpace) -> ParameterSpace:
        allowed = self.calibratable_parameters(action)
        names = [n for n in space.names if n in allowed]
        dropped = [n for n in space.names if n not in allowed]
        if dropped:
            logger.warning(
                "action '%s' cannot identify %s (sensitivity policy); "
                "keeping them fixed at initial values",
                action.value,
                dropped,
            )
        if not names:
            raise ValueError(
                f"action '{action.value}' identifies none of {space.names}; "
                f"calibratable: {allowed}"
            )
        # reject jointly non-identifiable combinations (friction+mass, ...)
        validate_identifiability(action, names)
        return space.subspace(names)


# ---------------------------------------------------------------------------
# Calibration result
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Full record of one calibration run (serializable via ``to_dict``)."""

    episode_id: str
    action_type: str
    initial_params: dict[str, float]
    optimized_params: dict[str, float]
    initial_loss: float
    final_loss: float
    optimization: OptimizationResult
    initial_metrics: dict[str, float]
    final_metrics: dict[str, float]
    improvement: dict[str, float]
    ee_replay_rmse_m: float
    # trajectories for reporting / plotting
    real_object: PoseTrajectory | None = None
    sim_before: PoseTrajectory | None = None
    sim_after: PoseTrajectory | None = None
    real_ee: PoseTrajectory | None = None
    sim_ee: PoseTrajectory | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "action_type": self.action_type,
            "initial_params": self.initial_params,
            "optimized_params": self.optimized_params,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "n_evaluations": self.optimization.n_evaluations,
            "initial_metrics": self.initial_metrics,
            "final_metrics": self.final_metrics,
            "improvement": self.improvement,
            "ee_replay_rmse_m": self.ee_replay_rmse_m,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class PhysicsCalibrator:
    """Grid/random/Bayesian calibration of object physics against one episode.

    Usage::

        calibrator = PhysicsCalibrator(sim, loss)
        result = calibrator.calibrate(episode, physical_object, space)
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        loss: TrajectoryLoss | None = None,
        policy: ParameterCalibrationPolicy | None = None,
        settle_time: float = 0.5,
        scene_cfg: dict | None = None,
        robot_cfg: dict | None = None,
        signal: str = "auto",
    ):
        """``signal`` selects the discrepancy observable: ``object_pose``
        (trajectory loss), ``ee_force`` (grip-force loss, the LIFT
        observable) or ``auto`` (dispatch per episode via
        :func:`pgtwin.physics.sensitivity.observable_for_action`)."""
        if signal not in ("auto", "object_pose", "ee_force"):
            raise ValueError(f"unknown signal '{signal}'")
        self.sim = simulator
        self.loss = loss or TrajectoryLoss()
        self.force_loss = ForceLoss(resample_rate=self.loss.resample_rate)
        self.policy = policy or ParameterCalibrationPolicy()
        self.settle_time = settle_time
        self.scene_cfg = scene_cfg or {}
        self.robot_cfg = robot_cfg or {}
        self.signal = signal
        self.replay = PushReplayController(simulator, mode="kinematic")

    def _signal_for(self, episode: InteractionEpisode) -> str:
        if self.signal != "auto":
            return self.signal
        return observable_for_action(episode.action_type)

    @staticmethod
    def _episode_forces(episode: InteractionEpisode) -> tuple[np.ndarray, np.ndarray]:
        forces = [s.ee_force for s in episode.robot_states]
        if any(f is None for f in forces):
            raise ValueError(
                "episode carries no complete ee_force data, but the selected "
                "observable is 'ee_force' (LIFT calibration needs fx,fy,fz)"
            )
        ts = np.array([s.timestamp for s in episode.robot_states])
        return ts, np.stack(forces)

    # -- single rollout ---------------------------------------------------------
    def simulate(
        self,
        episode: InteractionEpisode,
        obj: PhysicalObject,
        params: dict[str, float],
    ) -> ReplayResult:
        """Reset -> place object at the episode's initial pose -> set params
        -> (grasp for LIFT) -> replay the real EE trajectory."""
        self.sim.reset()
        self.sim.set_object_pose(obj.object_id, episode.initial_object_pose())
        self.sim.set_physics_parameters(obj.object_id, params)
        if episode.action_type is ActionType.LIFT:
            # LIFT episodes start already grasped (protocol §2.2): attach
            # the object so the grip-force observable is generated.
            self.sim.set_ee_pose(episode.ee_trajectory().pose(0))
            self.sim.attach_object(obj.object_id)
        return self.replay.replay(
            episode.ee_trajectory(), obj.object_id, settle_time=self.settle_time
        )

    def _setup_scene(self, episode: InteractionEpisode, obj: PhysicalObject) -> None:
        self.sim.load_scene(self.scene_cfg)
        self.sim.load_robot(self.robot_cfg)
        self.sim.load_object(obj, episode.initial_object_pose())

    # -- main entry ----------------------------------------------------------------
    def calibrate(
        self,
        episode: InteractionEpisode,
        obj: PhysicalObject,
        space: ParameterSpace,
        optimizer: BaseOptimizer | None = None,
    ) -> CalibrationResult:
        optimizer = optimizer or GridSearchOptimizer()
        episode.validate()
        space = self.policy.restrict(episode.action_type, space)
        signal = self._signal_for(episode)
        self._setup_scene(episode, obj)

        real_obj = episode.object_trajectory()
        real_ee = episode.ee_trajectory()
        initial_params = space.initial_params()

        if signal == "ee_force":
            real_force_ts, real_forces = self._episode_forces(episode)

            def rollout_loss(rollout: ReplayResult) -> float:
                if rollout.ee_forces is None:
                    raise RuntimeError(
                        "simulator produced no grip forces; does the backend "
                        "implement attach_object()?"
                    )
                return self.force_loss(
                    real_force_ts,
                    real_forces,
                    rollout.ee_trajectory.timestamps,
                    rollout.ee_forces,
                )

            def rollout_metrics(rollout: ReplayResult) -> dict[str, float]:
                return {"force_mae_n": rollout_loss(rollout)}

        else:

            def rollout_loss(rollout: ReplayResult) -> float:
                return self.loss(real_obj, rollout.object_trajectory)

            def rollout_metrics(rollout: ReplayResult) -> dict[str, float]:
                return trajectory_metrics(real_obj, rollout.object_trajectory)

        # -- baseline (uncalibrated) rollout
        logger.info(
            "baseline rollout with initial params %s (signal=%s)",
            initial_params,
            signal,
        )
        before = self.simulate(episode, obj, initial_params)
        initial_loss = rollout_loss(before)
        initial_metrics = rollout_metrics(before)

        # -- search
        def objective(params: dict[str, float]) -> float:
            return rollout_loss(self.simulate(episode, obj, params))

        opt_result = optimizer.optimize(objective, space)

        # -- calibrated rollout
        after = self.simulate(episode, obj, opt_result.best_params)
        final_loss = rollout_loss(after)
        final_metrics = rollout_metrics(after)
        improvement = improvement_percent(initial_metrics, final_metrics)
        ee_replay_rmse = after.ee_replay_rmse()

        logger.info(
            "calibration done: %s -> %s | loss %.6f -> %.6f | EE replay RMSE %.2e m",
            initial_params,
            opt_result.best_params,
            initial_loss,
            final_loss,
            ee_replay_rmse,
        )

        return CalibrationResult(
            episode_id=episode.metadata.episode_id,
            action_type=episode.action_type.value,
            initial_params=initial_params,
            optimized_params=opt_result.best_params,
            initial_loss=float(initial_loss),
            final_loss=float(final_loss),
            optimization=opt_result,
            initial_metrics=initial_metrics,
            final_metrics=final_metrics,
            improvement=improvement,
            ee_replay_rmse_m=ee_replay_rmse,
            real_object=real_obj,
            sim_before=before.object_trajectory,
            sim_after=after.object_trajectory,
            real_ee=real_ee,
            sim_ee=after.ee_trajectory,
        )


class HierarchicalCalibrator:
    """Staged multi-parameter calibration (Phase 8).

    Runs one :class:`PhysicsCalibrator` stage per episode in
    identifiability order (LIFT -> mass, PUSH/SLIDE -> friction,
    COLLISION -> restitution, PRESS -> stiffness), feeding each stage's
    optimum into the next so parameters are never jointly optimized from
    a single ambiguous action.

    Two entry styles::

        hc = HierarchicalCalibrator(sim, settle_time=0.5)
        # explicit stages: [(episode, space_or_config), ...]
        results = hc.calibrate([(push_ep, {"friction": {...}})], obj)
        # automatic staging from a mixed episode pool + one joint space
        results = hc.calibrate_auto(episodes, obj, joint_space)
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        loss: TrajectoryLoss | None = None,
        settle_time: float = 0.5,
        scene_cfg: dict | None = None,
        robot_cfg: dict | None = None,
    ):
        self.calibrator = PhysicsCalibrator(
            simulator,
            loss=loss,
            settle_time=settle_time,
            scene_cfg=scene_cfg,
            robot_cfg=robot_cfg,
        )

    @staticmethod
    def _as_space(space: "ParameterSpace | dict") -> ParameterSpace:
        return space if isinstance(space, ParameterSpace) else ParameterSpace.from_config(space)

    def calibrate(
        self,
        stages: list[tuple[InteractionEpisode, "ParameterSpace | dict"]],
        obj: PhysicalObject,
        optimizer: BaseOptimizer | None = None,
    ) -> list[CalibrationResult]:
        """Run explicit (episode, space) stages, threading results through."""
        results: list[CalibrationResult] = []
        current_obj = obj
        for episode, raw_space in stages:
            result = self.calibrator.calibrate(
                episode, current_obj, self._as_space(raw_space), optimizer
            )
            # record which physics the stage started from (provenance)
            result.extra["stage_input_physics"] = current_obj.physics.to_dict()
            results.append(result)
            current_obj = PhysicalObject(
                object_id=current_obj.object_id,
                collision=current_obj.collision,
                physics=current_obj.physics.update(result.optimized_params),
                gaussian_asset=current_obj.gaussian_asset,
                initial_pose=current_obj.initial_pose,
                calibration_info=current_obj.calibration_info,
            )
        return results

    def calibrate_auto(
        self,
        episodes: list[InteractionEpisode],
        obj: PhysicalObject,
        space: ParameterSpace,
        optimizer: BaseOptimizer | None = None,
    ) -> list[CalibrationResult]:
        """Plan stages from available episodes via the sensitivity table."""
        by_action: dict[ActionType, InteractionEpisode] = {}
        for ep in episodes:
            by_action.setdefault(ep.action_type, ep)

        stages: list[tuple[InteractionEpisode, ParameterSpace]] = []
        for action, params in suggest_stages(list(by_action)):
            names = [p for p in params if p in space]
            if not names:
                logger.warning(
                    "skipping stage %s -> %s: parameters not in search space",
                    action.value,
                    params,
                )
                continue
            stages.append((by_action[action], space.subspace(names)))
        if not stages:
            raise ValueError(
                "no runnable calibration stage: episode actions "
                f"{[a.value for a in by_action]} do not cover parameters {space.names}"
            )
        return self.calibrate(stages, obj, optimizer)
