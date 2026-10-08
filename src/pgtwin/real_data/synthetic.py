"""Synthetic episode generation for end-to-end pipeline validation.

Since real robot data may not exist yet, :class:`SyntheticEpisodeGenerator`
manufactures a "real" push episode by rolling out the MockSimulator with a
*ground-truth* friction value, then optionally corrupting the object
trajectory with measurement noise.

The core system test: generate with friction 0.8, calibrate starting from
0.2 — the pipeline must recover ~0.8.  If this fails, nothing downstream
(real simulators, real data) can be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pgtwin.common.logging import get_logger
from pgtwin.common.se3 import Pose, quat_from_axis_angle, quat_multiply
from pgtwin.physics.physical_object import PhysicalObject
from pgtwin.real_data.episode import (
    ActionType,
    EpisodeMetadata,
    InteractionEpisode,
    ObjectState,
    RobotState,
)
from pgtwin.real_data.pose_utils import PoseTrajectory, make_timestamps
from pgtwin.real_data.trajectory_loader import EpisodeLoader
from pgtwin.simulator import create_simulator
from pgtwin.simulator.base_simulator import SimulatorConfig
from pgtwin.simulator.replay import PushReplayController

logger = get_logger("real_data.synthetic")


@dataclass
class SyntheticPushConfig:
    """Configuration for a synthetic straight-line push episode."""

    episode_id: str = "synthetic_push_001"
    object_id: str = "box_01"
    #: episode action tag: "push" or "slide" (same generator; slide = push
    #: with a longer free-slide emphasis, both identify friction)
    action_type: str = "push"
    # ground-truth physics (what calibration must recover)
    true_friction: float = 0.8
    true_mass: float = 0.42
    true_restitution: float = 0.1
    # object initial pose (world frame, on the table)
    object_xy: tuple[float, float] = (0.5, 0.0)
    object_yaw: float = 0.0
    #: full extents (m). Default is a squat box: tip threshold scales with
    #: half_width / com_height = 2.0, so the whole friction grid (<= 1.5)
    #: stays in the sliding regime — tall objects rock/tip at high mu in
    #: full-physics backends, burying the friction signal (see
    #: docs/experiment_protocol.md).
    object_size: tuple[float, float, float] = (0.08, 0.08, 0.04)
    # push profile
    push_direction: tuple[float, float] = (1.0, 0.0)  # unit XY direction
    push_speed: float = 0.5  # m/s (fast push -> long friction-dependent slide)
    push_distance: float = 0.12  # m of EE travel after first contact
    approach_distance: float = 0.10  # m EE starts before contact
    lateral_offset: float = 0.0  # m offset of the push line (excites rotation)
    #: EE pull-back distance after the push (m). Retracting guarantees a
    #: clean separation at a well-defined object velocity; an EE that just
    #: stops leaves the object oscillating on the contact spring in
    #: full-physics backends, making the release state chaotic in mu.
    retract_distance: float = 0.08
    hold_time: float = 2.0  # s of stillness after push (object slides free)
    # recording
    sampling_rate: float = 30.0  # Hz ("real" sensor rate)
    sim_dt: float = 0.002
    noise_std_pos: float = 0.0  # m, gaussian noise on object positions
    noise_std_rot: float = 0.0  # rad, gaussian yaw noise
    seed: int = 0
    #: simulator backend used to roll out the ground-truth physics
    #: ("mock" | "mujoco"); cross-backend generate/calibrate pairs measure
    #: the sim2sim domain gap.
    backend: str = "mock"
    #: extra simulator options (contact_dampratio, ee_weld_solref, ...);
    #: MUST match the calibration-side simulator config.
    sim_options: dict = field(default_factory=dict)
    #: EE push height as a fraction of the object's half-height. 1.0 =
    #: through the center of mass: zero pitch torque from the contact
    #: normal, which minimizes rocking (rocking phase is chaotic in mu and
    #: buries the friction signal in full-physics backends).
    push_height_ratio: float = 1.0
    #: z of the support surface (m): 0 for the floor, ~0.75 for a tabletop
    #: object. Object rest height, EE height and the simulator's contact
    #: plane all follow it.
    table_height: float = 0.0

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


@dataclass
class SyntheticEpisodeGenerator:
    """Generate synthetic push episodes with known ground-truth physics."""

    config: SyntheticPushConfig = field(default_factory=SyntheticPushConfig)

    def build_object(
        self, size: tuple[float, float, float] | None = None
    ) -> PhysicalObject:
        from pgtwin.physics.physical_object import CollisionProxy, PhysicsParameters

        c = self.config
        size = size or c.object_size
        return PhysicalObject(
            object_id=c.object_id,
            collision=CollisionProxy("box", {"half_extents": [s / 2 for s in size]}),
            physics=PhysicsParameters(
                mass=c.true_mass,
                friction_static=max(0.6, c.true_friction),
                friction_dynamic=c.true_friction,
                restitution=c.true_restitution,
            ),
        )

    def build_ee_trajectory(self, obj: PhysicalObject) -> PoseTrajectory:
        """Straight-line push: approach -> constant-velocity push ->
        retract (clean separation) -> hold."""
        c = self.config
        d = np.asarray(c.push_direction, dtype=float)
        d = d / np.linalg.norm(d)
        normal_offset = np.array([-d[1], d[0]]) * c.lateral_offset

        obj_xy = np.asarray(c.object_xy, dtype=float)
        r_contact = obj.collision.footprint_radius()
        ee_radius = SimulatorConfig().ee_radius
        # EE starts approach_distance before touching the object surface
        start_xy = obj_xy - d * (r_contact + ee_radius + c.approach_distance) + normal_offset
        travel = c.approach_distance + c.push_distance
        t_push = travel / c.push_speed
        t_retract = t_push + c.retract_distance / c.push_speed
        total_time = t_retract + c.hold_time
        # push below the object center: raises the tipping threshold while
        # keeping clearance above the table for the EE sphere
        ee_z = c.table_height + max(
            c.push_height_ratio * obj.collision.half_height(), ee_radius + 0.004
        )

        ts = make_timestamps(0.0, total_time, c.sampling_rate)
        positions = np.zeros((len(ts), 3))
        for i, t in enumerate(ts):
            if t <= t_push:  # approach + push
                progress = t * c.push_speed
            elif t <= t_retract:  # retract at the same speed
                progress = travel - (t - t_push) * c.push_speed
            else:  # hold at the retracted position
                progress = travel - c.retract_distance
            xy = start_xy + d * progress
            positions[i] = [xy[0], xy[1], ee_z]
        # EE orientation: tool pointing down (identity is fine for a point pusher)
        quats = np.tile([0.0, 0.0, 0.0, 1.0], (len(ts), 1))
        return PoseTrajectory(ts, positions, quats)

    def generate(
        self,
        save_dir: str | Path | None = None,
        obj: PhysicalObject | None = None,
    ) -> tuple[InteractionEpisode, PhysicalObject]:
        """Roll out ground-truth physics and package as a real-format episode.

        Pass ``obj`` to push an externally-defined object (e.g. one whose
        collision proxy came from real scene gaussians); its ground-truth
        physics are taken from ``config.true_*`` regardless.
        """
        c = self.config
        if obj is None:
            obj = self.build_object()
        else:
            obj = PhysicalObject(
                object_id=obj.object_id,
                collision=obj.collision,
                physics=self.build_object().physics,  # ground-truth values
                gaussian_asset=obj.gaussian_asset,
                initial_pose=obj.initial_pose,
                calibration_info=dict(obj.calibration_info),
            )
        ee_traj = self.build_ee_trajectory(obj)
        initial_pose = Pose.from_xyz_yaw(
            c.object_xy[0], c.object_xy[1],
            c.table_height + obj.collision.half_height(), c.object_yaw,
        )

        sim = create_simulator({"backend": c.backend, "dt": c.sim_dt, **c.sim_options})
        sim.load_scene({"table_height": c.table_height})
        sim.load_robot({})
        sim.load_object(obj, initial_pose)
        replay = PushReplayController(sim)
        result = replay.replay(ee_traj, obj.object_id, settle_time=0.5)

        # resample the dense sim rollout at the "sensor" rate
        obj_traj = result.object_trajectory.resample(
            make_timestamps(ee_traj.start_time, result.object_trajectory.end_time, c.sampling_rate)
        )

        rng = np.random.default_rng(c.seed)
        positions = obj_traj.positions.copy()
        quats = obj_traj.quaternions.copy()
        if c.noise_std_pos > 0:
            positions += rng.normal(0.0, c.noise_std_pos, positions.shape)
        if c.noise_std_rot > 0:
            for i in range(len(quats)):
                dq = quat_from_axis_angle(
                    np.array([0.0, 0.0, 1.0]), rng.normal(0.0, c.noise_std_rot)
                )
                quats[i] = quat_multiply(dq, quats[i])

        metadata = EpisodeMetadata(
            episode_id=c.episode_id,
            action_type=ActionType.parse(c.action_type),
            robot="franka",
            object_id=obj.object_id,
            scene_id="synthetic_tabletop",
            sampling_rate=c.sampling_rate,
            start_time=0.0,
            end_time=float(obj_traj.end_time),
            extra={
                "synthetic": True,
                "ground_truth": {
                    "friction": c.true_friction,
                    "mass": c.true_mass,
                    "restitution": c.true_restitution,
                },
                "generator_config": c.to_dict(),
            },
        )

        robot_states = [
            RobotState(timestamp=float(t), ee_pose=ee_traj.pose_at(float(t)))
            for t in obj_traj.timestamps
        ]
        object_states = [
            ObjectState(timestamp=float(t), pose=Pose(positions[i], quats[i]))
            for i, t in enumerate(obj_traj.timestamps)
        ]
        episode = InteractionEpisode(metadata, robot_states, object_states)
        episode.validate()
        sim.close()

        logger.info(
            "synthetic episode '%s': true friction=%.2f, object moved %.1f cm",
            c.episode_id,
            c.true_friction,
            episode.object_trajectory().total_displacement() * 100,
        )
        if save_dir is not None:
            EpisodeLoader.save(episode, save_dir)
        return episode, obj

    @classmethod
    def from_config(cls, cfg: dict) -> "SyntheticEpisodeGenerator":
        known = {f for f in SyntheticPushConfig.__dataclass_fields__}
        kwargs = {k: v for k, v in cfg.items() if k in known}
        for key in ("object_xy", "push_direction", "object_size"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        return cls(SyntheticPushConfig(**kwargs))


# ---------------------------------------------------------------------------
# LIFT episodes (mass calibration)
# ---------------------------------------------------------------------------

@dataclass
class SyntheticLiftConfig:
    """Configuration for a synthetic grasp-and-lift episode.

    The episode starts already grasped (protocol §2.2): EE at the object
    center, object attached.  The EE rises by ``lift_height`` on a smooth
    cosine profile (bounded acceleration), then holds.  The mass observable
    is the grip force ``F = m (a + g_up)``: ``m g`` during the hold,
    ``m (g + a)`` during the ramp.
    """

    episode_id: str = "synthetic_lift_001"
    object_id: str = "box_01"
    # ground-truth physics
    true_mass: float = 0.42
    true_friction: float = 0.8  # inert during lift; carried for staging demos
    true_restitution: float = 0.1
    # object geometry / start pose
    object_xy: tuple[float, float] = (0.5, 0.0)
    object_size: tuple[float, float, float] = (0.08, 0.08, 0.04)
    # lift profile
    lift_height: float = 0.10  # m
    lift_duration: float = 1.2  # s (cosine ramp)
    hold_time: float = 1.5  # s of steady hold at the top
    # recording
    sampling_rate: float = 100.0  # Hz (force signals want denser sampling)
    sim_dt: float = 0.002
    noise_std_force: float = 0.0  # N, gaussian noise on measured forces
    noise_std_pos: float = 0.0  # m, noise on object positions
    seed: int = 0
    backend: str = "mock"
    sim_options: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()
        }


@dataclass
class SyntheticLiftGenerator:
    """Generate synthetic lift episodes with known ground-truth mass."""

    config: SyntheticLiftConfig = field(default_factory=SyntheticLiftConfig)

    def build_object(self) -> PhysicalObject:
        from pgtwin.physics.physical_object import CollisionProxy, PhysicsParameters

        c = self.config
        return PhysicalObject(
            object_id=c.object_id,
            collision=CollisionProxy(
                "box", {"half_extents": [s / 2 for s in c.object_size]}
            ),
            physics=PhysicsParameters(
                mass=c.true_mass,
                friction_static=max(0.6, c.true_friction),
                friction_dynamic=c.true_friction,
                restitution=c.true_restitution,
            ),
        )

    def build_ee_trajectory(self, obj: PhysicalObject) -> PoseTrajectory:
        """Grasp point at the object center; cosine rise; steady hold."""
        c = self.config
        z0 = obj.collision.half_height()
        total = c.lift_duration + c.hold_time
        ts = make_timestamps(0.0, total, c.sampling_rate)
        positions = np.zeros((len(ts), 3))
        for i, t in enumerate(ts):
            phase = min(t, c.lift_duration) / c.lift_duration
            z = z0 + c.lift_height * 0.5 * (1.0 - np.cos(np.pi * phase))
            positions[i] = [c.object_xy[0], c.object_xy[1], z]
        quats = np.tile([0.0, 0.0, 0.0, 1.0], (len(ts), 1))
        return PoseTrajectory(ts, positions, quats)

    def generate(
        self,
        save_dir: str | Path | None = None,
        obj: PhysicalObject | None = None,
    ) -> tuple[InteractionEpisode, PhysicalObject]:
        c = self.config
        if obj is None:
            obj = self.build_object()
        else:
            obj = PhysicalObject(
                object_id=obj.object_id,
                collision=obj.collision,
                physics=self.build_object().physics,
                gaussian_asset=obj.gaussian_asset,
                initial_pose=obj.initial_pose,
                calibration_info=dict(obj.calibration_info),
            )
        ee_traj = self.build_ee_trajectory(obj)
        initial_pose = Pose.from_xyz_yaw(
            c.object_xy[0], c.object_xy[1], obj.collision.half_height(), 0.0
        )

        sim = create_simulator({"backend": c.backend, "dt": c.sim_dt, **c.sim_options})
        sim.load_scene({})
        sim.load_robot({})
        sim.load_object(obj, initial_pose)
        sim.set_ee_pose(ee_traj.pose(0))
        sim.attach_object(obj.object_id)
        result = PushReplayController(sim).replay(ee_traj, obj.object_id, settle_time=0.0)
        if result.ee_forces is None:
            raise RuntimeError("backend produced no grip forces during lift")

        ts = make_timestamps(0.0, ee_traj.end_time, c.sampling_rate)
        obj_traj = result.object_trajectory.resample(ts)
        forces = np.column_stack(
            [
                np.interp(ts, result.ee_trajectory.timestamps, result.ee_forces[:, k])
                for k in range(3)
            ]
        )

        rng = np.random.default_rng(c.seed)
        if c.noise_std_force > 0:
            forces = forces + rng.normal(0.0, c.noise_std_force, forces.shape)
        positions = obj_traj.positions.copy()
        if c.noise_std_pos > 0:
            positions = positions + rng.normal(0.0, c.noise_std_pos, positions.shape)

        metadata = EpisodeMetadata(
            episode_id=c.episode_id,
            action_type=ActionType.LIFT,
            robot="franka",
            object_id=obj.object_id,
            scene_id="synthetic_tabletop",
            sampling_rate=c.sampling_rate,
            start_time=0.0,
            end_time=float(ts[-1]),
            extra={
                "synthetic": True,
                "ground_truth": {
                    "mass": c.true_mass,
                    "friction": c.true_friction,
                    "restitution": c.true_restitution,
                },
                "generator_config": c.to_dict(),
            },
        )
        robot_states = [
            RobotState(
                timestamp=float(t),
                ee_pose=ee_traj.pose_at(float(t)),
                gripper_state=1.0,
                ee_force=forces[i],
            )
            for i, t in enumerate(ts)
        ]
        object_states = [
            ObjectState(timestamp=float(t), pose=Pose(positions[i], obj_traj.quaternions[i]))
            for i, t in enumerate(ts)
        ]
        episode = InteractionEpisode(metadata, robot_states, object_states)
        episode.validate()
        sim.close()

        logger.info(
            "synthetic lift '%s': true mass=%.3f kg, hold force ~%.2f N",
            c.episode_id,
            c.true_mass,
            c.true_mass * 9.81,
        )
        if save_dir is not None:
            EpisodeLoader.save(episode, save_dir)
        return episode, obj

    @classmethod
    def from_config(cls, cfg: dict) -> "SyntheticLiftGenerator":
        known = {f for f in SyntheticLiftConfig.__dataclass_fields__}
        kwargs = {k: v for k, v in cfg.items() if k in known}
        for key in ("object_xy", "object_size"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        return cls(SyntheticLiftConfig(**kwargs))
