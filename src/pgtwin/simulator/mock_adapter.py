"""MockSimulator: dependency-free planar push physics.

A deliberately simple but *physically meaningful* quasi-static model of
tabletop pushing, sufficient to exercise the full calibration loop:

* The object is a rigid body resting on a plane (z fixed at table +
  half-height; roll/pitch locked; free x, y, yaw).
* The EE is a kinematically driven sphere (position-target replay).
* Contact (disc-disc in XY): while the EE overlaps the object's footprint
  circle, penetration is projected out and the object inherits the EE's
  normal approach velocity (inelastic kinematic contact).  Off-center
  contact transfers angular momentum via the contact impulse.
* Free sliding: Coulomb friction decelerates the object at ``mu * g``
  (and spins down yaw analogously).

Identifiability: with a kinematic pusher, the object's release velocity is
set by the push profile (independent of friction), while its free-sliding
distance is ``v^2 / (2 * mu * g)`` — strictly decreasing in ``mu``.  A push
episode therefore identifies dynamic friction, exactly matching the
Action -> Parameter sensitivity table.  Restitution is stored but inert for
sustained pushes (it matters for COLLISION episodes / real backends).

The model is fully deterministic — required for reproducible calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pgtwin.common.se3 import Pose, quat_to_yaw, yaw_to_quat
from pgtwin.physics.physical_object import PhysicalObject
from pgtwin.real_data.episode import ObjectState, RobotState
from pgtwin.simulator.base_simulator import BaseSimulator, SimulatorConfig

# Fraction of the footprint radius acting as the effective torsional
# friction lever arm (ring-contact approximation).
_TORSION_LEVER = 0.6


@dataclass
class _MockObject:
    object_id: str
    radius: float  # XY footprint radius (m)
    half_height: float  # (m)
    mass: float
    friction: float  # dynamic Coulomb friction vs table
    restitution: float
    init_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    init_yaw: float = 0.0
    # dynamic state (planar)
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))  # xy
    omega: float = 0.0  # yaw rate

    @property
    def inertia(self) -> float:
        """Yaw inertia of a uniform disc: 0.5 * m * r^2."""
        return 0.5 * self.mass * self.radius**2


class MockSimulator(BaseSimulator):
    """Planar quasi-static push simulator (see module docstring)."""

    def __init__(self, config: SimulatorConfig | None = None, **overrides):
        super().__init__(config, **overrides)
        self._objects: dict[str, _MockObject] = {}
        self._ee_pos: np.ndarray = np.array([0.0, 0.0, 0.5])
        self._ee_quat: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])
        self._ee_prev: np.ndarray | None = None
        self._ee_vel: np.ndarray = np.zeros(3)
        self._ee_acc: np.ndarray = np.zeros(3)
        self._ee_init: np.ndarray = self._ee_pos.copy()
        self._attached: str | None = None  # object_id grasped by the EE
        self._attach_offset: np.ndarray = np.zeros(3)
        self._time: float = 0.0

    # -- lifecycle -------------------------------------------------------------
    def reset(self) -> None:
        for obj in self._objects.values():
            obj.pos = obj.init_pos.copy()
            obj.yaw = obj.init_yaw
            obj.vel = np.zeros(2)
            obj.omega = 0.0
        self._ee_pos = self._ee_init.copy()
        self._ee_prev = None
        self._ee_vel = np.zeros(3)
        self._ee_acc = np.zeros(3)
        self._attached = None
        self._time = 0.0

    def close(self) -> None:
        self._objects.clear()

    # -- scene setup -----------------------------------------------------------
    def load_scene(self, scene_cfg: dict) -> None:
        self.config.table_height = float(
            scene_cfg.get("table_height", self.config.table_height)
        )

    def load_robot(self, robot_cfg: dict) -> None:
        # Kinematic EE proxy only; radius may come from the robot config.
        self.config.ee_radius = float(robot_cfg.get("ee_radius", self.config.ee_radius))

    def load_object(self, obj: PhysicalObject, initial_pose: Pose | None = None) -> None:
        initial_pose = initial_pose or obj.initial_pose
        mock = _MockObject(
            object_id=obj.object_id,
            radius=obj.collision.footprint_radius(),
            half_height=obj.collision.half_height(),
            mass=obj.physics.mass,
            friction=obj.physics.friction_dynamic,
            restitution=obj.physics.restitution,
        )
        self._objects[obj.object_id] = mock
        self.set_object_pose(obj.object_id, initial_pose)
        mock.init_pos = mock.pos.copy()
        mock.init_yaw = mock.yaw

    # -- parameters / state ------------------------------------------------------
    def set_physics_parameters(self, object_id: str, params: dict[str, float]) -> None:
        obj = self._get(object_id)
        for key, value in params.items():
            if key in ("friction", "friction_dynamic"):
                obj.friction = float(value)
            elif key == "mass":
                obj.mass = float(value)
            elif key == "restitution":
                obj.restitution = float(value)
            else:
                raise KeyError(f"MockSimulator does not model parameter '{key}'")

    def set_object_pose(self, object_id: str, pose: Pose) -> None:
        obj = self._get(object_id)
        obj.pos = pose.position.copy()
        # rest height on the table (planar model)
        obj.pos[2] = self.config.table_height + obj.half_height
        obj.yaw = quat_to_yaw(pose.quaternion)
        obj.vel = np.zeros(2)
        obj.omega = 0.0

    def set_ee_pose(self, pose: Pose) -> None:
        self._ee_pos = pose.position.copy()
        self._ee_quat = pose.quaternion.copy()
        if self._ee_prev is None:
            self._ee_prev = self._ee_pos.copy()
            self._ee_init = self._ee_pos.copy()

    def get_object_state(self, object_id: str) -> ObjectState:
        obj = self._get(object_id)
        return ObjectState(
            timestamp=self._time,
            pose=Pose(position=obj.pos.copy(), quaternion=yaw_to_quat(obj.yaw)),
            linear_velocity=np.array([obj.vel[0], obj.vel[1], 0.0]),
            angular_velocity=np.array([0.0, 0.0, obj.omega]),
        )

    def get_robot_state(self) -> RobotState:
        return RobotState(
            timestamp=self._time,
            ee_pose=Pose(position=self._ee_pos.copy(), quaternion=self._ee_quat.copy()),
            ee_force=self._grip_force(),
        )

    def _grip_force(self) -> np.ndarray | None:
        """Force the gripper exerts on a grasped object: F = m (a + g_up).

        At rest this is (0, 0, m*g); vertical acceleration adds m*a_z.  None
        when nothing is grasped (free-space motion, no load cell signal)."""
        if self._attached is None:
            return None
        obj = self._objects[self._attached]
        g_up = np.array([0.0, 0.0, self.config.gravity])
        return obj.mass * (self._ee_acc + g_up)

    # -- grasping ----------------------------------------------------------------
    def attach_object(self, object_id: str) -> None:
        obj = self._get(object_id)
        self._attached = object_id
        self._attach_offset = obj.pos - self._ee_pos
        obj.vel = np.zeros(2)
        obj.omega = 0.0

    def detach_object(self, object_id: str) -> None:
        self._get(object_id)
        if self._attached == object_id:
            # inherit the EE's planar velocity at release
            self._objects[object_id].vel = self._ee_vel[:2].copy()
            self._attached = None

    # -- stepping -----------------------------------------------------------------
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self._step_once()

    def _step_once(self) -> None:
        dt = self.config.dt
        g = self.config.gravity
        if self._ee_prev is None:
            self._ee_prev = self._ee_pos.copy()
        ee_vel = (self._ee_pos - self._ee_prev) / dt
        self._ee_acc = (ee_vel - self._ee_vel) / dt
        self._ee_vel = ee_vel
        ee_vel_xy = ee_vel[:2]

        for obj in self._objects.values():
            if self._attached == obj.object_id:
                # kinematic grasp: object rigidly follows the EE
                obj.pos = self._ee_pos + self._attach_offset
                continue
            self._resolve_contact(obj, ee_vel_xy)
            self._apply_friction(obj, g, dt)
            # integrate
            obj.pos[:2] += obj.vel * dt
            obj.yaw += obj.omega * dt

        self._ee_prev = self._ee_pos.copy()
        self._time += dt

    def _ee_in_contact_band(self, obj: _MockObject) -> bool:
        """EE z must be within the object's vertical extent to touch it."""
        z_lo = self.config.table_height - self.config.ee_radius
        z_hi = self.config.table_height + 2.0 * obj.half_height + self.config.ee_radius
        return z_lo <= self._ee_pos[2] <= z_hi

    def _resolve_contact(self, obj: _MockObject, ee_vel_xy: np.ndarray) -> None:
        if not self._ee_in_contact_band(obj):
            return
        delta = obj.pos[:2] - self._ee_pos[:2]
        dist = float(np.linalg.norm(delta))
        min_dist = obj.radius + self.config.ee_radius
        penetration = min_dist - dist
        if penetration <= 0.0 or dist < 1e-9:
            return
        normal = delta / dist  # from EE towards object center

        # positional projection (kinematic pusher cannot be displaced)
        obj.pos[:2] += penetration * normal

        # inelastic normal contact: object matches EE approach speed
        ee_vn = float(np.dot(ee_vel_xy, normal))
        obj_vn = float(np.dot(obj.vel, normal))
        if ee_vn > obj_vn:
            dv_n = ee_vn - obj_vn
            obj.vel = obj.vel + dv_n * normal
            impulse_n = obj.mass * dv_n
            # Coulomb friction at the pusher-object contact: tangential slip
            # transfers a bounded tangential impulse, which both drags the
            # object sideways and spins it (this is what rotates a body under
            # an off-center push; a central push has zero slip -> no spin).
            tangent = np.array([-normal[1], normal[0]])
            v_slip = float(np.dot(ee_vel_xy - obj.vel, tangent))
            impulse_t_max = obj.friction * impulse_n
            impulse_t = float(np.clip(obj.mass * v_slip, -impulse_t_max, impulse_t_max))
            obj.vel = obj.vel + (impulse_t / obj.mass) * tangent
            # torque about the center from the tangential impulse at the
            # contact point r = -radius * normal
            r_contact = -obj.radius * normal
            torque_z = r_contact[0] * impulse_t * tangent[1] - r_contact[1] * impulse_t * tangent[0]
            obj.omega += torque_z / obj.inertia

    @staticmethod
    def _coulomb_decay(speed: float, decel: float, dt: float) -> float:
        return max(0.0, speed - decel * dt)

    def _apply_friction(self, obj: _MockObject, g: float, dt: float) -> None:
        # linear Coulomb sliding friction: a = mu * g
        speed = float(np.linalg.norm(obj.vel))
        if speed > 1e-9:
            new_speed = self._coulomb_decay(speed, obj.friction * g, dt)
            obj.vel *= new_speed / speed
        # torsional friction: alpha = mu * g * lever / (r^2 / 2) ~ ring contact
        if abs(obj.omega) > 1e-9 and obj.radius > 1e-9:
            alpha = obj.friction * g * _TORSION_LEVER / (0.5 * obj.radius)
            new_omega = self._coulomb_decay(abs(obj.omega), alpha, dt)
            obj.omega = np.sign(obj.omega) * new_omega

    @property
    def time(self) -> float:
        return self._time

    def _get(self, object_id: str) -> _MockObject:
        if object_id not in self._objects:
            raise KeyError(
                f"object '{object_id}' not loaded; available: {list(self._objects)}"
            )
        return self._objects[object_id]
