"""MuJoCo simulator adapter (Phase 3, validated).

Requires ``pip install mujoco`` (imported lazily so the rest of pgtwin
works without it).  Validated against MuJoCo 3.14: synthetic friction
recovery passes noise-free (exact) and with 1 mm measurement noise
(recovered 0.70-0.75 for truth 0.80 across seeds).

The push scene uses a *compliant* pusher (mocap target + soft weld) and
overdamped contacts — see scene_builder.build_push_scene_mjcf for why a
rigid kinematic pusher makes friction unidentifiable.
"""

from __future__ import annotations

import numpy as np

from pgtwin.common.se3 import Pose
from pgtwin.physics.physical_object import PhysicalObject
from pgtwin.real_data.episode import ObjectState, RobotState
from pgtwin.simulator.base_simulator import BaseSimulator, SimulatorConfig
from pgtwin.simulator.scene_builder import build_push_scene_mjcf

try:  # graceful fallback: never break import of pgtwin.simulator
    import mujoco  # type: ignore

    MUJOCO_AVAILABLE = True
except ImportError:  # pragma: no cover
    mujoco = None
    MUJOCO_AVAILABLE = False


def _require_mujoco() -> None:
    if not MUJOCO_AVAILABLE:
        raise ImportError(
            "MuJoCo backend requested but 'mujoco' is not installed. "
            "Run: pip install mujoco  (or use simulation.backend: mock)"
        )


class MuJoCoSimulator(BaseSimulator):
    """Tabletop push scene in MuJoCo with a compliant mocap-driven EE."""

    def __init__(self, config: SimulatorConfig | None = None, **overrides):
        _require_mujoco()
        super().__init__(config, **overrides)
        self.model = None
        self.data = None
        self._object: PhysicalObject | None = None
        self._initial_pose: Pose | None = None
        self._scene_cfg: dict = {}
        self._ee_teleported = False

    # -- lifecycle -----------------------------------------------------------
    def reset(self) -> None:
        if self.model is None:
            raise RuntimeError("load_object() must be called before reset()")
        mujoco.mj_resetData(self.model, self.data)
        assert self._initial_pose is not None and self._object is not None
        self._ee_teleported = False
        self.set_object_pose(self._object.object_id, self._initial_pose)
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        self.model = None
        self.data = None

    # -- scene setup -----------------------------------------------------------
    def load_scene(self, scene_cfg: dict) -> None:
        self._scene_cfg = dict(scene_cfg)
        self.config.table_height = float(
            scene_cfg.get("table_height", self.config.table_height)
        )

    def load_robot(self, robot_cfg: dict) -> None:
        self.config.ee_radius = float(robot_cfg.get("ee_radius", self.config.ee_radius))

    @classmethod
    def is_available(cls) -> bool:
        return MUJOCO_AVAILABLE

    def load_object(self, obj: PhysicalObject, initial_pose: Pose | None = None) -> None:
        initial_pose = initial_pose or obj.initial_pose
        self._object = obj
        self._initial_pose = initial_pose.copy()
        xml = build_push_scene_mjcf(
            obj,
            initial_pose,
            ee_radius=self.config.ee_radius,
            table_height=self.config.table_height,
            dt=self.config.dt,
            solref=self.config.contact_solref,
            dampratio=self.config.contact_dampratio,
            ee_weld_solref=self.config.ee_weld_solref,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._ee_teleported = False
        mujoco.mj_forward(self.model, self.data)

    # -- parameters / state ------------------------------------------------------
    def set_physics_parameters(self, object_id: str, params: dict[str, float]) -> None:
        self._check_loaded(object_id)
        geom_id = self._object_geom_id()
        body_id = self._object_body_id()
        for key, value in params.items():
            if key in ("friction", "friction_dynamic"):
                self.model.geom_friction[geom_id, 0] = float(value)
            elif key == "mass":
                # scale only this body: inertia scales linearly with mass
                old_mass = float(self.model.body_mass[body_id])
                if old_mass > 0:
                    self.model.body_inertia[body_id] *= float(value) / old_mass
                self.model.body_mass[body_id] = float(value)
            elif key == "restitution":
                # MuJoCo has no direct restitution; map to solref damping.
                # dampratio < 1 -> underdamped contact -> bounce (approximation).
                self.model.geom_solref[geom_id, 1] = 1.0 - 0.5 * float(value)
            else:
                raise KeyError(f"MuJoCoSimulator does not map parameter '{key}'")
        mujoco.mj_forward(self.model, self.data)

    def set_object_pose(self, object_id: str, pose: Pose) -> None:
        self._check_loaded(object_id)
        jnt_adr = self.model.body(self._object.object_id).jntadr[0]
        qpos_adr = self.model.jnt_qposadr[jnt_adr]
        self.data.qpos[qpos_adr : qpos_adr + 3] = pose.position
        x, y, z, w = pose.quaternion
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = [w, x, y, z]  # wxyz
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_ee_pose(self, pose: Pose) -> None:
        """Move the kinematic EE *target*; the physical tip follows through
        the soft weld.  The first target after a reset teleports the tip
        directly onto it (otherwise the weld would drag the tip across the
        scene from its model-default position)."""
        self.data.mocap_pos[0] = pose.position
        x, y, z, w = pose.quaternion
        self.data.mocap_quat[0] = [w, x, y, z]
        if not self._ee_teleported:
            jnt_adr = self.model.body("ee").jntadr[0]
            qpos_adr = self.model.jnt_qposadr[jnt_adr]
            dof_adr = self.model.jnt_dofadr[jnt_adr]
            self.data.qpos[qpos_adr : qpos_adr + 3] = pose.position
            self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = [w, x, y, z]
            self.data.qvel[dof_adr : dof_adr + 6] = 0.0
            self._ee_teleported = True
            mujoco.mj_forward(self.model, self.data)

    def get_object_state(self, object_id: str) -> ObjectState:
        self._check_loaded(object_id)
        body = self.data.body(self._object.object_id)
        w, x, y, z = body.xquat
        return ObjectState(
            timestamp=self.time,
            pose=Pose(position=np.array(body.xpos), quaternion=np.array([x, y, z, w])),
            linear_velocity=np.array(body.cvel[3:6]),
            angular_velocity=np.array(body.cvel[0:3]),
        )

    def get_robot_state(self) -> RobotState:
        """Actual simulated tip pose (not the mocap target): the difference
        between this and the commanded trajectory is the EE replay error."""
        body = self.data.body("ee")
        w, x, y, z = body.xquat
        return RobotState(
            timestamp=self.time,
            ee_pose=Pose(
                position=np.array(body.xpos),
                quaternion=np.array([x, y, z, w]),
            ),
        )

    # -- stepping -----------------------------------------------------------------
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time) if self.data is not None else 0.0

    # -- helpers --------------------------------------------------------------------
    def _check_loaded(self, object_id: str) -> None:
        if self._object is None or self._object.object_id != object_id:
            raise KeyError(f"object '{object_id}' not loaded")

    def _object_body_id(self) -> int:
        return self.model.body(self._object.object_id).id

    def _object_geom_id(self) -> int:
        body_id = self._object_body_id()
        return int(self.model.body_geomadr[body_id])
