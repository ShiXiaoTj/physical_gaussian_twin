"""Build simulator scenes from configs and physical objects.

For MJCF-based backends this module generates the XML model string; other
backends can consume the same :class:`PhysicalObject` description through
their own adapters.  Keeping generation here (not inside the adapter) means
scene composition logic is backend-independent and unit-testable without
MuJoCo installed.
"""

from __future__ import annotations

from pgtwin.common.se3 import Pose
from pgtwin.physics.physical_object import PhysicalObject


def _geom_for_proxy(obj: PhysicalObject, solref: float = 0.01, dampratio: float = 2.0) -> str:
    """MJCF geom attributes for a collision proxy.

    ``priority="1"`` makes MuJoCo use *this geom's* friction for its
    contacts (instead of the element-wise max with the table/EE geom) —
    without it the table's friction masks the object's and the calibrated
    parameter has no effect on the rollout.
    """
    proxy = obj.collision
    friction = (
        f'{obj.physics.friction_dynamic} 0.005 0.0001'  # slide, torsion, roll
    )
    common = (
        f'friction="{friction}" '
        f'priority="1" '
        f'solref="{solref} {dampratio}" '
        f'rgba="0.8 0.5 0.2 1"'
    )
    if proxy.kind == "box":
        hx, hy, hz = proxy.params["half_extents"]
        return f'<geom type="box" size="{hx} {hy} {hz}" {common}/>'
    if proxy.kind == "sphere":
        return f'<geom type="sphere" size="{proxy.params["radius"]}" {common}/>'
    if proxy.kind == "cylinder":
        r, h = proxy.params["radius"], proxy.params["half_height"]
        return f'<geom type="cylinder" size="{r} {h}" {common}/>'
    if proxy.kind in ("convex_hull", "mesh"):
        # MJCF mesh assets need file export; approximate with the bounding
        # cylinder for Phase 3, full mesh export lands in Phase 6.
        r, h = proxy.footprint_radius(), proxy.half_height()
        return f'<geom type="cylinder" size="{r} {h}" {common}/>'
    raise ValueError(f"unsupported collision proxy '{proxy.kind}'")


def build_push_scene_mjcf(
    obj: PhysicalObject,
    initial_pose: Pose,
    ee_radius: float = 0.012,
    table_height: float = 0.0,
    dt: float = 0.002,
    solref: float = 0.01,
    dampratio: float = 2.0,
    ee_weld_solref: float = 0.04,
) -> str:
    """MJCF for a tabletop push scene: plane + free object + compliant EE.

    The EE is a *compliant* pusher: a mocap body carries the kinematic
    target (``data.mocap_pos``, written by PushReplayController) and the
    physical tip body is welded to it through a soft equality constraint
    (``ee_weld_solref``).  A rigid mocap pusher hammers the object —
    contact-spring surging makes the rollout chaotic in both friction and
    initial pose, burying the calibration signal; the compliant coupling
    reproduces impedance-controlled robot pushing and restores smooth,
    parameter-identifiable dynamics.  The tip's actual pose (not the
    target) is reported as the robot state, so the EE replay error is a
    physically meaningful tracking lag.
    """
    x, y, z = initial_pose.position
    qx, qy, qz, qw = initial_pose.quaternion  # MJCF wants wxyz
    return f"""
<mujoco model="pgtwin_push">
  <option timestep="{dt}" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>
  <worldbody>
    <geom name="table" type="plane" pos="0 0 {table_height}" size="2 2 0.1"
          friction="1.0 0.005 0.0001" rgba="0.9 0.9 0.9 1"/>
    <body name="{obj.object_id}" pos="{x} {y} {z}" quat="{qw} {qx} {qy} {qz}">
      <freejoint name="{obj.object_id}_joint"/>
      <inertial pos="0 0 0" mass="{obj.physics.mass}"
                diaginertia="{_diag_inertia(obj)}"/>
      {_geom_for_proxy(obj, solref, dampratio)}
    </body>
    <body name="ee_target" mocap="true" pos="0 0 0.5">
    </body>
    <body name="ee" pos="0 0 0.5" gravcomp="1">
      <freejoint name="ee_joint"/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="1e-4 1e-4 1e-4"/>
      <!-- priority=2 > object's 1: the EE-object contact always uses this
           fixed friction, so the calibrated object mu only affects the
           object-table contact (clean parameter separation). -->
      <geom name="ee_tip" type="sphere" size="{ee_radius}" priority="2"
            rgba="0.2 0.4 0.9 1" friction="0.6 0.005 0.0001"/>
    </body>
  </worldbody>
  <equality>
    <weld name="ee_weld" body1="ee_target" body2="ee"
          solref="{ee_weld_solref} 1"/>
  </equality>
</mujoco>
"""


def _diag_inertia(obj: PhysicalObject) -> str:
    """Diagonal inertia from the collision proxy (box approximation)."""
    m = obj.physics.mass
    r = obj.collision.footprint_radius()
    h = obj.collision.half_height()
    ixx = m / 12.0 * ((2 * r) ** 2 + (2 * h) ** 2)
    izz = m / 12.0 * (2 * (2 * r) ** 2)
    return f"{max(ixx, 1e-6)} {max(ixx, 1e-6)} {max(izz, 1e-6)}"
