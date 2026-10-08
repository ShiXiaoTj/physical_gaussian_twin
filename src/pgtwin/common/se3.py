"""SE(3) rigid-body math.

Global conventions (see docs/architecture.md, "Coordinate Frame Convention"):

* Right-handed coordinate frames, z-up world frame.
* Quaternions are stored as ``[x, y, z, w]`` (scalar-last, matching
  scipy / ROS / MuJoCo's xyzw order used by pgtwin CSV formats).
* A :class:`Pose` named ``T_a_b`` maps points from frame ``b`` to frame
  ``a``:  ``p_a = T_a_b * p_b``.  Composition follows
  ``T_a_c = T_a_b.compose(T_b_c)``.
* All angles are radians unless a function name says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Quaternion primitives (xyzw)
# ---------------------------------------------------------------------------

def quat_identity() -> np.ndarray:
    """Identity quaternion ``[0, 0, 0, 1]``."""
    return np.array([0.0, 0.0, 0.0, 1.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return the unit quaternion; falls back to identity for zero norm."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < _EPS:
        return quat_identity()
    return q / n


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion (== conjugate)."""
    return quat_conjugate(quat_normalize(q))


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product so that ``R(q1 @ q2) = R(q1) @ R(q2)``."""
    x1, y1, z1, w1 = np.asarray(q1, dtype=float)
    x2, y2, z2, w2 = np.asarray(q2, dtype=float)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by unit quaternion ``q``."""
    return quat_to_matrix(q) @ np.asarray(v, dtype=float)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (xyzw) -> 3x3 rotation matrix."""
    x, y, z, w = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion (xyzw), Shepperd's method."""
    m = np.asarray(rot, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([x, y, z, w]))


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), _EPS)
    half = 0.5 * angle
    return np.concatenate([axis * np.sin(half), [np.cos(half)]])


def quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation, shortest arc, ``alpha`` in [0, 1]."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # take shortest path
        q1 = -q1
        dot = -dot
    if dot > 1.0 - 1e-8:  # nearly identical: nlerp
        return quat_normalize(q0 + alpha * (q1 - q0))
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    return quat_normalize(
        (np.sin((1.0 - alpha) * theta) / s) * q0 + (np.sin(alpha * theta) / s) * q1
    )


def quat_angular_distance(q0: np.ndarray, q1: np.ndarray) -> float:
    """Geodesic angle (radians, in [0, pi]) between two orientations."""
    dot = abs(float(np.dot(quat_normalize(q0), quat_normalize(q1))))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def ensure_quat_continuity(quats: np.ndarray) -> np.ndarray:
    """Flip signs along a quaternion sequence so consecutive dots are >= 0."""
    out = np.array(quats, dtype=float, copy=True)
    for i in range(1, len(out)):
        if np.dot(out[i - 1], out[i]) < 0.0:
            out[i] = -out[i]
    return out


def yaw_to_quat(yaw: float) -> np.ndarray:
    """Planar yaw about world z -> quaternion (xyzw)."""
    return np.array([0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw)])


def quat_to_yaw(q: np.ndarray) -> float:
    """Extract yaw about z (valid for near-planar rotations)."""
    x, y, z, w = quat_normalize(q)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


# ---------------------------------------------------------------------------
# Pose (SE(3))
# ---------------------------------------------------------------------------

@dataclass
class Pose:
    """Rigid transform: rotation (unit quaternion, xyzw) + translation (m)."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=quat_identity)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.quaternion = quat_normalize(
            np.asarray(self.quaternion, dtype=float).reshape(4)
        )

    # -- constructors -------------------------------------------------------
    @classmethod
    def identity(cls) -> "Pose":
        return cls()

    @classmethod
    def from_matrix(cls, mat: np.ndarray) -> "Pose":
        mat = np.asarray(mat, dtype=float)
        if mat.shape != (4, 4):
            raise ValueError(f"expected 4x4 homogeneous matrix, got {mat.shape}")
        return cls(position=mat[:3, 3], quaternion=matrix_to_quat(mat[:3, :3]))

    @classmethod
    def from_xyz_yaw(cls, x: float, y: float, z: float, yaw: float = 0.0) -> "Pose":
        return cls(position=np.array([x, y, z]), quaternion=yaw_to_quat(yaw))

    # -- conversions --------------------------------------------------------
    def to_matrix(self) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, :3] = quat_to_matrix(self.quaternion)
        mat[:3, 3] = self.position
        return mat

    @property
    def rotation_matrix(self) -> np.ndarray:
        return quat_to_matrix(self.quaternion)

    @property
    def yaw(self) -> float:
        return quat_to_yaw(self.quaternion)

    # -- group operations ---------------------------------------------------
    def compose(self, other: "Pose") -> "Pose":
        """``T_a_c = T_a_b.compose(T_b_c)`` where ``self`` is ``T_a_b``."""
        return Pose(
            position=self.position + quat_rotate(self.quaternion, other.position),
            quaternion=quat_multiply(self.quaternion, other.quaternion),
        )

    def inverse(self) -> "Pose":
        q_inv = quat_inverse(self.quaternion)
        return Pose(position=-quat_rotate(q_inv, self.position), quaternion=q_inv)

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Map a point expressed in ``self``'s child frame into its parent."""
        return self.position + quat_rotate(self.quaternion, point)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        return pts @ self.rotation_matrix.T + self.position

    # -- metrics / interpolation -------------------------------------------
    def interpolate(self, other: "Pose", alpha: float) -> "Pose":
        """Linear position + SLERP orientation blend, ``alpha`` in [0, 1]."""
        return Pose(
            position=(1.0 - alpha) * self.position + alpha * other.position,
            quaternion=quat_slerp(self.quaternion, other.quaternion, alpha),
        )

    def position_distance(self, other: "Pose") -> float:
        """Euclidean translation distance in meters."""
        return float(np.linalg.norm(self.position - other.position))

    def rotation_distance(self, other: "Pose") -> float:
        """Geodesic rotation distance in radians."""
        return quat_angular_distance(self.quaternion, other.quaternion)

    def copy(self) -> "Pose":
        return Pose(position=self.position.copy(), quaternion=self.quaternion.copy())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        p = np.array2string(self.position, precision=4, suppress_small=True)
        q = np.array2string(self.quaternion, precision=4, suppress_small=True)
        return f"Pose(p={p}, q_xyzw={q})"


def relative_transform(a: Pose, b: Pose) -> Pose:
    """Transform ``T`` such that ``a.compose(T) == b`` (i.e. ``a^-1 * b``)."""
    return a.inverse().compose(b)
