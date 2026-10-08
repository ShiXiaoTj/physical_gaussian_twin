"""SE(3) / quaternion math tests: identities, composition, round-trips."""

import numpy as np
import pytest

from pgtwin.common.se3 import (
    Pose,
    matrix_to_quat,
    quat_angular_distance,
    quat_from_axis_angle,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_slerp,
    quat_to_matrix,
    quat_to_yaw,
    relative_transform,
    yaw_to_quat,
)


def random_pose(rng: np.random.Generator) -> Pose:
    axis = rng.normal(size=3)
    angle = rng.uniform(-np.pi, np.pi)
    return Pose(
        position=rng.uniform(-2, 2, 3),
        quaternion=quat_from_axis_angle(axis, angle),
    )


class TestQuaternion:
    def test_normalize(self):
        q = quat_normalize([0.0, 0.0, 0.0, 2.0])
        np.testing.assert_allclose(q, [0, 0, 0, 1])

    def test_multiply_identity(self, rng):
        q = quat_from_axis_angle(rng.normal(size=3), 1.2)
        np.testing.assert_allclose(quat_multiply(q, [0, 0, 0, 1]), q, atol=1e-12)

    def test_multiply_matches_matrix_product(self, rng):
        q1 = quat_from_axis_angle(rng.normal(size=3), 0.7)
        q2 = quat_from_axis_angle(rng.normal(size=3), -1.1)
        np.testing.assert_allclose(
            quat_to_matrix(quat_multiply(q1, q2)),
            quat_to_matrix(q1) @ quat_to_matrix(q2),
            atol=1e-10,
        )

    def test_matrix_roundtrip(self, rng):
        for _ in range(50):
            q = quat_from_axis_angle(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
            q2 = matrix_to_quat(quat_to_matrix(q))
            assert quat_angular_distance(q, q2) < 1e-8

    def test_rotate_z90(self):
        q = yaw_to_quat(np.pi / 2)
        np.testing.assert_allclose(quat_rotate(q, [1, 0, 0]), [0, 1, 0], atol=1e-12)

    def test_yaw_roundtrip(self):
        for yaw in np.linspace(-3.0, 3.0, 13):
            assert quat_to_yaw(yaw_to_quat(yaw)) == pytest.approx(yaw, abs=1e-10)

    def test_slerp_endpoints_and_midpoint(self):
        q0 = yaw_to_quat(0.0)
        q1 = yaw_to_quat(np.pi / 2)
        np.testing.assert_allclose(quat_slerp(q0, q1, 0.0), q0, atol=1e-12)
        assert quat_angular_distance(quat_slerp(q0, q1, 1.0), q1) < 1e-12
        mid = quat_slerp(q0, q1, 0.5)
        assert quat_to_yaw(mid) == pytest.approx(np.pi / 4, abs=1e-10)

    def test_slerp_shortest_path(self):
        q0 = yaw_to_quat(0.1)
        q1 = -yaw_to_quat(0.2)  # same rotation, opposite sign
        mid = quat_slerp(q0, q1, 0.5)
        assert quat_to_yaw(mid) == pytest.approx(0.15, abs=1e-9)

    def test_angular_distance_units(self):
        assert quat_angular_distance(
            yaw_to_quat(0.0), yaw_to_quat(np.pi / 3)
        ) == pytest.approx(np.pi / 3, abs=1e-10)


class TestPose:
    def test_identity_compose(self, rng):
        p = random_pose(rng)
        q = p.compose(Pose.identity())
        assert p.position_distance(q) < 1e-12
        assert p.rotation_distance(q) < 1e-9

    def test_inverse(self, rng):
        for _ in range(20):
            p = random_pose(rng)
            ident = p.compose(p.inverse())
            assert np.linalg.norm(ident.position) < 1e-10
            assert ident.rotation_distance(Pose.identity()) < 1e-8

    def test_matrix_roundtrip(self, rng):
        p = random_pose(rng)
        p2 = Pose.from_matrix(p.to_matrix())
        assert p.position_distance(p2) < 1e-12
        assert p.rotation_distance(p2) < 1e-9

    def test_compose_matches_matrix(self, rng):
        a, b = random_pose(rng), random_pose(rng)
        np.testing.assert_allclose(
            a.compose(b).to_matrix(), a.to_matrix() @ b.to_matrix(), atol=1e-10
        )

    def test_relative_transform(self, rng):
        a, b = random_pose(rng), random_pose(rng)
        t = relative_transform(a, b)
        c = a.compose(t)
        assert c.position_distance(b) < 1e-10
        assert c.rotation_distance(b) < 1e-8

    def test_transform_point(self):
        p = Pose.from_xyz_yaw(1.0, 2.0, 0.0, np.pi / 2)
        np.testing.assert_allclose(p.transform_point([1, 0, 0]), [1, 3, 0], atol=1e-12)

    def test_transform_points_batch(self, rng):
        p = random_pose(rng)
        pts = rng.normal(size=(10, 3))
        batch = p.transform_points(pts)
        for i in range(10):
            np.testing.assert_allclose(batch[i], p.transform_point(pts[i]), atol=1e-12)

    def test_interpolate(self):
        a = Pose.from_xyz_yaw(0, 0, 0, 0.0)
        b = Pose.from_xyz_yaw(2, 0, 0, np.pi / 2)
        mid = a.interpolate(b, 0.5)
        np.testing.assert_allclose(mid.position, [1, 0, 0], atol=1e-12)
        assert mid.yaw == pytest.approx(np.pi / 4, abs=1e-9)

    def test_quaternion_auto_normalized(self):
        p = Pose(position=[0, 0, 0], quaternion=[0, 0, 0, 5.0])
        assert np.linalg.norm(p.quaternion) == pytest.approx(1.0)
