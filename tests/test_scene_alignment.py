"""Scene alignment (RANSAC ground plane, metric scale) and .splat IO tests."""

import numpy as np
import pytest

from pgtwin.common.se3 import Pose, quat_from_axis_angle
from pgtwin.gaussian.alignment import (
    align_scene_to_ground,
    estimate_metric_scale,
    fit_ground_plane,
)
from pgtwin.gaussian.scene_loader import GaussianScene


def tilted_room_scene(seed=0, n_floor=4000, n_stuff=1500):
    """Synthetic 'reconstruction': floor plane + objects, rotated + offset
    into an arbitrary frame (like raw COLMAP/3DGS output)."""
    rng = np.random.default_rng(seed)
    floor = np.column_stack([
        rng.uniform(-2, 2, n_floor),
        rng.uniform(-1.5, 1.5, n_floor),
        rng.normal(0, 0.004, n_floor),
    ])
    stuff = np.column_stack([
        rng.uniform(-1.5, 1.5, n_stuff),
        rng.uniform(-1, 1, n_stuff),
        rng.uniform(0.05, 1.2, n_stuff),
    ])
    pts_world = np.vstack([floor, stuff])
    # arbitrary reconstruction frame: rotate and translate
    T = Pose(
        position=[3.0, -1.0, 2.0],
        quaternion=quat_from_axis_angle([1.0, 0.4, -0.2], 1.1),
    )
    pts_gaussian = T.transform_points(pts_world)
    opac = np.full(len(pts_gaussian), 0.9)
    return GaussianScene(positions=pts_gaussian, opacities=opac), T


class TestGroundPlane:
    def test_fit_finds_floor(self):
        scene, _ = tilted_room_scene()
        plane = fit_ground_plane(scene.positions, scene.opacities,
                                 distance_threshold=0.02)
        assert plane.inlier_ratio > 0.6  # floor dominates

    def test_align_puts_floor_at_z0_up(self):
        scene, _ = tilted_room_scene()
        aligned, plane = align_scene_to_ground(scene, distance_threshold=0.02)
        pts = aligned.positions_world()
        floor_z = np.percentile(pts[:, 2], 30)  # floor points dominate low z
        assert abs(floor_z) < 0.02
        # objects must be ABOVE the floor (normal oriented into the scene)
        assert np.percentile(pts[:, 2], 95) > 0.5
        assert pts[:, 2].min() > -0.1

    def test_metric_scale_applied(self):
        """Doubling metric_scale must exactly double world extents
        (in-plane rotation is arbitrary, so compare ratios, not axes)."""
        scene, _ = tilted_room_scene()
        a1, _ = align_scene_to_ground(scene, metric_scale=1.0, distance_threshold=0.02)
        a2, _ = align_scene_to_ground(scene, metric_scale=2.0, distance_threshold=0.02)
        e1 = a1.positions_world().max(axis=0) - a1.positions_world().min(axis=0)
        e2 = a2.positions_world().max(axis=0) - a2.positions_world().min(axis=0)
        np.testing.assert_allclose(e2, 2.0 * e1, rtol=1e-6)

    def test_alignment_survives_npz_roundtrip(self, tmp_path):
        scene, _ = tilted_room_scene()
        aligned, _ = align_scene_to_ground(scene, metric_scale=1.5,
                                           distance_threshold=0.02)
        path = aligned.save(tmp_path / "scene.npz")
        loaded = GaussianScene.load(path)
        np.testing.assert_allclose(
            loaded.positions_world(), aligned.positions_world(), atol=1e-9
        )
        assert loaded.metric_scale == pytest.approx(1.5)


class TestMetricScale:
    def test_estimate(self):
        # a 1.2 m table edge measured as 0.6 units in the reconstruction
        assert estimate_metric_scale(1.2, 0.6) == pytest.approx(2.0)

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            estimate_metric_scale(1.0, 0.0)


class TestSplatIO:
    def test_roundtrip_synthetic_splat(self, tmp_path):
        """Write a synthetic .splat byte-exact and load it back."""
        rng = np.random.default_rng(0)
        n = 100
        pos = rng.normal(size=(n, 3)).astype("<f4")
        scale = rng.uniform(0.01, 0.05, size=(n, 3)).astype("<f4")
        rgba = rng.integers(0, 256, size=(n, 4), dtype=np.uint8)
        quat_wxyz = rng.normal(size=(n, 4))
        quat_wxyz /= np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
        rot_u8 = np.clip(np.round(quat_wxyz * 128 + 128), 0, 255).astype(np.uint8)

        raw = b""
        for i in range(n):
            raw += pos[i].tobytes() + scale[i].tobytes()
            raw += rgba[i].tobytes() + rot_u8[i].tobytes()
        path = tmp_path / "test.splat"
        path.write_bytes(raw)

        scene = GaussianScene.load(path)
        assert len(scene) == n
        np.testing.assert_allclose(scene.positions, pos, atol=1e-6)
        np.testing.assert_allclose(scene.scales, scale, atol=1e-6)
        np.testing.assert_allclose(scene.opacities, rgba[:, 3] / 255.0, atol=1e-6)
        # quaternion decode: wxyz bytes -> xyzw floats (quantized to 1/128,
        # plus renormalization shift)
        decoded_wxyz = scene.rotations[:, [3, 0, 1, 2]]
        np.testing.assert_allclose(decoded_wxyz, quat_wxyz, atol=2.0 / 128)

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.splat"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            GaussianScene.load(path)

    def test_max_points_subsampling(self, tmp_path):
        raw = b"\x00" * 32 * 50
        path = tmp_path / "zeros.splat"
        path.write_bytes(raw)
        scene = GaussianScene.load(path, max_points=10)
        assert len(scene) == 10


class TestSplatExport:
    def test_save_load_roundtrip(self, tmp_path):
        """save_splat -> load must preserve positions/colors within
        quantization (colors: 1/255, rotations: ~1/128)."""
        from pgtwin.gaussian.splat_io import save_splat

        scene, _ = tilted_room_scene(n_floor=300, n_stuff=100)
        scene.colors = np.tile([0.2, 0.5, 0.8], (len(scene), 1))
        scene.scales = np.full((len(scene), 3), 0.01)
        scene.rotations = np.tile([0.0, 0.0, 0.0, 1.0], (len(scene), 1))
        path = save_splat(scene, tmp_path / "x.splat", bake_world=False)
        loaded = GaussianScene.load(path)
        np.testing.assert_allclose(loaded.positions, scene.positions, atol=1e-6)
        np.testing.assert_allclose(loaded.colors, scene.colors, atol=1.5 / 255)
        np.testing.assert_allclose(loaded.scales, scene.scales, atol=1e-6)

    def test_bake_world_applies_alignment(self, tmp_path):
        """With bake_world=True the file stores world-frame positions."""
        from pgtwin.gaussian.splat_io import save_splat
        from pgtwin.gaussian.alignment import align_scene_to_ground

        scene, _ = tilted_room_scene()
        aligned, _ = align_scene_to_ground(scene, distance_threshold=0.02)
        path = save_splat(aligned, tmp_path / "w.splat", bake_world=True)
        loaded = GaussianScene.load(path)
        np.testing.assert_allclose(
            loaded.positions, aligned.positions_world(), atol=1e-5
        )


def test_plane_rank_skips_dominant_wall():
    """rank=1 must find the floor when a wall carries more points."""
    rng = np.random.default_rng(3)
    wall = np.column_stack([
        np.full(6000, 2.0) + rng.normal(0, 0.004, 6000),
        rng.uniform(-2, 2, 6000),
        rng.uniform(0, 2.5, 6000),
    ])
    floor = np.column_stack([
        rng.uniform(-2, 2, 3000),
        rng.uniform(-2, 2, 3000),
        rng.normal(0, 0.004, 3000),
    ])
    pts = np.vstack([wall, floor])
    p0 = fit_ground_plane(pts, distance_threshold=0.02, rank=0)
    p1 = fit_ground_plane(pts, distance_threshold=0.02, rank=1)
    assert abs(p0.normal[0]) > 0.95      # rank 0 -> the wall (x-normal)
    assert abs(p1.normal[2]) > 0.95      # rank 1 -> the floor (z-normal)
