"""GaussianScene / instance segmentation / collision proxy tests."""

import numpy as np
import pytest

from pgtwin.common.se3 import Pose
from pgtwin.gaussian.instance_manager import InstanceManager, ManualInstanceSelector
from pgtwin.gaussian.scene_loader import GaussianScene


@pytest.fixture
def tabletop_scene():
    table = GaussianScene.synthetic_tabletop(n_points=500, seed=0)
    box = GaussianScene.synthetic_box(
        center=(0.5, 0.0, 0.03), size=(0.06, 0.06, 0.06), n_points=400, seed=1
    )
    return table.merge(box)


class TestGaussianScene:
    def test_synthetic_box_bounds(self):
        box = GaussianScene.synthetic_box((0.5, 0.0, 0.03), (0.06, 0.06, 0.06), 1000)
        lo, hi = box.bounds_world()
        np.testing.assert_allclose(lo, [0.47, -0.03, 0.0], atol=0.01)
        np.testing.assert_allclose(hi, [0.53, 0.03, 0.06], atol=0.01)

    def test_merge_counts(self, tabletop_scene):
        assert len(tabletop_scene) == 900

    def test_transform_to_world(self):
        box = GaussianScene.synthetic_box((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), 200)
        shifted = box.transform(Pose(position=[1.0, 0.0, 0.0]))
        np.testing.assert_allclose(
            shifted.positions_world().mean(axis=0)[0], 1.0, atol=0.02
        )
        # original untouched
        np.testing.assert_allclose(box.positions_world().mean(axis=0)[0], 0.0, atol=0.02)

    def test_npz_roundtrip(self, tabletop_scene, tmp_path):
        path = tabletop_scene.save(tmp_path / "scene.npz")
        loaded = GaussianScene.load(path)
        assert len(loaded) == len(tabletop_scene)
        np.testing.assert_allclose(loaded.positions, tabletop_scene.positions)

    def test_ply_roundtrip_ascii(self, tmp_path):
        """Minimal PLY reader handles an ASCII xyz cloud."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
        ply = tmp_path / "cloud.ply"
        header = (
            "ply\nformat ascii 1.0\nelement vertex 3\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
        )
        body = "\n".join(" ".join(str(v) for v in row) for row in pts)
        ply.write_text(header + body + "\n")
        scene = GaussianScene.load(ply)
        np.testing.assert_allclose(scene.positions, pts)


class TestInstanceSelection:
    def test_bbox_crop_selects_box_only(self, tabletop_scene):
        selector = ManualInstanceSelector()
        inst = selector.select(
            tabletop_scene,
            "box_01",
            bbox_min=[0.44, -0.06, 0.005],
            bbox_max=[0.56, 0.06, 0.09],
        )
        assert 300 <= len(inst.gaussian_indices) <= 400
        center = inst.center_world()
        np.testing.assert_allclose(center, [0.5, 0.0, 0.03], atol=0.01)

    def test_empty_selection_raises(self, tabletop_scene):
        with pytest.raises(ValueError, match="empty"):
            ManualInstanceSelector().select(
                tabletop_scene, "ghost", bbox_min=[9, 9, 9], bbox_max=[10, 10, 10]
            )

    def test_manager_background(self, tabletop_scene):
        manager = InstanceManager(tabletop_scene)
        inst = manager.segment(
            "box_01", bbox_min=[0.44, -0.06, 0.005], bbox_max=[0.56, 0.06, 0.09]
        )
        bg = manager.background_indices()
        assert len(bg) + len(inst.gaussian_indices) == len(tabletop_scene)


class TestCollisionProxy:
    @pytest.fixture
    def box_instance(self, tabletop_scene):
        return ManualInstanceSelector().select(
            tabletop_scene,
            "box_01",
            bbox_min=[0.44, -0.06, 0.005],
            bbox_max=[0.56, 0.06, 0.09],
        )

    def test_bbox_proxy_size(self, box_instance):
        proxy = box_instance.build_collision_proxy("bbox")
        assert proxy.kind == "box"
        np.testing.assert_allclose(proxy.params["half_extents"], [0.03, 0.03, 0.03],
                                   atol=0.005)

    def test_convex_hull_proxy(self, box_instance):
        proxy = box_instance.build_collision_proxy("convex_hull")
        assert proxy.kind == "convex_hull"
        assert proxy.params["volume"] == pytest.approx(0.06**3, rel=0.25)
        assert proxy.footprint_radius() == pytest.approx(0.042, abs=0.01)

    def test_sphere_proxy(self, box_instance):
        proxy = box_instance.build_collision_proxy("sphere")
        assert proxy.kind == "sphere"
        assert 0.03 < proxy.params["radius"] < 0.06

    def test_to_physical_object_binding(self, box_instance):
        obj = box_instance.to_physical_object(proxy_method="bbox")
        assert obj.object_id == "box_01"
        np.testing.assert_allclose(obj.initial_pose.position, [0.5, 0.0, 0.03], atol=0.01)
        # physics side never references gaussians directly (decoupling)
        assert obj.collision.kind == "box"
