"""Integration tests on the real downloaded 3DGS classroom scene.

Skipped when the scene assets are absent (they are large and gitignored;
see configs/experiments/push_desk_real_scene.yaml header for how to
download and prepare them).
"""

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_NPZ = PROJECT_ROOT / "data" / "gaussian" / "classroom_scene.npz"
RAW_PLY = PROJECT_ROOT / "data" / "gaussian" / "classroom_model.ply"
RAW_SPLAT = PROJECT_ROOT / "data" / "gaussian" / "classroom.splat"

INSTANCE_BBOX_MIN = [0.55, 1.80, -0.05]
INSTANCE_BBOX_MAX = [1.95, 2.75, 1.40]


@pytest.mark.skipif(not SCENE_NPZ.exists(), reason="prepared classroom scene absent")
class TestRealSceneNPZ:
    @pytest.fixture(scope="class")
    def scene(self):
        from pgtwin.gaussian.scene_loader import GaussianScene

        return GaussianScene.load(SCENE_NPZ)

    def test_scene_is_metric_and_grounded(self, scene):
        pts = scene.positions_world()
        assert 100_000 < len(scene) < 2_500_000
        # ground (desk plane) at z=0, room content above, within ~4 m
        assert -1.2 < pts[:, 2].min() < 0.5
        assert 2.0 < pts[:, 2].max() < 4.5
        # footprint of a classroom: meters, not arbitrary units
        extent = pts.max(axis=0) - pts.min(axis=0)
        assert 5.0 < extent[0] < 15.0
        assert 5.0 < extent[1] < 15.0

    def test_instance_segmentation(self, scene):
        from pgtwin.gaussian.instance_manager import InstanceManager

        inst = InstanceManager(scene).segment(
            "desk_01", bbox_min=INSTANCE_BBOX_MIN, bbox_max=INSTANCE_BBOX_MAX
        )
        assert len(inst.gaussian_indices) > 1000
        np.testing.assert_allclose(
            inst.center_world()[:2], [1.20, 2.05], atol=0.15
        )

    def test_collision_proxy_from_real_gaussians(self, scene):
        from pgtwin.gaussian.instance_manager import InstanceManager

        inst = InstanceManager(scene).segment(
            "desk_01", bbox_min=INSTANCE_BBOX_MIN, bbox_max=INSTANCE_BBOX_MAX
        )
        bbox = inst.build_collision_proxy("bbox")
        assert 0.3 < bbox.footprint_radius() < 0.8
        assert 0.05 < bbox.half_height() < 0.3
        hull = inst.build_collision_proxy("convex_hull")
        assert hull.kind == "convex_hull"
        assert 0.01 < hull.params["volume"] < 1.0
        # proxies are body-frame: hull vertices centered near origin
        verts = np.asarray(hull.params["vertices"])
        assert np.linalg.norm(verts.mean(axis=0)) < 0.2

    def test_end_to_end_calibration_on_real_instance(self, scene):
        """The Phase 5-6 gate: gaussian instance -> proxy -> synthetic push
        at the instance's true location -> friction recovery."""
        from pgtwin.gaussian.instance_manager import InstanceManager
        from pgtwin.physics.calibrator import PhysicsCalibrator
        from pgtwin.physics.losses import TrajectoryLoss
        from pgtwin.physics.parameter_space import ParameterSpace
        from pgtwin.real_data.synthetic import (
            SyntheticEpisodeGenerator,
            SyntheticPushConfig,
        )
        from pgtwin.simulator import create_simulator

        inst = InstanceManager(scene).segment(
            "desk_01", bbox_min=INSTANCE_BBOX_MIN, bbox_max=INSTANCE_BBOX_MAX
        )
        obj = inst.to_physical_object(proxy_method="bbox")
        center = inst.center_world()
        gen = SyntheticEpisodeGenerator(SyntheticPushConfig(
            episode_id="real_scene_gate",
            object_id="desk_01",
            true_friction=0.5,
            true_mass=3.0,
            object_xy=(float(center[0]), float(center[1])),
            hold_time=1.5,
        ))
        episode, obj = gen.generate(obj=obj)

        sim = create_simulator({"backend": "mock", "dt": 0.002})
        cal = PhysicsCalibrator(sim, loss=TrajectoryLoss(), settle_time=0.5)
        space = ParameterSpace.from_config(
            {"friction": {"min": 0.05, "max": 1.5, "step": 0.05, "initial": 0.2}}
        )
        result = cal.calibrate(episode, obj, space)
        sim.close()
        assert result.optimized_params["friction"] == pytest.approx(0.5, abs=0.05)
        assert result.improvement["rmse_position_cm"] > 80.0


@pytest.mark.skipif(not RAW_PLY.exists(), reason="raw classroom PLY absent")
def test_raw_3dgs_ply_parses():
    from pgtwin.gaussian.scene_loader import GaussianScene

    scene = GaussianScene.load(RAW_PLY, max_points=50_000)
    assert len(scene) == 50_000
    assert scene.scales is not None and scene.rotations is not None
    assert scene.opacities is not None and scene.colors is not None
    # activations applied: opacity in [0,1], scales positive metric-ish
    assert 0.0 <= scene.opacities.min() and scene.opacities.max() <= 1.0
    assert (scene.scales > 0).all()


@pytest.mark.skipif(not RAW_SPLAT.exists(), reason="classroom .splat absent")
def test_raw_splat_parses():
    from pgtwin.gaussian.scene_loader import GaussianScene

    scene = GaussianScene.load(RAW_SPLAT, max_points=50_000)
    assert len(scene) == 50_000
    assert 0.0 <= scene.opacities.min() and scene.opacities.max() <= 1.0
    norms = np.linalg.norm(scene.rotations, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)  # loader renormalizes
