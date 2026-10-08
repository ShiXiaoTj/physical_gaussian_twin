"""GaussianObject: instance-level Gaussian <-> physics binding (S3 + S7).

A GaussianObject owns a subset of scene Gaussians (visual), an SE(3)
transform, and knows how to derive a collision proxy for physics — the
object-level binding at the heart of the Physical Gaussian Twin.  Rendering
and simulation stay decoupled: physics only ever sees the proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pgtwin.common.logging import get_logger
from pgtwin.common.se3 import Pose
from pgtwin.gaussian.scene_loader import GaussianScene
from pgtwin.physics.physical_object import CollisionProxy, PhysicalObject, PhysicsParameters

logger = get_logger("gaussian.object")


@dataclass
class GaussianObject:
    """One segmented object instance inside a GaussianScene."""

    instance_id: str
    scene: GaussianScene  # sub-scene containing only this instance's gaussians
    gaussian_indices: np.ndarray  # indices into the parent scene
    transform: Pose = field(default_factory=Pose)  # T_world_object
    visual_properties: dict = field(default_factory=dict)

    def positions_world(self) -> np.ndarray:
        return self.scene.positions_world()

    def center_world(self) -> np.ndarray:
        return self.positions_world().mean(axis=0)

    # -- collision proxy -----------------------------------------------------
    def build_collision_proxy(
        self, method: str = "convex_hull", opacity_threshold: float = 0.3
    ) -> CollisionProxy:
        """Derive collision geometry from the gaussian point set.

        methods: ``bbox`` (axis-aligned box), ``convex_hull`` (scipy Qhull),
        ``sphere`` (bounding sphere).  Points with opacity below
        ``opacity_threshold`` are ignored (floaters).

        All proxies are expressed in the **object body frame** (origin at
        the instance centroid), so the same proxy is valid at any pose —
        the world placement comes from ``PhysicalObject.initial_pose``.
        """
        pts = self.positions_world()
        if self.scene.opacities is not None:
            mask = self.scene.opacities >= opacity_threshold
            if mask.sum() >= 8:
                pts = pts[mask]
        center = pts.mean(axis=0)
        local = pts - center

        if method == "bbox":
            half = (local.max(axis=0) - local.min(axis=0)) / 2.0
            return CollisionProxy("box", {"half_extents": half.tolist()})
        if method == "sphere":
            radius = float(np.max(np.linalg.norm(local, axis=1)))
            return CollisionProxy("sphere", {"radius": radius})
        if method == "convex_hull":
            try:
                from scipy.spatial import ConvexHull

                hull = ConvexHull(local)
                vertices = local[hull.vertices]
                return CollisionProxy(
                    "convex_hull",
                    {
                        "vertices": vertices.tolist(),
                        "faces": hull.simplices.tolist(),
                        "volume": float(hull.volume),
                    },
                )
            except Exception as exc:  # degenerate cloud -> bbox fallback
                logger.warning("convex hull failed (%s); falling back to bbox", exc)
                return self.build_collision_proxy("bbox", opacity_threshold)
        raise ValueError(f"unknown collision proxy method '{method}'")

    # -- physics binding --------------------------------------------------------
    def to_physical_object(
        self,
        physics: PhysicsParameters | None = None,
        proxy_method: str = "convex_hull",
        gaussian_asset: str | None = None,
    ) -> PhysicalObject:
        """Bind this visual instance to physics: the S7 output object."""
        return PhysicalObject(
            object_id=self.instance_id,
            collision=self.build_collision_proxy(proxy_method),
            physics=physics or PhysicsParameters(),
            gaussian_asset=gaussian_asset or self.scene.source_path,
            initial_pose=Pose(position=self.center_world()),
        )
