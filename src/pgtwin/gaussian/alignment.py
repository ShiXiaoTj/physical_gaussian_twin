"""Align a raw 3DGS reconstruction to the pgtwin world frame.

A COLMAP/3DGS reconstruction lives in an arbitrary similarity frame:
unknown orientation (often y-down), unknown origin, unknown scale.
Calibration needs metric, z-up world coordinates (docs/architecture.md §3),
so ingestion must estimate ``T_world_gaussian`` and ``metric_scale``:

1. :func:`fit_ground_plane` — RANSAC plane fit over gaussian centers finds
   the dominant support surface (floor / tabletop).
2. :func:`align_scene_to_ground` — rotates the plane normal onto +z, puts
   the plane at ``z = 0`` and centers the XY footprint.
3. Scale cannot be recovered from geometry alone: pass a known physical
   distance via :func:`estimate_metric_scale` (e.g. a measured table edge,
   fiducial spacing, or camera baseline).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pgtwin.common.logging import get_logger
from pgtwin.common.se3 import Pose, matrix_to_quat
from pgtwin.gaussian.scene_loader import GaussianScene

logger = get_logger("gaussian.alignment")


@dataclass
class GroundPlane:
    """Plane ``normal . p = offset`` in the gaussian frame (unit normal)."""

    normal: np.ndarray
    offset: float
    inlier_ratio: float
    n_inliers: int

    def distances(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal - self.offset


def fit_ground_plane(
    points: np.ndarray,
    opacities: np.ndarray | None = None,
    distance_threshold: float = 0.02,
    n_iterations: int = 300,
    min_opacity: float = 0.5,
    max_points: int = 50_000,
    seed: int = 0,
    rank: int = 0,
) -> GroundPlane:
    """RANSAC fit of the dominant plane in a gaussian point set.

    ``distance_threshold`` is in *gaussian-frame units* — for an unscaled
    reconstruction pass roughly (expected floor thickness) / (scene size)
    scaled values, or simply a fraction of the bounding-box diagonal.

    ``rank`` skips the ``rank`` most dominant planes before fitting: in
    indoor scenes a wall can carry more gaussians than the floor, so if the
    aligned preview looks sideways, retry with ``rank=1`` (then 2, ...).

    The returned normal is oriented so that most points lie on its positive
    side (i.e. it points from the floor up into the scene).
    """
    pts = np.asarray(points, dtype=float)
    if opacities is not None:
        mask = np.asarray(opacities).reshape(-1) >= min_opacity
        if mask.sum() > 1000:
            pts = pts[mask]
    rng = np.random.default_rng(seed)
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]

    for level in range(rank + 1):
        if len(pts) < 3:
            raise ValueError("not enough points for plane fitting")
        best: tuple[int, np.ndarray, float] | None = None
        for _ in range(n_iterations):
            idx = rng.choice(len(pts), 3, replace=False)
            p0, p1, p2 = pts[idx]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            offset = float(normal @ p0)
            n_inliers = int(np.sum(np.abs(pts @ normal - offset) < distance_threshold))
            if best is None or n_inliers > best[0]:
                best = (n_inliers, normal, offset)
        assert best is not None
        n_inliers, normal, offset = best
        if level < rank:
            # discard this plane's inliers and fit the next one
            keep = np.abs(pts @ normal - offset) >= distance_threshold
            logger.info(
                "skipping dominant plane %d (normal=%s, %d inliers)",
                level,
                np.round(normal, 3),
                n_inliers,
            )
            pts = pts[keep]

    # least-squares refinement on the inliers
    inliers = pts[np.abs(pts @ normal - offset) < distance_threshold]
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[2] / np.linalg.norm(vh[2])
    offset = float(normal @ centroid)

    # orient the normal towards the bulk of the scene (floor points up)
    if np.mean(pts @ normal - offset) < 0:
        normal, offset = -normal, -offset

    n_inliers = int(np.sum(np.abs(pts @ normal - offset) < distance_threshold))
    plane = GroundPlane(
        normal=normal,
        offset=offset,
        inlier_ratio=n_inliers / len(pts),
        n_inliers=n_inliers,
    )
    logger.info(
        "ground plane: normal=%s, offset=%.4f, inliers=%.1f%%",
        np.round(normal, 3),
        offset,
        100 * plane.inlier_ratio,
    )
    return plane


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix taking unit vector ``a`` onto unit vector ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:  # opposite: rotate pi about any orthogonal axis
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def align_scene_to_ground(
    scene: GaussianScene,
    plane: GroundPlane | None = None,
    metric_scale: float = 1.0,
    center_xy: bool = True,
    **fit_kwargs,
) -> tuple[GaussianScene, GroundPlane]:
    """Set the scene's world transform so the ground plane becomes z=0.

    Applies ``metric_scale`` first, then the rigid alignment, and returns a
    new scene whose ``positions_world()`` are z-up metric coordinates with
    the ground at z = 0 (XY centered on the scene footprint by default).
    """
    if plane is None:
        plane = fit_ground_plane(scene.positions, scene.opacities, **fit_kwargs)

    rot = _rotation_between(plane.normal, np.array([0.0, 0.0, 1.0]))
    aligned = scene.copy()
    aligned.metric_scale = float(metric_scale)

    # p_world = R * (s * p) + t ; choose t so the plane maps to z=0
    # plane point in scaled coords: s * (offset * normal)
    plane_point = metric_scale * plane.offset * plane.normal
    translation = -rot @ plane_point

    if center_xy:
        scaled = metric_scale * scene.positions
        world = scaled @ rot.T + translation
        translation = translation - np.array(
            [world[:, 0].mean(), world[:, 1].mean(), 0.0]
        )

    aligned.transform_to_world = Pose(
        position=translation, quaternion=matrix_to_quat(rot)
    )
    aligned.metadata = {
        **scene.metadata,
        "alignment": {
            "ground_normal_gaussian": plane.normal.tolist(),
            "ground_offset": plane.offset,
            "inlier_ratio": plane.inlier_ratio,
            "metric_scale": metric_scale,
        },
    }
    return aligned, plane


def estimate_metric_scale(known_distance_m: float, measured_distance_units: float) -> float:
    """Scale factor from a known physical distance.

    Measure the same distance in the reconstruction (e.g. between two
    fiducials, or a table edge of known length): ``scale = real / measured``.
    """
    if measured_distance_units <= 0:
        raise ValueError("measured distance must be positive")
    return float(known_distance_m / measured_distance_units)
