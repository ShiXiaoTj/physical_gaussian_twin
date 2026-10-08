"""GaussianScene: backend-agnostic Gaussian scene container.

pgtwin does **not** reimplement 3D Gaussian Splatting.  A scene is a plain
array-of-primitives container (positions, scales, rotations, opacity,
colors) plus a frame transform ``T_gaussian_world``.  Reconstruction
backends (original 3DGS, gsplat, Splatfacto, PhysGaussian, ...) plug in via
:mod:`pgtwin.gaussian.reconstruction`; rendering stays entirely on the
backend side so physics never couples to a renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pgtwin.common.logging import get_logger
from pgtwin.common.se3 import Pose

logger = get_logger("gaussian.scene")


@dataclass
class GaussianScene:
    """A set of 3D Gaussians in a common frame.

    Arrays are parallel with ``len(positions) == n``:
    ``positions`` (n,3) m, ``scales`` (n,3) m, ``rotations`` (n,4) xyzw,
    ``opacities`` (n,), ``colors`` (n,3) in [0,1].
    """

    positions: np.ndarray
    scales: np.ndarray | None = None
    rotations: np.ndarray | None = None
    opacities: np.ndarray | None = None
    colors: np.ndarray | None = None
    #: Transform from Gaussian/reconstruction frame to pgtwin world frame.
    transform_to_world: Pose = field(default_factory=Pose)
    #: Metric scale: world_position = T * (metric_scale * gaussian_position).
    #: 3DGS reconstructions have arbitrary scale; set this from a known
    #: physical distance (see gaussian.alignment.estimate_metric_scale).
    metric_scale: float = 1.0
    source_path: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = np.atleast_2d(np.asarray(self.positions, dtype=float))
        n = len(self.positions)
        for name in ("scales", "rotations", "opacities", "colors"):
            arr = getattr(self, name)
            if arr is not None:
                arr = np.asarray(arr, dtype=float)
                if len(arr) != n:
                    raise ValueError(f"{name} has {len(arr)} entries, expected {n}")
                setattr(self, name, arr)

    def __len__(self) -> int:
        return len(self.positions)

    # -- frames ---------------------------------------------------------------
    def positions_world(self) -> np.ndarray:
        """Gaussian centers in the pgtwin world frame (metric scale applied)."""
        return self.transform_to_world.transform_points(
            self.metric_scale * self.positions
        )

    def scales_world(self) -> np.ndarray | None:
        """Gaussian extents in metric units (rotation does not change them)."""
        if self.scales is None:
            return None
        return self.metric_scale * self.scales

    def transform(self, pose: Pose) -> "GaussianScene":
        """Return a copy with ``pose`` pre-applied to ``T_gaussian_world``."""
        new = self.copy()
        new.transform_to_world = pose.compose(self.transform_to_world)
        return new

    def with_metric_scale(self, scale: float) -> "GaussianScene":
        new = self.copy()
        new.metric_scale = float(scale)
        return new

    # -- selection ------------------------------------------------------------
    def select(self, indices: np.ndarray) -> "GaussianScene":
        """Sub-scene from Gaussian indices (view semantics: copies arrays)."""
        idx = np.asarray(indices, dtype=int)
        return GaussianScene(
            positions=self.positions[idx],
            scales=None if self.scales is None else self.scales[idx],
            rotations=None if self.rotations is None else self.rotations[idx],
            opacities=None if self.opacities is None else self.opacities[idx],
            colors=None if self.colors is None else self.colors[idx],
            transform_to_world=self.transform_to_world.copy(),
            metric_scale=self.metric_scale,
            source_path=self.source_path,
            metadata=dict(self.metadata),
        )

    def bounds_world(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.positions_world()
        return p.min(axis=0), p.max(axis=0)

    def copy(self) -> "GaussianScene":
        return self.select(np.arange(len(self)))

    # -- IO ---------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "GaussianScene":
        """Load from ``.ply`` (3DGS convention), ``.splat`` or ``.npz``."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".ply":
            from pgtwin.gaussian.ply_io import load_gaussian_ply

            return load_gaussian_ply(path, **kwargs)
        if suffix == ".splat":
            from pgtwin.gaussian.splat_io import load_splat

            return load_splat(path, **kwargs)
        if suffix == ".npz":
            data = np.load(path)
            return cls(
                positions=data["positions"],
                scales=data.get("scales"),
                rotations=data.get("rotations"),
                opacities=data.get("opacities"),
                colors=data.get("colors"),
                transform_to_world=Pose(
                    position=data["transform_position"],
                    quaternion=data["transform_quaternion"],
                )
                if "transform_position" in data
                else Pose(),
                metric_scale=float(data["metric_scale"])
                if "metric_scale" in data
                else 1.0,
                source_path=str(path),
            )
        raise ValueError(f"unsupported Gaussian scene format: {path.suffix}")

    def save(self, path: str | Path) -> Path:
        """Save as ``.npz`` (lossless for this container, incl. alignment)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "positions": self.positions,
            "transform_position": self.transform_to_world.position,
            "transform_quaternion": self.transform_to_world.quaternion,
            "metric_scale": np.float64(self.metric_scale),
        }
        for name in ("scales", "rotations", "opacities", "colors"):
            arr = getattr(self, name)
            if arr is not None:
                arrays[name] = arr
        np.savez_compressed(path, **arrays)
        logger.info("saved GaussianScene (%d gaussians) to %s", len(self), path)
        return path

    # -- synthetic fallback ------------------------------------------------------
    @classmethod
    def synthetic_box(
        cls,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        n_points: int = 2000,
        seed: int = 0,
        color: tuple[float, float, float] = (0.8, 0.5, 0.2),
    ) -> "GaussianScene":
        """Synthetic box-shaped Gaussian cloud.

        Used when no real 3DGS reconstruction is available so the full
        pipeline (including Gaussian-physics binding) still runs.
        """
        rng = np.random.default_rng(seed)
        half = np.asarray(size, dtype=float) / 2.0
        pts = rng.uniform(-half, half, size=(n_points, 3)) + np.asarray(center)
        n = len(pts)
        return cls(
            positions=pts,
            scales=np.full((n, 3), 0.004),
            rotations=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
            opacities=np.full(n, 0.9),
            colors=np.tile(color, (n, 1)),
            metadata={"synthetic": True, "shape": "box", "size": list(size)},
        )

    @classmethod
    def synthetic_tabletop(
        cls,
        table_size: tuple[float, float] = (1.2, 0.8),
        table_height: float = 0.0,
        n_points: int = 4000,
        seed: int = 0,
    ) -> "GaussianScene":
        """Synthetic table plane as scene background."""
        rng = np.random.default_rng(seed)
        pts = np.column_stack(
            [
                rng.uniform(0.1, 0.1 + table_size[0], n_points),
                rng.uniform(-table_size[1] / 2, table_size[1] / 2, n_points),
                np.full(n_points, table_height) + rng.normal(0, 0.001, n_points),
            ]
        )
        n = len(pts)
        return cls(
            positions=pts,
            scales=np.full((n, 3), 0.006),
            rotations=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
            opacities=np.full(n, 0.85),
            colors=np.tile((0.85, 0.85, 0.87), (n, 1)),
            metadata={"synthetic": True, "shape": "table"},
        )

    def merge(self, other: "GaussianScene") -> "GaussianScene":
        """Concatenate two scenes (both converted to world frame)."""

        def stack(a, b, default_fn):
            if a is None and b is None:
                return None
            a = default_fn(len(self)) if a is None else a
            b = default_fn(len(other)) if b is None else b
            return np.concatenate([a, b], axis=0)

        return GaussianScene(
            positions=np.concatenate([self.positions_world(), other.positions_world()]),
            scales=stack(self.scales, other.scales, lambda n: np.full((n, 3), 0.005)),
            rotations=stack(
                self.rotations, other.rotations,
                lambda n: np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
            ),
            opacities=stack(self.opacities, other.opacities, lambda n: np.full(n, 0.9)),
            colors=stack(self.colors, other.colors, lambda n: np.full((n, 3), 0.7)),
            metadata={"merged": True},
        )
