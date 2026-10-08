"""Physical Gaussian Object: visual representation + physics binding.

This is the "S7" output of the pipeline: an object-level binding between a
gaussian (visual) asset, a collision proxy (simulation geometry) and a set
of physics parameters — plus provenance of how those parameters were
calibrated.  Serialized as ``physical_object.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pgtwin.common.io import load_json, save_json
from pgtwin.common.se3 import Pose


@dataclass
class PhysicsParameters:
    """Rigid-body physics parameters (SI units).

    Phase 1 calibrates ``friction_dynamic`` only; the rest are carried with
    sensible defaults and calibrated in later phases (see
    ``ParameterCalibrationPolicy``).
    """

    mass: float = 0.5  # kg
    friction_static: float = 0.6  # unitless (Coulomb)
    friction_dynamic: float = 0.5  # unitless (Coulomb)
    restitution: float = 0.1  # unitless [0, 1]
    young_modulus: float | None = None  # Pa (deformables, Phase 8+)
    poisson_ratio: float | None = None  # unitless (deformables, Phase 8+)

    def to_dict(self) -> dict:
        d = {
            "mass": self.mass,
            "friction_static": self.friction_static,
            "friction_dynamic": self.friction_dynamic,
            "restitution": self.restitution,
        }
        if self.young_modulus is not None:
            d["young_modulus"] = self.young_modulus
        if self.poisson_ratio is not None:
            d["poisson_ratio"] = self.poisson_ratio
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "PhysicsParameters":
        def _opt_float(value):
            return None if value is None else float(value)

        return cls(
            mass=float(data.get("mass", 0.5)),
            friction_static=float(data.get("friction_static", 0.6)),
            friction_dynamic=float(
                data.get("friction_dynamic", data.get("friction", 0.5))
            ),
            restitution=float(data.get("restitution", 0.1)),
            young_modulus=_opt_float(data.get("young_modulus")),
            poisson_ratio=_opt_float(data.get("poisson_ratio")),
        )

    def update(self, params: dict[str, float]) -> "PhysicsParameters":
        """Return a copy with calibrated values applied.

        ``friction`` is an alias for ``friction_dynamic`` (and keeps
        ``friction_static`` >= ``friction_dynamic`` consistent).
        """
        out = PhysicsParameters(**{**self.__dict__})
        for key, value in params.items():
            if key == "friction":
                out.friction_dynamic = float(value)
                out.friction_static = max(out.friction_static, float(value))
            elif hasattr(out, key):
                setattr(out, key, float(value))
            else:
                raise KeyError(f"unknown physics parameter '{key}'")
        return out


@dataclass
class CollisionProxy:
    """Simulation collision geometry decoupled from the visual gaussians.

    ``kind`` is one of ``box | sphere | cylinder | convex_hull | mesh``.
    ``params`` holds kind-specific data:
      box: {"half_extents": [hx, hy, hz]}
      sphere: {"radius": r}
      cylinder: {"radius": r, "half_height": h}
      convex_hull / mesh: {"vertices": (N,3) list, "faces": optional}
    """

    kind: str = "box"
    params: dict[str, Any] = field(default_factory=lambda: {"half_extents": [0.03, 0.03, 0.03]})

    def footprint_radius(self) -> float:
        """Radius of the XY footprint circle (used by planar contact models)."""
        if self.kind == "sphere":
            return float(self.params["radius"])
        if self.kind == "cylinder":
            return float(self.params["radius"])
        if self.kind == "box":
            hx, hy, _ = self.params["half_extents"]
            return float(max(hx, hy))
        if self.kind in ("convex_hull", "mesh"):
            verts = np.asarray(self.params["vertices"], dtype=float)
            center = verts.mean(axis=0)
            return float(np.max(np.linalg.norm((verts - center)[:, :2], axis=1)))
        raise ValueError(f"unknown collision proxy kind '{self.kind}'")

    def half_height(self) -> float:
        if self.kind == "sphere":
            return float(self.params["radius"])
        if self.kind == "cylinder":
            return float(self.params["half_height"])
        if self.kind == "box":
            return float(self.params["half_extents"][2])
        if self.kind in ("convex_hull", "mesh"):
            verts = np.asarray(self.params["vertices"], dtype=float)
            return float((verts[:, 2].max() - verts[:, 2].min()) / 2.0)
        raise ValueError(f"unknown collision proxy kind '{self.kind}'")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": self.params}

    @classmethod
    def from_dict(cls, data: dict) -> "CollisionProxy":
        return cls(kind=data["kind"], params=data.get("params", {}))


@dataclass
class PhysicalObject:
    """Object-level binding of visual, collision and physical properties.

    ``gaussian_asset`` points at the visual representation (a PLY/NPZ file
    or an instance id inside a GaussianScene) — the physics side never
    depends on it, which keeps rendering and simulation decoupled.
    """

    object_id: str
    collision: CollisionProxy = field(default_factory=CollisionProxy)
    physics: PhysicsParameters = field(default_factory=PhysicsParameters)
    gaussian_asset: str | None = None
    initial_pose: Pose = field(default_factory=Pose.identity)
    calibration_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "gaussian_asset": self.gaussian_asset,
            "collision": self.collision.to_dict(),
            "physics": self.physics.to_dict(),
            "initial_pose": {
                "position": self.initial_pose.position.tolist(),
                "quaternion_xyzw": self.initial_pose.quaternion.tolist(),
            },
            "calibration": self.calibration_info,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PhysicalObject":
        pose_d = data.get("initial_pose", {})
        return cls(
            object_id=data["object_id"],
            collision=CollisionProxy.from_dict(data.get("collision", {"kind": "box"})),
            physics=PhysicsParameters.from_dict(data.get("physics", {})),
            gaussian_asset=data.get("gaussian_asset"),
            initial_pose=Pose(
                position=pose_d.get("position", [0, 0, 0]),
                quaternion=pose_d.get("quaternion_xyzw", [0, 0, 0, 1]),
            ),
            calibration_info=data.get("calibration", {}),
        )

    def save(self, path: str | Path) -> None:
        save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "PhysicalObject":
        return cls.from_dict(load_json(path))

    @classmethod
    def from_config(cls, cfg: dict) -> "PhysicalObject":
        """Build from an experiment-config ``object:`` section."""
        shape = cfg.get("shape", "box")
        if shape == "box":
            size = cfg.get("size", [0.06, 0.06, 0.06])
            proxy = CollisionProxy("box", {"half_extents": [s / 2 for s in size]})
        elif shape == "cylinder":
            proxy = CollisionProxy(
                "cylinder",
                {
                    "radius": float(cfg.get("radius", 0.03)),
                    "half_height": float(cfg.get("height", 0.06)) / 2,
                },
            )
        elif shape == "sphere":
            proxy = CollisionProxy("sphere", {"radius": float(cfg.get("radius", 0.03))})
        else:
            raise ValueError(f"unsupported object shape '{shape}'")
        # physics may be nested under 'physics:' or given as flat keys
        physics = PhysicsParameters.from_dict(cfg.get("physics", cfg))
        return cls(
            object_id=cfg.get("object_id", "object_0"),
            collision=proxy,
            physics=physics,
            gaussian_asset=cfg.get("gaussian_asset"),
        )
