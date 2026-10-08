"""Loader for the compact ``.splat`` format (antimatter15 / SuperSplat).

Layout: 32 bytes per gaussian, no header —
  float32[3] position, float32[3] scale (linear, already exp'd),
  uint8[4] RGBA color (SH DC baked in, alpha = opacity),
  uint8[4] rotation quaternion, stored (w, x, y, z) as round(q*128+128).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_RECORD = np.dtype(
    [
        ("position", "<f4", 3),
        ("scale", "<f4", 3),
        ("rgba", "u1", 4),
        ("rot", "u1", 4),
    ]
)


def load_splat(path: str | Path, max_points: int | None = None):
    """Load an antimatter15-format ``.splat`` file into a GaussianScene."""
    from pgtwin.gaussian.scene_loader import GaussianScene

    raw = np.fromfile(str(path), dtype=_RECORD)
    if len(raw) == 0:
        raise ValueError(f"{path}: empty or not a .splat file")

    positions = raw["position"].astype(float)
    scales = raw["scale"].astype(float)
    colors = raw["rgba"][:, :3].astype(float) / 255.0
    opacities = raw["rgba"][:, 3].astype(float) / 255.0
    # stored (w, x, y, z) as round(q*128+128); per-component quantization
    # drifts the norm, so renormalize (as every splat viewer does)
    q_wxyz = (raw["rot"].astype(float) - 128.0) / 128.0
    norms = np.linalg.norm(q_wxyz, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    rotations = (q_wxyz / norms)[:, [1, 2, 3, 0]]

    if max_points is not None and len(positions) > max_points:
        idx = np.linspace(0, len(positions) - 1, max_points, dtype=int)
        positions, scales = positions[idx], scales[idx]
        colors, opacities, rotations = colors[idx], opacities[idx], rotations[idx]

    return GaussianScene(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
        source_path=str(path),
    )


def save_splat(scene, path: str | Path, bake_world: bool = True) -> Path:
    """Write a GaussianScene as an antimatter15 ``.splat`` file.

    ``bake_world=True`` applies the scene's alignment (``metric_scale`` +
    ``transform_to_world``) to positions, scales and rotations, so viewers
    like SuperSplat show the scene in pgtwin's metric, z-up world frame.
    """
    from pathlib import Path as _Path

    from pgtwin.common.se3 import quat_multiply

    n = len(scene)
    if bake_world:
        positions = scene.positions_world()
        scales = scene.scales_world() if scene.scales is not None else None
        q_align = scene.transform_to_world.quaternion
        if scene.rotations is not None:
            rotations = np.stack(
                [quat_multiply(q_align, q) for q in scene.rotations]
            )
        else:
            rotations = None
    else:
        positions = scene.positions
        scales = scene.scales
        rotations = scene.rotations

    if scales is None:
        scales = np.full((n, 3), 0.005)
    if rotations is None:
        rotations = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    colors = scene.colors if scene.colors is not None else np.full((n, 3), 0.7)
    opacities = scene.opacities if scene.opacities is not None else np.full(n, 0.9)

    raw = np.empty(n, dtype=_RECORD)
    raw["position"] = positions.astype("<f4")
    raw["scale"] = scales.astype("<f4")
    raw["rgba"][:, :3] = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    raw["rgba"][:, 3] = np.clip(opacities * 255.0, 0, 255).astype(np.uint8)
    q_wxyz = rotations[:, [3, 0, 1, 2]]
    norms = np.linalg.norm(q_wxyz, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    raw["rot"] = np.clip(
        np.round((q_wxyz / norms) * 128.0 + 128.0), 0, 255
    ).astype(np.uint8)

    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.tofile(str(path))
    return path
