#!/usr/bin/env python3
"""Export viewer-ready .splat files with the target instance highlighted.

    python scripts/export_instance_splat.py \
        --scene data/gaussian/classroom_scene.npz \
        --bbox 0.55 1.80 -0.05 1.95 2.75 1.40 \
        --out-prefix data/gaussian/classroom_desk

Writes:
  <prefix>_highlight.splat   aligned scene, instance recolored bright red
  <prefix>_only.splat        just the instance gaussians
  <prefix>_preview.png       static 3-view preview with the highlight

Drag the .splat files into https://superspl.at/editor to inspect in 3D.
The export bakes pgtwin's world alignment and converts to the viewer's
y-up convention, so the room appears upright.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pgtwin.common.se3 import Pose, quat_from_axis_angle  # noqa: E402
from pgtwin.gaussian.instance_manager import InstanceManager  # noqa: E402
from pgtwin.gaussian.scene_loader import GaussianScene  # noqa: E402
from pgtwin.gaussian.splat_io import save_splat  # noqa: E402

HIGHLIGHT = np.array([1.0, 0.12, 0.08])  # bright red


def to_viewer_frame(scene: GaussianScene) -> GaussianScene:
    """Bake world alignment, then rotate z-up world -> y-up viewer frame
    (SuperSplat / PlayCanvas convention): (x, y, z) -> (x, z, -y)."""
    baked = GaussianScene(
        positions=scene.positions_world(),
        scales=scene.scales_world(),
        rotations=scene.rotations,
        opacities=scene.opacities,
        colors=scene.colors,
    )
    # rotations must be baked with the world alignment too
    from pgtwin.common.se3 import quat_multiply

    q_align = scene.transform_to_world.quaternion
    if baked.rotations is not None:
        baked.rotations = np.stack(
            [quat_multiply(q_align, q) for q in baked.rotations]
        )
    viewer = Pose(quaternion=quat_from_axis_angle([1.0, 0.0, 0.0], -np.pi / 2))
    out = baked.transform(viewer)
    return out


def preview_png(scene: GaussianScene, mask: np.ndarray, path: Path) -> None:
    pts = scene.positions_world()
    colors = np.clip(scene.colors, 0, 1) if scene.colors is not None else None
    rng = np.random.default_rng(0)
    bg = np.nonzero(~mask)[0]
    if len(bg) > 120_000:
        bg = rng.choice(bg, 120_000, replace=False)
    hi = np.nonzero(mask)[0]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    views = [("top-down XY", 0, 1), ("side XZ", 0, 2), ("side YZ", 1, 2)]
    for ax, (title, i, j) in zip(axes, views):
        ax.scatter(pts[bg, i], pts[bg, j], s=0.3, lw=0,
                   c=colors[bg] if colors is not None else "0.6")
        ax.scatter(pts[hi, i], pts[hi, j], s=1.2, lw=0, c=[HIGHLIGHT])
        ax.set_title(f"{title} — target instance in RED")
        ax.axis("equal")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="data/gaussian/classroom_scene.npz")
    parser.add_argument("--bbox", type=float, nargs=6, required=True,
                        metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                        help="instance AABB in world frame (m): min xyz, max xyz")
    parser.add_argument("--out-prefix", default="data/gaussian/instance_export")
    parser.add_argument("--highlight-min-opacity", type=float, default=0.25,
                        help="only recolor in-bbox gaussians at least this "
                             "opaque (avoids a red fog of floaters)")
    parser.add_argument("--no-viewer-up", action="store_true",
                        help="keep z-up instead of converting to viewer y-up")
    args = parser.parse_args()

    print(f"[1] loading {args.scene}")
    scene = GaussianScene.load(args.scene)
    x0, y0, z0, x1, y1, z1 = args.bbox
    instance = InstanceManager(scene).segment(
        "target", bbox_min=[x0, y0, z0], bbox_max=[x1, y1, z1]
    )
    in_bbox = np.zeros(len(scene), dtype=bool)
    in_bbox[instance.gaussian_indices] = True
    # recolor only the solid part of the instance: low-opacity gaussians
    # painted red would look like a red fog around the object
    mask = in_bbox.copy()
    if scene.opacities is not None:
        mask &= scene.opacities >= args.highlight_min_opacity
    print(f"    instance: {mask.sum():,} highlighted "
          f"(of {in_bbox.sum():,} in bbox, {len(scene):,} total)")

    print("[2] converting to viewer frame (y-up)" if not args.no_viewer_up
          else "[2] keeping z-up world frame")
    viewer_scene = scene if args.no_viewer_up else to_viewer_frame(scene)

    # highlight: bright red + opacity floor so it pops
    highlighted = viewer_scene.copy()
    colors = (highlighted.colors.copy() if highlighted.colors is not None
              else np.full((len(highlighted), 3), 0.7))
    colors[mask] = HIGHLIGHT
    highlighted.colors = colors
    if highlighted.opacities is not None:
        op = highlighted.opacities.copy()
        op[mask] = np.maximum(op[mask], 0.95)
        highlighted.opacities = op

    prefix = Path(args.out_prefix)
    p1 = save_splat(highlighted, prefix.parent / f"{prefix.name}_highlight.splat")
    # the standalone file keeps ALL in-bbox gaussians (incl. low-opacity
    # surface filler) so the object renders solid on its own
    p2 = save_splat(viewer_scene.select(np.nonzero(in_bbox)[0]),
                    prefix.parent / f"{prefix.name}_only.splat")
    print(f"[3] wrote {p1}  ({p1.stat().st_size/1e6:.1f} MB)")
    print(f"    wrote {p2}  ({p2.stat().st_size/1e6:.2f} MB)")

    preview = prefix.parent / f"{prefix.name}_preview.png"
    preview_png(scene, mask, preview)
    print(f"[4] preview -> {preview}")
    print("\nDrag the _highlight.splat file into https://superspl.at/editor —")
    print("the target instance is the RED object.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
