#!/usr/bin/env python3
"""Prepare a raw 3DGS export for pgtwin: align to z-up, set metric scale.

    python scripts/prepare_gaussian_scene.py \
        --input data/gaussian/classroom_model.ply \
        --output data/gaussian/classroom_scene.npz \
        --scale 1.0 --max-points 400000 --preview

Outputs an NPZ scene (alignment + scale baked into metadata) and preview
scatter plots (top-down XY, side XZ) for choosing instance bounding boxes.
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

from pgtwin.common.logging import get_logger  # noqa: E402
from pgtwin.gaussian.alignment import align_scene_to_ground  # noqa: E402
from pgtwin.gaussian.scene_loader import GaussianScene  # noqa: E402

logger = get_logger("scripts.prepare_scene")


def preview_plots(scene: GaussianScene, out_prefix: Path, z_slice: float | None = None) -> None:
    pts = scene.positions_world()
    colors = scene.colors if scene.colors is not None else None
    # subsample for plotting
    if len(pts) > 150_000:
        idx = np.random.default_rng(0).choice(len(pts), 150_000, replace=False)
        pts = pts[idx]
        colors = colors[idx] if colors is not None else None
    kw = dict(s=0.3, lw=0, c=np.clip(colors, 0, 1) if colors is not None else "k")

    if z_slice is not None:
        mask = (pts[:, 2] > -0.05) & (pts[:, 2] < z_slice)
        pts_slice = pts[mask]
        kw_slice = dict(kw, c=kw["c"][mask] if colors is not None else "k")
    else:
        pts_slice, kw_slice = pts, kw

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(pts_slice[:, 0], pts_slice[:, 1], **kw_slice)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    title = "Top-down (XY)" + (f", z < {z_slice} m" if z_slice is not None else "")
    ax.set_title(title); ax.axis("equal"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out_prefix}_topdown.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(pts[:, 0], pts[:, 2], **kw)
    ax.axhline(0.0, color="r", lw=0.8, ls="--", label="ground z=0")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title("Side view (XZ)"); ax.axis("equal"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out_prefix}_side.png", dpi=130); plt.close(fig)
    logger.info("previews: %s_topdown.png, %s_side.png", out_prefix, out_prefix)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, help=".ply / .splat / .npz scene")
    parser.add_argument("--output", required=True, help="output .npz path")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="metric scale (world_m = scale * gaussian_units)")
    parser.add_argument("--max-points", type=int, default=None,
                        help="subsample to at most N gaussians")
    parser.add_argument("--min-opacity", type=float, default=0.5,
                        help="opacity threshold for plane fitting")
    parser.add_argument("--plane-threshold", type=float, default=0.02,
                        help="RANSAC inlier distance (gaussian units)")
    parser.add_argument("--plane-rank", type=int, default=0,
                        help="skip the N most dominant planes (indoor walls)")
    parser.add_argument("--no-align", action="store_true",
                        help="skip ground alignment (scene already z-up)")
    parser.add_argument("--crop", type=float, nargs=6, metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                        help="world-frame AABB crop applied after alignment (m)")
    parser.add_argument("--min-keep-opacity", type=float, default=0.0,
                        help="drop gaussians below this opacity (floater removal)")
    parser.add_argument("--preview", action="store_true", help="write scatter previews")
    parser.add_argument("--z-slice", type=float, default=None,
                        help="top-down preview only shows z < this (m)")
    args = parser.parse_args()

    print(f"[1] loading {args.input}")
    scene = GaussianScene.load(args.input, max_points=args.max_points)
    lo, hi = scene.positions.min(axis=0), scene.positions.max(axis=0)
    print(f"    {len(scene)} gaussians; raw bounds {np.round(lo, 2)} .. {np.round(hi, 2)}")

    if args.no_align:
        scene = scene.with_metric_scale(args.scale)
        print("[2] alignment skipped (--no-align)")
    else:
        print("[2] fitting ground plane + aligning to z-up world frame")
        scene, plane = align_scene_to_ground(
            scene,
            metric_scale=args.scale,
            distance_threshold=args.plane_threshold,
            min_opacity=args.min_opacity,
            rank=args.plane_rank,
        )
        print(f"    inliers {100 * plane.inlier_ratio:.1f}%  "
              f"normal(gaussian frame) {np.round(plane.normal, 3)}")

    pts = scene.positions_world()
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    print(f"[3] world bounds (m): {np.round(lo, 2)} .. {np.round(hi, 2)}")

    keep = np.ones(len(scene), dtype=bool)
    if args.crop is not None:
        x0, x1, y0, y1, z0, z1 = args.crop
        keep &= (
            (pts[:, 0] >= x0) & (pts[:, 0] <= x1)
            & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
            & (pts[:, 2] >= z0) & (pts[:, 2] <= z1)
        )
    if args.min_keep_opacity > 0 and scene.opacities is not None:
        keep &= scene.opacities >= args.min_keep_opacity
    if not keep.all():
        scene = scene.select(np.nonzero(keep)[0])
        print(f"    crop/opacity filter kept {len(scene):,} gaussians")

    out = scene.save(args.output)
    print(f"[4] saved -> {out}")
    if args.preview:
        preview_plots(scene, Path(args.output).with_suffix(""), args.z_slice)
        print("[5] previews written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
