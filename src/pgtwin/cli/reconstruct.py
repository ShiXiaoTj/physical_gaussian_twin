"""`pgtwin reconstruct`: import / inspect Gaussian scenes (Phase 5 seed)."""

from __future__ import annotations

import argparse
from pathlib import Path

from pgtwin.gaussian.reconstruction import create_backend
from pgtwin.gaussian.scene_loader import GaussianScene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pgtwin reconstruct",
        description="Import a Gaussian scene (PLY/NPZ) or run a reconstruction backend.",
    )
    parser.add_argument("--backend", default="ply_import",
                        help="ply_import | gsplat | splatfacto")
    parser.add_argument("--ply", help="existing PLY/NPZ export (ply_import backend)")
    parser.add_argument("--images", help="images directory (training backends)")
    parser.add_argument("--out", default="data/gaussian/scene.npz", help="output NPZ")
    args = parser.parse_args(argv)

    backend = create_backend(args.backend)
    if not backend.is_available():
        print(f"backend '{args.backend}' dependencies not installed")
        return 1
    scene = backend.reconstruct(args.images or ".", ply_path=args.ply)
    lo, hi = scene.bounds_world()
    print(f"loaded {len(scene)} gaussians; world bounds {lo.round(3)} .. {hi.round(3)}")
    out = scene.save(Path(args.out))
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
