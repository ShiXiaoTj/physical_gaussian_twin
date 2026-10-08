#!/usr/bin/env python3
"""Generate a synthetic push episode with known ground-truth physics.

    python scripts/generate_synthetic_episode.py \
        --out data/generated/push_box_002 --friction 0.6 --noise 0.001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pgtwin.real_data.synthetic import (  # noqa: E402
    SyntheticEpisodeGenerator,
    SyntheticPushConfig,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="episode output directory")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--friction", type=float, default=0.8, help="ground truth")
    parser.add_argument("--mass", type=float, default=0.42)
    parser.add_argument("--noise", type=float, default=0.0, help="pos noise std (m)")
    parser.add_argument("--push-speed", type=float, default=0.5, help="m/s")
    parser.add_argument("--lateral-offset", type=float, default=0.0,
                        help="push-line offset from object center (m)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    config = SyntheticPushConfig(
        episode_id=args.episode_id or out.name,
        true_friction=args.friction,
        true_mass=args.mass,
        noise_std_pos=args.noise,
        push_speed=args.push_speed,
        lateral_offset=args.lateral_offset,
        seed=args.seed,
    )
    episode, _ = SyntheticEpisodeGenerator(config).generate(save_dir=out)
    moved = episode.object_trajectory().total_displacement() * 100
    print(f"episode '{config.episode_id}' -> {out}")
    print(f"  ground-truth friction: {args.friction}, object moved {moved:.1f} cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
