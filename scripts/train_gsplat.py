#!/usr/bin/env python3
"""Train a real 3DGS scene with gsplat from a COLMAP capture (S2, for real).

    conda run -n pgtwin-gs python scripts/train_gsplat.py \
        --data data/mipnerf360/garden --downscale 4 \
        --iters 15000 --out data/gaussian/garden_trained

Expects <data>/sparse/0/{cameras,images,points3D}.bin and an image dir
(<data>/images_<downscale> if present, else <data>/images).

Outputs under --out:
    scene.npz            pgtwin GaussianScene (feeds the whole pipeline)
    scene.splat          for drag-in viewing at superspl.at/editor
    eval_XXXX.png        held-out view: ground truth | render, with PSNR
    train_log.csv        loss / PSNR curve

Uses gsplat's rasterizer + DefaultStrategy (official densify/prune schedule);
the training loop itself is deliberately minimal (L1 + SSIM, per-group Adam).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from pgtwin.gaussian.colmap_io import load_colmap_scene  # noqa: E402


# ----------------------------------------------------------------------------
# SSIM (standard 11x11 gaussian window)
# ----------------------------------------------------------------------------

def _gaussian_window(size: int = 11, sigma: float = 1.5, device="cuda"):
    xs = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(xs**2) / (2 * sigma**2))
    g = (g / g.sum()).outer(g / g.sum())
    return g[None, None].expand(3, 1, size, size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """img*: (H,W,3) in [0,1]."""
    x = img1.permute(2, 0, 1)[None]
    y = img2.permute(2, 0, 1)[None]
    pad = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=3)
    mu_y = F.conv2d(y, window, padding=pad, groups=3)
    sig_x = F.conv2d(x * x, window, padding=pad, groups=3) - mu_x**2
    sig_y = F.conv2d(y * y, window, padding=pad, groups=3) - mu_y**2
    sig_xy = F.conv2d(x * y, window, padding=pad, groups=3) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2)
    )
    return s.mean()


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_dataset(data_dir: Path, downscale: int):
    cams, images, xyz, rgb = load_colmap_scene(data_dir / "sparse" / "0")
    img_dir = data_dir / (f"images_{downscale}" if downscale > 1 else "images")
    if not img_dir.exists():
        raise FileNotFoundError(f"{img_dir} not found")

    viewmats, Ks, paths = [], [], []
    for im in sorted(images.values(), key=lambda i: i.name):
        p = img_dir / im.name
        if not p.exists():
            continue
        cam = cams[im.camera_id]
        with Image.open(p) as pil:
            w, h = pil.size
        K = cam.K().copy()
        K[0] *= w / cam.width
        K[1] *= h / cam.height
        viewmats.append(im.world_to_cam())
        Ks.append(K)
        paths.append(p)
    print(f"[data] {len(paths)} images at {w}x{h}, "
          f"{len(xyz):,} sparse points")
    return (np.stack(viewmats), np.stack(Ks), paths, (w, h), xyz, rgb)


def load_image(path: Path, device) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device)


# ----------------------------------------------------------------------------
# Model init / export
# ----------------------------------------------------------------------------

def knn_mean_dist(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean distance to k nearest neighbours (chunked, CPU)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(x)
    d, _ = tree.query(x, k=k + 1)
    return d[:, 1:].mean(axis=1)


def init_params(xyz: np.ndarray, rgb: np.ndarray, sh_degree: int, device):
    n = len(xyz)
    dist = np.clip(knn_mean_dist(xyz), 1e-4, None)
    scales = np.log(np.tile(dist[:, None], (1, 3)))
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0  # wxyz identity
    # SH DC from rgb: c = (rgb - 0.5) / C0
    sh0 = (rgb - 0.5) / 0.28209479177387814
    k_rest = (sh_degree + 1) ** 2 - 1
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float32)),
        "scales": torch.nn.Parameter(torch.tensor(scales, dtype=torch.float32)),
        "quats": torch.nn.Parameter(torch.tensor(quats)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1))),
        "sh0": torch.nn.Parameter(torch.tensor(sh0[:, None, :], dtype=torch.float32)),
        "shN": torch.nn.Parameter(torch.zeros(n, k_rest, 3)),
    }).to(device)
    return params


def export_scene(params, out_dir: Path):
    """Write pgtwin scene.npz (+ .splat for viewing)."""
    means = params["means"].detach().cpu().numpy()
    scales = np.exp(params["scales"].detach().cpu().numpy())
    quats_wxyz = params["quats"].detach().cpu().numpy()
    quats_wxyz /= np.maximum(np.linalg.norm(quats_wxyz, axis=1, keepdims=True), 1e-9)
    rotations = quats_wxyz[:, [1, 2, 3, 0]]  # -> xyzw
    opacities = 1.0 / (1.0 + np.exp(-params["opacities"].detach().cpu().numpy()))
    colors = np.clip(
        0.5 + 0.28209479177387814 * params["sh0"].detach().cpu().numpy()[:, 0, :],
        0.0, 1.0,
    )
    from pgtwin.gaussian.scene_loader import GaussianScene
    from pgtwin.gaussian.splat_io import save_splat

    scene = GaussianScene(
        positions=means, scales=scales, rotations=rotations,
        opacities=opacities, colors=colors,
        metadata={"trained_by": "pgtwin train_gsplat", "n_gaussians": len(means)},
    )
    scene.save(out_dir / "scene.npz")
    save_splat(scene, out_dir / "scene.splat", bake_world=False)
    print(f"[export] {len(means):,} gaussians -> {out_dir}/scene.npz + scene.splat")


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="scene dir with sparse/0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--iters", type=int, default=15000)
    parser.add_argument("--sh-degree", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--n-eval", type=int, default=4, help="held-out views")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    viewmats_np, Ks_np, paths, (W, H), xyz, rgb = load_dataset(data_dir, args.downscale)

    # held-out eval views (every-Nth convention from mipnerf360: llffhold=8)
    eval_ids = list(range(0, len(paths), max(len(paths) // args.n_eval, 1)))[: args.n_eval]
    train_ids = [i for i in range(len(paths)) if i not in eval_ids]

    viewmats = torch.tensor(viewmats_np, dtype=torch.float32, device=device)
    Ks = torch.tensor(Ks_np, dtype=torch.float32, device=device)
    params = init_params(xyz, rgb, args.sh_degree, device)

    lrs = {"means": 1.6e-4 * 5.0, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {
        k: torch.optim.Adam([{"params": [params[k]], "lr": lr, "name": k}], eps=1e-15)
        for k, lr in lrs.items()
    }
    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(params, optimizers)
    # scene scale steers densification thresholds (as in gsplat's trainer)
    campos = np.linalg.inv(viewmats_np)[:, :3, 3]
    scene_scale = float(np.linalg.norm(campos - campos.mean(0), axis=1).max()) * 1.1
    state = strategy.initialize_state(scene_scale=scene_scale)
    print(f"[train] scene_scale = {scene_scale:.2f}")

    window = _gaussian_window(device=device)
    sh_deg_max = args.sh_degree
    log_rows = []
    t_start = time.time()

    print(f"[train] {args.iters} iters on {len(train_ids)} views (cuda: "
          f"{torch.cuda.get_device_name(0)})")
    for step in range(args.iters):
        i = train_ids[np.random.randint(len(train_ids))]
        gt = load_image(paths[i], device)
        sh_degree = min(step // 1000, sh_deg_max)

        renders, alphas, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=torch.cat([params["sh0"], params["shN"]], dim=1),
            viewmats=viewmats[i][None],
            Ks=Ks[i][None],
            width=W, height=H,
            sh_degree=sh_degree,
            packed=False,
        )
        render = renders[0].clamp(0, 1)

        strategy.step_pre_backward(params, optimizers, state, step, info)
        l1 = (render - gt).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1.0 - ssim(render, gt, window))
        loss.backward()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        if (step + 1) % 500 == 0:
            with torch.no_grad():
                psnr = -10 * math.log10(((render - gt) ** 2).mean().item())
            n_g = len(params["means"])
            print(f"  step {step+1:6d}  loss {loss.item():.4f}  "
                  f"train-PSNR {psnr:5.2f}  gaussians {n_g:,}  "
                  f"({time.time()-t_start:5.0f}s)")
            log_rows.append({"step": step + 1, "loss": loss.item(),
                             "psnr": psnr, "n_gaussians": n_g})

        if (step + 1) % args.eval_every == 0 or step + 1 == args.iters:
            with torch.no_grad():
                psnrs = []
                for j in eval_ids:
                    gt_e = load_image(paths[j], device)
                    r, _, _ = rasterization(
                        means=params["means"], quats=params["quats"],
                        scales=torch.exp(params["scales"]),
                        opacities=torch.sigmoid(params["opacities"]),
                        colors=torch.cat([params["sh0"], params["shN"]], dim=1),
                        viewmats=viewmats[j][None], Ks=Ks[j][None],
                        width=W, height=H, sh_degree=sh_deg_max, packed=False,
                    )
                    r = r[0].clamp(0, 1)
                    p = -10 * math.log10(((r - gt_e) ** 2).mean().item())
                    psnrs.append(p)
                    side = torch.cat([gt_e, r], dim=1).cpu().numpy()
                    Image.fromarray((side * 255).astype(np.uint8)).save(
                        out_dir / f"eval_{step+1:06d}_view{j:03d}_psnr{p:.1f}.png")
                print(f"  [eval @ {step+1}] held-out PSNR: "
                      f"{np.mean(psnrs):.2f} dB  {[f'{p:.1f}' for p in psnrs]}")

    with open(out_dir / "train_log.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "loss", "psnr", "n_gaussians"])
        writer.writeheader()
        writer.writerows(log_rows)
    export_scene(params, out_dir)
    print(f"[done] total {time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
