#!/usr/bin/env python3
"""Render demo videos: make the calibration story visible.

Two modes:

  # 2D before/after animation from a saved experiment (optionally over the
  # real gaussian scene as background):
  python scripts/render_video.py anim --experiment outputs/push_chair_playroom_001 \
      --scene data/gaussian/playroom_scene.npz

  # 3D MuJoCo render of the push demo, blue ghost = where the real object went:
  python scripts/render_video.py mujoco3d --experiment outputs/push_box_mujoco_001

Videos land in outputs/videos/*.mp4 (H.264, plays anywhere).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pgtwin.common.se3 import quat_to_matrix, quat_to_yaw  # noqa: E402
from pgtwin.real_data.pose_utils import PoseTrajectory  # noqa: E402

VIDEOS_DIR = PROJECT_ROOT / "outputs" / "videos"

REAL_COLOR = "#1f77b4"
BEFORE_COLOR = "#d62728"
AFTER_COLOR = "#2ca02c"


def load_traj(path: Path) -> PoseTrajectory:
    df = pd.read_csv(path)
    return PoseTrajectory(
        df["timestamp"].to_numpy(),
        df[["x", "y", "z"]].to_numpy(),
        df[["qx", "qy", "qz", "qw"]].to_numpy(),
    )


def footprint_halfextents(physical_object: dict) -> tuple[float, float]:
    col = physical_object.get("collision", {})
    params = col.get("params", {})
    if col.get("kind") == "box":
        hx, hy, _ = params["half_extents"]
        return float(hx), float(hy)
    if col.get("kind") in ("sphere", "cylinder"):
        r = float(params["radius"])
        return r, r
    verts = np.asarray(params.get("vertices", [[0.03, 0.03, 0]]))
    return float(np.max(np.abs(verts[:, 0]))), float(np.max(np.abs(verts[:, 1])))


def rect_corners(cx: float, cy: float, hx: float, hy: float, yaw: float) -> np.ndarray:
    base = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return base @ rot.T + [cx, cy]


def render_scene_background(scene_npz: Path, xlim, ylim, px: int = 900):
    """Rasterize a top-down scatter of the scene once, reused every frame."""
    from pgtwin.gaussian.scene_loader import GaussianScene

    scene = GaussianScene.load(scene_npz)
    pts = scene.positions_world()
    colors = np.clip(scene.colors, 0, 1) if scene.colors is not None else None
    m = ((pts[:, 0] >= xlim[0]) & (pts[:, 0] <= xlim[1])
         & (pts[:, 1] >= ylim[0]) & (pts[:, 1] <= ylim[1]))
    idx = np.nonzero(m)[0]
    if len(idx) > 250_000:
        idx = np.random.default_rng(0).choice(idx, 250_000, replace=False)
    aspect = (ylim[1] - ylim[0]) / max(xlim[1] - xlim[0], 1e-9)
    fig = plt.figure(figsize=(px / 100, px * aspect / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(pts[idx, 0], pts[idx, 1], s=0.4, lw=0,
               c=colors[idx] if colors is not None else "0.5", alpha=0.8)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.axis("off")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img


def mode_anim(args) -> Path:
    exp = Path(args.experiment)
    result = json.loads((exp / "result.json").read_text())
    pobj = json.loads((exp / "physical_object.json").read_text())
    real = load_traj(exp / "real_trajectory.csv")
    before = load_traj(exp / "sim_before.csv")
    after = load_traj(exp / "sim_after.csv")
    hx, hy = footprint_halfextents(pobj)

    param = list(result["optimized_params"])[0]
    v0 = result["initial_params"][param]
    v1 = result["optimized_params"][param]
    truth = (result.get("ground_truth_params") or {}).get(param)

    # view: union of all object corners over time + margin
    all_xy = []
    for traj in (real, before, after):
        for i in range(len(traj)):
            all_xy.append(rect_corners(traj.positions[i, 0], traj.positions[i, 1],
                                       hx, hy, quat_to_yaw(traj.quaternions[i])))
    all_xy = np.concatenate(all_xy)
    lo, hi = all_xy.min(0), all_xy.max(0)
    margin = 0.35 * max(hi - lo) + 0.05
    xlim = (lo[0] - margin, hi[0] + margin)
    ylim = (lo[1] - margin, hi[1] + margin)

    background = None
    if args.scene:
        background = render_scene_background(Path(args.scene), xlim, ylim)

    # timeline (with an end-hold so the final state sinks in)
    t0, t1 = real.start_time, real.end_time
    n_frames = int((t1 - t0) * args.fps / args.speed)
    ts = t0 + np.arange(n_frames) * (t1 - t0) / max(n_frames - 1, 1)
    ts = np.concatenate([ts, np.full(int(1.5 * args.fps), t1)])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), dpi=100)
    fig.suptitle(f"{result['episode_id']} — real-vs-sim replay "
                 f"({param} calibration)", fontsize=13)
    artists = {}
    for ax, tag, sim_traj, v, color in (
        (axes[0], "BEFORE", before, v0, BEFORE_COLOR),
        (axes[1], "AFTER", after, v1, AFTER_COLOR),
    ):
        if background is not None:
            ax.imshow(background, extent=[*xlim, *ylim], origin="upper",
                      zorder=0, interpolation="bilinear")
        title = f"{tag} calibration:  {param} = {v:.2f}"
        if tag == "AFTER" and truth is not None:
            title += f"   (ground truth {truth:.2f})"
        ax.set_title(title, color=color, fontsize=11)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        real_poly = plt.Polygon(np.zeros((4, 2)), closed=True, fill=False,
                                ec=REAL_COLOR, lw=2.5, zorder=5, label="real object")
        sim_poly = plt.Polygon(np.zeros((4, 2)), closed=True, fill=False,
                               ec=color, lw=2.2, ls="--", zorder=6,
                               label=f"sim object")
        ax.add_patch(real_poly); ax.add_patch(sim_poly)
        real_trail, = ax.plot([], [], color=REAL_COLOR, lw=1.2, alpha=0.7, zorder=4)
        sim_trail, = ax.plot([], [], color=color, lw=1.2, ls="--", alpha=0.7, zorder=4)
        err_txt = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=11,
                          va="bottom", bbox=dict(fc="white", alpha=0.75, ec="none"))
        ax.legend(loc="upper right", fontsize=9)
        artists[tag] = (sim_traj, real_poly, sim_poly, real_trail, sim_trail, err_txt)
    time_txt = fig.text(0.5, 0.015, "", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    def draw(frame_i):
        t = ts[frame_i]
        rp = real.pose_at(t)
        for tag, (sim_traj, real_poly, sim_poly, r_tr, s_tr, err_txt) in artists.items():
            sp = sim_traj.pose_at(t)
            real_poly.set_xy(rect_corners(rp.position[0], rp.position[1], hx, hy, rp.yaw))
            sim_poly.set_xy(rect_corners(sp.position[0], sp.position[1], hx, hy, sp.yaw))
            past = real.timestamps <= t
            r_tr.set_data(real.positions[past, 0], real.positions[past, 1])
            past_s = sim_traj.timestamps <= t
            s_tr.set_data(sim_traj.positions[past_s, 0], sim_traj.positions[past_s, 1])
            err = np.linalg.norm(rp.position[:2] - sp.position[:2]) * 100
            err_txt.set_text(f"position error: {err:5.2f} cm")
        time_txt.set_text(f"t = {t - t0:.2f} s   (blue solid = real, dashed = simulated)")
        return []

    from matplotlib.animation import FFMpegWriter, FuncAnimation

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    out = VIDEOS_DIR / f"{exp.name}_anim.mp4"
    anim = FuncAnimation(fig, draw, frames=len(ts), blit=False)
    anim.save(out, writer=FFMpegWriter(fps=args.fps, bitrate=2200))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3D MuJoCo mode
# ---------------------------------------------------------------------------

def _caption(width: int, text: str, height: int = 52) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), (18, 22, 28))
    draw = ImageDraw.Draw(img)
    font = None
    for path, idx in (("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc", 0),
                      ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", None)):
        try:
            font = (ImageFont.truetype(path, 22, index=idx) if idx is not None
                    else ImageFont.truetype(path, 22))
            break
        except Exception:
            continue
    draw.text((width // 2, height // 2), text, fill=(235, 238, 242),
              font=font, anchor="mm")
    return np.asarray(img)


def mode_mujoco3d(args) -> Path:
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from pgtwin.common.io import load_yaml
    from pgtwin.physics.physical_object import PhysicalObject
    from pgtwin.real_data.trajectory_loader import EpisodeLoader
    from pgtwin.simulator import create_simulator

    exp = Path(args.experiment)
    result = json.loads((exp / "result.json").read_text())
    cfg = load_yaml(exp / "config.yaml")
    episode_dir = PROJECT_ROOT / "data" / "generated" / exp.name
    episode = EpisodeLoader.load(episode_dir)
    real = episode.object_trajectory()
    ee_traj = episode.ee_trajectory()

    obj_cfg = dict(cfg.get("object", {}))
    obj_cfg.setdefault("object_id", episode.metadata.object_id)
    obj = PhysicalObject.from_config(obj_cfg)
    hx, hy, hz = obj.collision.params.get("half_extents", [0.04, 0.04, 0.02])

    param = list(result["optimized_params"])[0]
    v0 = result["initial_params"][param]
    v1 = result["optimized_params"][param]
    truth = (result.get("ground_truth_params") or {}).get(param)

    W, H = 640, 480
    fps = args.fps
    dt_frame = args.speed / fps  # sim-seconds per video frame

    def rollout_frames(friction: float) -> list[np.ndarray]:
        sim = create_simulator(cfg.get("simulation", {}))
        sim.load_scene(cfg.get("scene", {}))
        sim.load_robot({"ee_radius": 0.012})
        sim.load_object(obj, episode.initial_object_pose())
        sim.reset()
        sim.set_object_pose(obj.object_id, episode.initial_object_pose())
        sim.set_physics_parameters(obj.object_id, {param: friction})
        # the generated MJCF defines no lights; brighten the headlight
        sim.model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
        sim.model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
        sim.model.vis.headlight.specular[:] = [0.3, 0.3, 0.3]

        renderer = mujoco.Renderer(sim.model, H, W)
        cam = mujoco.MjvCamera()
        start = episode.initial_object_pose().position
        cam.lookat[:] = [start[0] + 0.06, start[1], start[2] + 0.02]
        cam.distance, cam.azimuth, cam.elevation = 0.62, 125, -22

        dt = sim.dt
        t0, t1 = ee_traj.start_time, ee_traj.end_time
        n_steps = int(np.ceil((t1 - t0 + 0.6) / dt))
        every = max(1, int(round(dt_frame / dt)))
        frames = []

        def snap():
            renderer.update_scene(sim.data, cam)
            scn = renderer.scene
            # ghost: where the REAL object was at this sim time
            gp = real.pose_at(t0 + sim.time)
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_BOX,
                np.array([hx, hy, hz]),
                gp.position.astype(np.float64),
                quat_to_matrix(gp.quaternion).flatten().astype(np.float64),
                np.array([0.15, 0.35, 1.0, 0.35], dtype=np.float32),
            )
            scn.ngeom += 1
            frames.append(renderer.render().copy())

        sim.set_ee_pose(ee_traj.pose_at(t0))
        snap()
        for i in range(n_steps):
            t_next = t0 + (i + 1) * dt
            sim.set_ee_pose(ee_traj.pose_at(min(t_next, t1)))
            sim.step()
            if (i + 1) % every == 0:
                snap()
        renderer.close()
        sim.close()
        return frames

    print(f"[1] rendering BEFORE rollout ({param}={v0}) ...")
    frames_b = rollout_frames(v0)
    print(f"[2] rendering AFTER rollout ({param}={v1}) ...")
    frames_a = rollout_frames(v1)
    n = min(len(frames_b), len(frames_a))

    cap_b = _caption(W, f"标定前 BEFORE   {param} = {v0:.2f}")
    truth_s = f"  (真值 truth {truth:.2f})" if truth is not None else ""
    cap_a = _caption(W, f"标定后 AFTER   {param} = {v1:.2f}{truth_s}")
    footer = _caption(2 * W, "蓝色半透明 = 真实物体实际轨迹 (blue ghost = real object)   ·   "
                             "Physical Gaussian Digital Twin", height=40)

    print("[3] encoding ...")
    tmp = Path(tempfile.mkdtemp(prefix="pgtwin_vid_"))
    hold = int(1.5 * fps)
    order = list(range(n)) + [n - 1] * hold
    for k, i in enumerate(order):
        row = np.hstack([np.vstack([cap_b, frames_b[i]]),
                         np.vstack([cap_a, frames_a[i]])])
        frame = np.vstack([row, footer])
        plt.imsave(tmp / f"f_{k:05d}.png", frame)

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    out = VIDEOS_DIR / f"{exp.name}_3d.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", str(tmp / "f_%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)],
        check=True,
    )
    shutil.rmtree(tmp)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    p_anim = sub.add_parser("anim", help="2D before/after animation")
    p_anim.add_argument("--experiment", required=True, help="outputs/<experiment_id>")
    p_anim.add_argument("--scene", help="gaussian scene .npz for the background")
    p_anim.add_argument("--fps", type=int, default=30)
    p_anim.add_argument("--speed", type=float, default=1.0,
                        help="playback speed (0.5 = slow motion)")

    p_sp = sub.add_parser("splat", help="gaussian point-splat video: the real "
                          "object's own gaussians moving in the real scene")
    p_sp.add_argument("--experiment", required=True)
    p_sp.add_argument("--scene", required=True, help="aligned scene .npz")
    p_sp.add_argument("--bbox", type=float, nargs=6, required=True,
                      metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                      help="instance bbox (world m): min xyz, max xyz")
    p_sp.add_argument("--eye", type=float, nargs=3, required=True)
    p_sp.add_argument("--lookat", type=float, nargs=3, required=True)
    p_sp.add_argument("--fov", type=float, default=55.0)
    p_sp.add_argument("--fps", type=int, default=20)
    p_sp.add_argument("--speed", type=float, default=0.8)
    p_sp.add_argument("--max-bg", type=int, default=450_000)
    p_sp.add_argument("--max-instance", type=int, default=70_000)

    p_3d = sub.add_parser("mujoco3d", help="3D MuJoCo before/after render")
    p_3d.add_argument("--experiment", required=True,
                      help="outputs/<experiment_id> (a mujoco-backend run)")
    p_3d.add_argument("--fps", type=int, default=30)
    p_3d.add_argument("--speed", type=float, default=0.6,
                      help="sim-seconds per wall-second (0.6 = slight slow-mo)")

    args = parser.parse_args()
    out = {"anim": mode_anim, "mujoco3d": mode_mujoco3d,
           "splat": mode_splat}[args.mode](args)
    size = out.stat().st_size / 1e6
    print(f"video -> {out}  ({size:.1f} MB)")
    return 0


# ---------------------------------------------------------------------------
# Splat mode: true EWA gaussian-splatting render (CPU), the object itself
# ---------------------------------------------------------------------------

def _batch_quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of one quaternion q1 (xyzw) with a batch q2 (N,4)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.column_stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def mode_splat(args) -> Path:
    """EWA gaussian-splatting video (CPU): background rasterized once, the
    segmented instance re-splatted per frame along the sim trajectory
    (before | after), with a blue ghost copy at the real pose."""
    from pgtwin.gaussian.cpu_splatter import render_gaussians
    from pgtwin.gaussian.scene_loader import GaussianScene

    exp = Path(args.experiment)
    result = json.loads((exp / "result.json").read_text())
    real = load_traj(exp / "real_trajectory.csv")
    before = load_traj(exp / "sim_before.csv")
    after = load_traj(exp / "sim_after.csv")

    param = list(result["optimized_params"])[0]
    v0 = result["initial_params"][param]
    v1 = result["optimized_params"][param]
    truth = (result.get("ground_truth_params") or {}).get(param)

    print("[1] loading scene + segmenting instance")
    scene = GaussianScene.load(args.scene)
    pts = scene.positions_world()
    op = scene.opacities if scene.opacities is not None else np.full(len(pts), 0.7)
    col = np.clip(scene.colors, 0, 1) if scene.colors is not None else np.full((len(pts), 3), 0.7)
    scl = scene.scales_world() if scene.scales is not None else np.full((len(pts), 3), 0.006)
    scl = np.clip(scl, 1e-4, 0.12)
    # bake alignment rotation into per-gaussian quats once
    quats = scene.rotations if scene.rotations is not None else np.tile([0., 0., 0., 1.], (len(pts), 1))
    quats = _batch_quat_multiply(scene.transform_to_world.quaternion, quats)

    bx = np.asarray(args.bbox, dtype=float)
    inst = np.all((pts >= bx[:3]) & (pts <= bx[3:]), axis=1)
    rng = np.random.default_rng(0)
    bg_idx = np.nonzero(~inst & (op > 0.03))[0]
    if len(bg_idx) > args.max_bg:
        bg_idx = rng.choice(bg_idx, args.max_bg, replace=False)
    in_idx = np.nonzero(inst)[0]
    if len(in_idx) > args.max_instance:  # keep the most opaque
        in_idx = in_idx[np.argsort(-op[in_idx])[: args.max_instance]]
    print(f"    background {len(bg_idx):,}, instance {len(in_idx):,} gaussians")

    W, H = 640, 520
    f = W / (2 * np.tan(np.radians(args.fov) / 2))
    eye = np.asarray(args.eye, float)
    lookat = np.asarray(args.lookat, float)

    print("[2] rasterizing static background (once)")
    bg_img = render_gaussians(
        pts[bg_idx], scl[bg_idx], quats[bg_idx], col[bg_idx], op[bg_idx],
        eye, lookat, f, W, H, base=np.full((H, W, 3), 0.04),
    )

    inst_pts0, inst_scl = pts[in_idx], scl[in_idx]
    inst_quats0, inst_col, inst_op = quats[in_idx], col[in_idx], op[in_idx]
    ghost_col = 0.25 * inst_col + 0.75 * np.array([0.25, 0.45, 1.0])
    pose0 = real.pose_at(real.start_time)

    t0, t1 = real.start_time, real.end_time
    n_frames = int((t1 - t0) * args.fps / args.speed)
    ts = t0 + np.arange(n_frames) * (t1 - t0) / max(n_frames - 1, 1)
    ts = np.concatenate([ts, np.full(int(1.5 * args.fps), t1)])

    def moved(traj, t):
        delta = traj.pose_at(t).compose(pose0.inverse())
        return (delta.transform_points(inst_pts0),
                _batch_quat_multiply(delta.quaternion, inst_quats0))

    def panel(sim_traj, t):
        gp, gq = moved(real, t)
        sp, sq = moved(sim_traj, t)
        merged_pts = np.concatenate([gp, sp])
        merged_scl = np.concatenate([inst_scl, inst_scl])
        merged_q = np.concatenate([gq, sq])
        merged_col = np.concatenate([ghost_col, inst_col])
        merged_op = np.concatenate([inst_op * 0.45, inst_op])
        return render_gaussians(merged_pts, merged_scl, merged_q, merged_col,
                                merged_op, eye, lookat, f, W, H, base=bg_img)

    cap_b = _caption(W, f"标定前 BEFORE   {param} = {v0:.2f}")
    truth_s = f"  (真值 truth {truth:.2f})" if truth is not None else ""
    cap_a = _caption(W, f"标定后 AFTER   {param} = {v1:.2f}{truth_s}")

    print(f"[3] rendering {len(ts)} frames x 2 panels")
    tmp = Path(tempfile.mkdtemp(prefix="pgtwin_gs_"))
    from PIL import Image, ImageDraw, ImageFont
    try:
        efont = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except Exception:
        efont = None
    for k, t_now in enumerate(ts):
        row = []
        for sim_traj in (before, after):
            img = np.clip(panel(sim_traj, t_now), 0, 1)
            frame = (img * 255).astype(np.uint8)
            pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil)
            err = np.linalg.norm(real.pose_at(t_now).position[:2]
                                 - sim_traj.pose_at(t_now).position[:2]) * 100
            draw.text((10, H - 26),
                      f"t = {t_now - t0:4.2f} s    position error: {err:5.2f} cm",
                      fill=(240, 242, 245), font=efont)
            row.append(np.asarray(pil))
        merged = np.hstack([np.vstack([cap_b, row[0]]), np.vstack([cap_a, row[1]])])
        footer = _caption(2 * W, "蓝色虚影 = 真实物体实际轨迹 (blue ghost = real object)"
                                 "   ·   CPU EWA gaussian splatting", height=38)
        plt.imsave(tmp / f"f_{k:05d}.png", np.vstack([merged, footer]))
        if (k + 1) % 25 == 0:
            print(f"    frame {k + 1}/{len(ts)}")

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    out = VIDEOS_DIR / f"{exp.name}_splat.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
         "-i", str(tmp / "f_%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(out)],
        check=True,
    )
    shutil.rmtree(tmp)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
