#!/usr/bin/env python3
"""Build a self-contained HTML demo report from all experiment outputs.

    python scripts/build_demo_report.py            # -> outputs/demo_report.html

Collects every outputs/<experiment>/result.json + plots (base64-embedded, so
the single HTML file can be opened anywhere / sent to anyone) plus the
gaussian-scene preview figures.  Re-run after new experiments.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUTS = PROJECT_ROOT / "outputs"
GAUSSIAN = PROJECT_ROOT / "data" / "gaussian"

#: demo registry: (experiment dir, title, rerun command, description)
DEMOS = [
    (
        "push_box_001",
        "Demo 1 · Push + 摩擦标定（MockSimulator）",
        "python scripts/run_calibration.py --example push_box",
        "合成“真实”推动数据（真值摩擦 0.8，含 1mm 测量噪声），从错误初值 0.2 出发，"
        "网格搜索通过 real-sim 轨迹差恢复摩擦系数。解析物理模型，零外部依赖。",
    ),
    (
        "push_box_mujoco_001",
        "Demo 2 · Push + 摩擦标定（MuJoCo 全物理引擎）",
        "python scripts/run_calibration.py --example push_box_mujoco",
        "同一标定框架切换到 MuJoCo 后端（仅改一行配置）。柔顺推杆（mocap 目标 + 软 weld）"
        "模拟阻抗控制推动；EE replay error 如实报告柔顺滞后。",
    ),
    (
        "push_desk_real_scene_001",
        "Demo 3 · 真实 3DGS 场景 → 实例分离 → 碰撞代理 → 标定（S2–S7 全链）",
        "python scripts/run_calibration.py --example push_desk_real_scene",
        "真实教室 3DGS 重建（230 万高斯，ModelScope 下载）经 RANSAC 地面对齐 + 米制尺度化后，"
        "手工 bbox 分离讲桌实例（6,545 高斯），由真实高斯点云导出碰撞代理，"
        "在实例真实位置执行推动标定，输出绑定真实资产的 Physical Gaussian Twin。",
    ),
    (
        "push_chair_playroom_001",
        "Demo 4 · 真实场景 #2：playroom 红色儿童椅（Deep Blending 基准）",
        "python scripts/run_calibration.py --example push_chair_playroom",
        "3DGS 原论文基准场景 playroom（192 万高斯）。米制尺度双锚点交叉验证"
        "（天花板 4.15 单位→2.4m，书桌面 1.28 单位→0.74m）；分离地板上的红色儿童椅"
        "（0.84×0.79m，高 0.55m），由其高斯导出碰撞代理并在原位完成推动摩擦标定。",
    ),
    (
        "slide_truck_001",
        "Demo 5 · 真实场景 #3：Tanks&Temples 卡车（车辆尺度泛化）",
        "python scripts/run_calibration.py --example slide_truck",
        "经典 truck 基准场景（206 万高斯，重建近似米制：车长 6.7 单位≈6.7m）。"
        "分离整车实例（16.6 万实心高斯，6.7×3.7×2.6m 代理），以锁轮滑移"
        "（橡胶-混凝土 μ≈0.6-0.8 的真实区间）标定摩擦 —— 同一管线从 6cm 方盒到 "
        "6.7m 卡车的尺度泛化演示。准静态滑移距离与质量无关，摩擦保持可辨识。",
    ),
    (
        "push_vase_garden_001",
        "★ Demo 9 · 全链路自建：185 张照片 → 本机训练 3DGS → 标定（无借用资产）",
        "python scripts/run_calibration.py --example push_vase_garden",
        "Mip-NeRF360 garden 官方采集（5187×3361×185），本机 RTX 4090D 用 gsplat 从零训练："
        "15k 迭代 655 秒、1004 万高斯、留出视角 PSNR 25.06 dB。自研 COLMAP 二进制解析器读位姿、"
        "RANSAC 地面对齐、桌面双锚点定尺度（0.568 vs 0.570）、桌上花瓶实例分离、"
        "实测圆柱碰撞代理，桌面 0.74m 高度推动标定。这是唯一不借用任何现成 PLY 的完整链路。",
    ),
    (
        "lift_push_hierarchical_001/stage1_lift",
        "Demo 6a · 分层标定 Stage 1：LIFT → mass（力信号）",
        "python scripts/run_hierarchical.py --example lift_push",
        "运动学抓取下 LIFT 轨迹由夹爪强加、不含质量信息 —— 质量的观测量是夹持力 "
        "F = m(a+g)。本级用 ForceLoss（单位 N）在力信号上搜索质量，"
        "并与闭式估计 m = F̄z/g 交叉验证。",
    ),
    (
        "lift_push_hierarchical_001/stage2_push",
        "Demo 6b · 分层标定 Stage 2：PUSH → friction（轨迹信号）",
        "python scripts/run_hierarchical.py --example lift_push",
        "Stage 1 标定出的质量被传入本级（溯源记录于 stage_input_physics），"
        "再由推动轨迹标定摩擦 —— 分阶段、动作敏感性驱动，规避 friction×mass 不可辨识问题。",
    ),
]

#: demo videos: (filename under outputs/videos/, caption)
VIDEOS = [
    ("push_vase_garden_001_splat.mp4",
     "★ 全链路自建场景 · Mip-NeRF360 garden：用官方 185 张照片在 RTX 4090D 上"
     "从零训练 3DGS（gsplat 1.5.1，15k 迭代 655 秒，留出视角 PSNR 25.06 dB，1004 万高斯），"
     "RANSAC 对齐后以桌面高度+直径双锚点定尺度（0.568/0.570 吻合 0.4%），"
     "分离桌上干花花瓶为实例，碰撞代理用实测瓶身圆柱（半径 9cm、高 26cm，"
     "干花不入物理），桌面高度 0.74m 处推动标定：μ 0.2 → 0.55（真值 0.55）。"),
    ("push_chair_playroom_001_splat.mp4",
     "playroom · CPU EWA 高斯溅射渲染（照片级）：懒人沙发以其自身高斯在真实场景中"
     "运动，字母地垫/书架/玩具全部连续呈现。蓝色虚影 = 真实位置；"
     "标定前（μ=0.20）误差 4.3cm，标定后（μ=0.60）0.07cm。"),
    ("slide_truck_001_splat.mp4",
     "truck · EWA 溅射渲染：6.7m 卡车整车滑移标定，误差 11.5cm → 0.2cm"
     "（车辆尺度泛化）。"),
    ("push_desk_real_scene_001_splat.mp4",
     "教室 · EWA 溅射渲染：黑板板书清晰可读，讲桌原位推动，"
     "误差 3.7cm → 0.10cm。"),
    ("push_box_mujoco_001_3d.mp4",
     "MuJoCo 3D 引擎渲染 · 蓝色半透明 = 真实轨迹，橙色 = 仿真：标定前滑过头，"
     "标定后贴合（μ 0.20 → 0.75，真值 0.8）。"),
]

#: per-demo plot files (label, filename)
PLOTS = [
    ("参数-损失曲线（含真值参考线）", "optimization_curve.png"),
    ("物体 XY 轨迹：real vs sim（标定前/后）", "trajectory_xy.png"),
    ("位置误差随时间（标定前/后）", "error_time.png"),
    ("EE 回放误差（动作保真度审计）", "ee_replay_error.png"),
]

SCENE_FIGURES = [
    ("教室：讲桌实例三视图定位（红色 = 被标定物体）", "classroom_desk_preview.png"),
    ("playroom：红色儿童椅实例定位（侧视图中可见楼梯与家具）", "playroom_chair_preview.png"),
    ("truck：整车实例定位（侧视图红色轮廓 = 驾驶室+平板货斗）", "truck_vehicle_preview.png"),
    ("playroom 实拍参考照片（Deep Blending 数据集）", "playroom_ref.jpg"),
    ("truck 实拍参考照片（Tanks&Temples 数据集）", "truck_ref.jpg"),
    ("教室：场景俯视图（z<0.6m 切片）", "classroom_scene_topdown.png"),
]


def b64_video(path: Path) -> str | None:
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{data}"


def video_section() -> str:
    items = ""
    for fname, caption in VIDEOS:
        src = b64_video(OUTPUTS / "videos" / fname)
        if src:
            items += (f'<figure><video controls loop muted playsinline '
                      f'src="{src}"></video><figcaption>{caption}</figcaption></figure>')
    if not items:
        return ""
    return f"""
<section>
  <h2>演示视频 · 标定前 vs 标定后</h2>
  <p>最直观的效果展示：同一条真实交互轨迹，分别用<b>标定前</b>与<b>标定后</b>的物理参数
  在仿真中回放。生成命令：<code>python scripts/render_video.py splat|anim|mujoco3d
  --experiment outputs/&lt;id&gt;</code>；视频文件位于 <code>outputs/videos/</code>。</p>
  <div class="vgrid">{items}</div>
</section>"""


def b64_img(path: Path) -> str | None:
    if not path.exists():
        return None
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def fmt_params(d: dict) -> str:
    return ", ".join(f"{k} = {v:.4g}" for k, v in d.items())


def metric_cards(result: dict) -> str:
    """Key metric tiles for one experiment."""
    init_p = fmt_params(result.get("initial_params", {}))
    opt_p = fmt_params(result.get("optimized_params", {}))
    gt = result.get("ground_truth_params") or {}
    gt_str = fmt_params({k: v for k, v in gt.items()
                         if k in result.get("optimized_params", {})}) or "—"
    im = result.get("initial_metrics", {})
    fm = result.get("final_metrics", {})
    imp = result.get("improvement", {})
    if "rmse_position_cm" in fm:
        err_label = "物体轨迹 RMSE"
        err_txt = f"{im['rmse_position_cm']:.2f} → {fm['rmse_position_cm']:.2f} cm"
        imp_txt = f"{imp.get('rmse_position_cm', 0):.1f}%"
    else:
        err_label = "夹持力 MAE"
        err_txt = f"{im.get('force_mae_n', 0):.2f} → {fm.get('force_mae_n', 0):.2f} N"
        imp_txt = f"{imp.get('force_mae_n', 0):.1f}%"
    ee = result.get("ee_replay_rmse_m")
    cards = [
        ("初始参数", init_p),
        ("优化后参数", f"<b>{opt_p}</b>"),
        ("真值（合成数据）", gt_str),
        (err_label, err_txt),
        ("误差改善", f"<b>{imp_txt}</b>"),
        ("EE 回放 RMSE", f"{ee * 100:.3f} cm" if ee is not None else "—"),
        ("评估次数", str(result.get("n_evaluations", "—"))),
    ]
    tiles = "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in cards
    )
    return f'<div class="cards">{tiles}</div>'


def demo_section(exp_rel: str, title: str, cmd: str, desc: str) -> str:
    exp_dir = OUTPUTS / exp_rel
    result_path = exp_dir / "result.json"
    if not result_path.exists():
        return (
            f'<section><h2>{title}</h2><p class="missing">未找到 '
            f"outputs/{exp_rel}/result.json —— 先运行：<code>{cmd}</code></p></section>"
        )
    result = json.loads(result_path.read_text())
    imgs = ""
    for label, fname in PLOTS:
        src = b64_img(exp_dir / fname)
        if src:
            imgs += (
                f'<figure><img src="{src}" loading="lazy" alt="{label}">'
                f"<figcaption>{label}</figcaption></figure>"
            )
    return f"""
<section>
  <h2>{title}</h2>
  <p>{desc}</p>
  <pre class="cmd">$ {cmd}</pre>
  {metric_cards(result)}
  <div class="grid">{imgs}</div>
</section>"""


def scene_section() -> str:
    figs = ""
    for label, fname in SCENE_FIGURES:
        src = b64_img(GAUSSIAN / fname)
        if src:
            figs += (
                f'<figure><img src="{src}" loading="lazy" alt="{label}">'
                f"<figcaption>{label}</figcaption></figure>"
            )
    if not figs:
        return ""
    return f"""
<section>
  <h2>附录 · 真实 3DGS 场景处理过程（Demo 3 的输入）</h2>
  <p>教室场景由 <code>scripts/prepare_gaussian_scene.py</code> 完成 RANSAC 地面对齐、
  米制尺度化、裁剪与去漂浮物。</p>
  <p><b>交互式 3D 查看（推荐）</b>：把以下任一文件拖入
  <a href="https://superspl.at/editor" target="_blank">superspl.at/editor</a>
  （均已对齐摆正，<span style="color:#d22">被标定实例染成亮红色</span>）：
  <code>data/gaussian/classroom_desk_highlight.splat</code>（教室，65MB）、
  <code>data/gaussian/playroom_chair_highlight.splat</code>（游戏房，48MB）、
  <code>data/gaussian/truck_vehicle_highlight.splat</code>（卡车广场，30MB）。
  对应的 <code>*_only.splat</code> 为单独实例。
  文件由 <code>scripts/export_instance_splat.py</code> 生成。</p>
  <p style="color:var(--muted);font-size:13px">关于观感：此场景为第三方随手拍摄的
  演示资产（深色反光课桌是 3DGS 最难的表面，桌面实测自带 ±2cm 几何噪声）——
  本项目不做重建、只做物理标定，重建质量由上游采集与训练决定；
  几何噪声对标定的影响已按"几何-摩擦耦合"规则纳入实验协议（规则 #7）。</p>
  <div class="grid">{figs}</div>
</section>"""


def summary_table() -> str:
    rows = ""
    for exp_rel, title, _, _ in DEMOS:
        rp = OUTPUTS / exp_rel / "result.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text())
        opt = fmt_params(r.get("optimized_params", {}))
        gt = r.get("ground_truth_params") or {}
        gt_s = fmt_params({k: v for k, v in gt.items()
                           if k in r.get("optimized_params", {})}) or "—"
        fm = r.get("final_metrics", {})
        imp = r.get("improvement", {})
        if "rmse_position_cm" in fm:
            err = f"{fm['rmse_position_cm']:.2f} cm"
            impv = f"{imp.get('rmse_position_cm', 0):.1f}%"
        else:
            err = f"{fm.get('force_mae_n', 0):.2f} N"
            impv = f"{imp.get('force_mae_n', 0):.1f}%"
        short = title.split("·", 1)[1].strip() if "·" in title else title
        rows += (f"<tr><td>{short}</td><td>{opt}</td><td>{gt_s}</td>"
                 f"<td>{err}</td><td>{impv}</td></tr>")
    return f"""
<table>
  <thead><tr><th>Demo</th><th>恢复参数</th><th>真值</th>
  <th>最终误差</th><th>改善</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""


CSS = """
:root { --fg:#1a1d21; --muted:#5c6670; --line:#e3e6ea; --accent:#0b6e4f;
        --bg:#fafbfc; --card:#ffffff; }
* { box-sizing:border-box; }
body { margin:0 auto; max-width:1060px; padding:32px 20px 80px;
       font:15px/1.65 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",
       sans-serif; color:var(--fg); background:var(--bg); }
h1 { font-size:26px; margin:0 0 4px; }
h2 { font-size:19px; margin:0 0 10px; padding-top:8px; }
.sub { color:var(--muted); margin:0 0 24px; }
section { background:var(--card); border:1px solid var(--line);
          border-radius:10px; padding:22px 24px; margin:18px 0; }
pre.cmd { background:#10151a; color:#c9f0dd; padding:10px 14px;
          border-radius:7px; overflow-x:auto; font-size:13px; }
.cards { display:flex; flex-wrap:wrap; gap:10px; margin:14px 0 6px; }
.card { border:1px solid var(--line); border-radius:8px;
        padding:8px 14px; background:var(--bg); min-width:130px; }
.card .k { font-size:12px; color:var(--muted); }
.card .v { font-size:14px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
        gap:14px; margin-top:14px; }
.vgrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr));
         gap:16px; margin-top:14px; }
.vgrid video { width:100%; border:1px solid var(--line); border-radius:6px;
               background:#000; }
figure { margin:0; }
figure img { width:100%; border:1px solid var(--line); border-radius:6px;
             background:#fff; }
figcaption { font-size:12.5px; color:var(--muted); margin-top:4px; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th,td { border:1px solid var(--line); padding:7px 11px; text-align:left; }
th { background:var(--bg); }
b { color:var(--accent); }
code { background:#eef1f4; padding:1px 5px; border-radius:4px; font-size:13px; }
a { color:var(--accent); }
.missing { color:#a33; }
.pipeline { color:var(--muted); font-size:13.5px; background:var(--bg);
            border:1px dashed var(--line); border-radius:8px; padding:10px 14px; }
"""


def main() -> int:
    sections = "".join(demo_section(*d) for d in DEMOS)
    html = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Physical Gaussian Digital Twin · Demo Report</title>
<style>{CSS}</style>
</head>
<body>
<h1>Physical Gaussian Digital Twin · Demo 报告</h1>
<p class="sub">机器人交互数据驱动的三维高斯物理数字孪生构建与自适应校准 ——
四个可复现 demo 的结果汇总。所有图片内嵌，本文件可单独发送 / 归档。</p>

<section>
  <h2>闭环总览</h2>
  <p class="pipeline">真实场景采集 → 3DGS 重建 → 实例级分离 → 初始物理属性 →
  机器人真实交互（push / lift）→ 仿真回放同一动作 → real-sim 轨迹/力差 →
  参数搜索 → <b>校准后的 Physical Gaussian Twin</b>（physical_object.json）</p>
  {summary_table()}
  <p style="color:var(--muted);font-size:13px">每个 demo 的完整产物
  （config.yaml / result.json / calibration_history.csv / 轨迹 CSV / 全部图表 /
  physical_object.json）位于 <code>outputs/&lt;experiment_id&gt;/</code>。</p>
</section>
{video_section()}
{sections}
{scene_section()}
</body>
</html>"""
    out = OUTPUTS / "demo_report.html"
    out.write_text(html, encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"demo report -> {out}  ({size_mb:.1f} MB, self-contained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
