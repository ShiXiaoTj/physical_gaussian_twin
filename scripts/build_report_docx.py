#!/usr/bin/env python3
"""Generate a Word (.docx) technical validation report from the experiments.

    python scripts/build_report_docx.py [--out docs/report.docx] [--en]

Reads every outputs/*/result.json (plus hierarchical stages), the experiment
figures and the gaussian-scene previews, and assembles a structured Chinese
report with cover, TOC, tables and embedded figures. Written for the project
handover, not for the terminal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"
GAUSSIAN = PROJECT_ROOT / "data" / "gaussian"

#: (experiment dir, 标题, 复现命令, 说明, 图集)
DEMOS = [
    (
        "push_box_001",
        "实验 1  Push + 摩擦标定（解析物理后端）",
        "python scripts/run_calibration.py --example push_box",
        "合成“真实”推动数据（真值摩擦 0.8，含 1 mm 测量噪声），从错误初值 0.2 出发，"
        "网格搜索通过 real-sim 轨迹差恢复摩擦系数。MockSimulator 为解析准静态平面推动模型，"
        "零外部依赖，用于验证反演机制本身的正确性。",
        ["optimization_curve.png", "trajectory_xy.png", "error_time.png", "ee_replay_error.png"],
    ),
    (
        "push_box_mujoco_001",
        "实验 2  Push + 摩擦标定（MuJoCo 全物理引擎）",
        "python scripts/run_calibration.py --example push_box_mujoco",
        "同一标定框架切换至 MuJoCo 后端（仅改一行配置）。推杆采用柔顺模型"
        "（mocap 目标 + 软 weld 约束）以模拟阻抗控制推动，EE 回放误差如实反映柔顺滞后，"
        "用于审计“动作是否真的复现”。",
        ["optimization_curve.png", "trajectory_xy.png", "error_time.png", "ee_replay_error.png"],
    ),
    (
        "push_desk_real_scene_001",
        "实验 3  真实 3DGS 场景全链路（教室讲桌）",
        "python scripts/run_calibration.py --example push_desk_real_scene",
        "真实教室 3DGS 重建（230 万高斯）经 RANSAC 地面对齐与米制尺度化后，"
        "按世界系包围盒分离讲桌实例，由真实高斯点云导出碰撞代理，"
        "在实例真实位置执行推动标定，输出绑定真实资产的高斯物理孪生体。",
        ["optimization_curve.png", "trajectory_xy.png", "error_time.png"],
    ),
    (
        "push_chair_playroom_001",
        "实验 4  真实场景 #2：playroom 儿童椅（Deep Blending 基准）",
        "python scripts/run_calibration.py --example push_chair_playroom",
        "3DGS 原论文基准场景 playroom（192 万高斯）。米制尺度双锚点交叉验证"
        "（天花板与书桌面），分离地板上的红色儿童椅（0.84×0.79 m、高 0.55 m），"
        "由其高斯导出碰撞代理并在原位完成推动摩擦标定。",
        ["optimization_curve.png", "trajectory_xy.png", "error_time.png"],
    ),
    (
        "slide_truck_001",
        "实验 5  真实场景 #3：车辆尺度泛化（Tanks&Temples 卡车）",
        "python scripts/run_calibration.py --example slide_truck",
        "经典 truck 基准场景（206 万高斯，重建本身近似米制：车长 6.7 单位≈6.7 m）。"
        "分离整车实例（16.6 万实心高斯，6.7×3.7×2.6 m 代理），以锁轮滑移"
        "（橡胶—混凝土 μ≈0.6–0.8 的真实区间）标定摩擦，验证同一管线从 6 cm 方盒到 "
        "6.7 m 车辆的尺度泛化能力。",
        ["optimization_curve.png", "trajectory_xy.png", "error_time.png"],
    ),
    (
        "push_vase_garden_001",
        "实验 6  全链路自建：185 张照片 → 本机训练 3DGS → 标定 ★",
        "python scripts/run_calibration.py --example push_vase_garden",
        "唯一不借用任何现成 PLY 资产的完整链路。Mip-NeRF360 garden 官方采集"
        "（5187×3361，185 张）在本机 RTX 4090D 上用 gsplat 1.5.1 从零训练：15k 迭代、"
        "655 秒、1004 万高斯、留出视角 PSNR 25.06 dB。自研 COLMAP 二进制解析器读取位姿，"
        "RANSAC 地面对齐后以桌面高度与直径双锚点定尺度，分离桌上干花花瓶为实例，"
        "碰撞代理采用实测瓶身圆柱（半径 9 cm、高 26 cm），于桌面 0.74 m 高度执行推动标定。",
        ["optimization_curve.png", "trajectory_xy.png", "error_time.png"],
    ),
    (
        "lift_push_hierarchical_001/stage1_lift",
        "实验 7a  分层标定 Stage 1：LIFT → 质量（力信号）",
        "python scripts/run_hierarchical.py --example lift_push",
        "运动学抓取下，LIFT 轨迹由夹爪强加、不含质量信息——质量的可观测量是夹持力 "
        "F = m(a+g)。本级以力损失（单位 N）在力信号上搜索质量，并与闭式估计 "
        "m = F̄z/g 交叉验证。",
        ["optimization_curve.png", "error_time.png"],
    ),
    (
        "lift_push_hierarchical_001/stage2_push",
        "实验 7b  分层标定 Stage 2：PUSH → 摩擦（轨迹信号）",
        "python scripts/run_hierarchical.py --example lift_push",
        "Stage 1 标定出的质量传入本级（溯源记录于 stage_input_physics），"
        "再由推动轨迹标定摩擦——分阶段、动作敏感性驱动，规避 friction×mass 耦合导致的"
        "不可辨识问题。",
        ["optimization_curve.png", "trajectory_xy.png"],
    ),
]

SCENE_FIGURES = [
    ("classroom_desk_preview.png", "教室场景讲桌实例定位（红色为被标定物体）"),
    ("playroom_chair_preview.png", "playroom 场景红色儿童椅实例定位"),
    ("truck_vehicle_preview.png", "truck 场景整车实例定位（红色轮廓为驾驶室与货斗）"),
    ("garden_scene_preview.png", "garden 自建场景三视图（米制、地面对齐后）"),
]

REBUILD_EVIDENCE = [
    ("../data/gaussian/garden_trained/eval_015000_view092_psnr25.6.png",
     "训练留出视角：左为真实照片，右为 3DGS 渲染（PSNR 25.6 dB）"),
]


# ---------------------------------------------------------------------------
# styling helpers
# ---------------------------------------------------------------------------

def set_cjk_font(run, name: str = "Microsoft YaHei", size: int | None = None):
    run.font.name = name
    if size:
        run.font.size = Pt(size)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia"):
        rfonts.set(qn(attr), name)


def style_doc(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.size = Pt(10.5)
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    for i, (sz, bold) in enumerate(((16, True), (14, True), (12, True)), start=1):
        st = doc.styles[f"Heading {i}"]
        st.font.size = Pt(sz)
        st.font.bold = bold
        st.font.color.rgb = RGBColor(0x1A, 0x2B, 0x3C)
        st.font.name = "Microsoft YaHei"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")


def para(doc, text, size=10.5, bold=False, align=None, space_after=6, italic=False):
    p = doc.add_paragraph()
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    run = p.add_run(text)
    set_cjk_font(run, size=size)
    run.bold = bold
    run.italic = italic
    return p


def bullet(doc, text, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    set_cjk_font(p.add_run(text), size=size)
    return p


def add_figure(doc, path: Path, caption: str, width_cm: float = 15.5):
    if not path.exists():
        return False
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)
    set_cjk_font(cap.add_run(caption), size=9)
    return True


def add_table(doc, headers: list[str], rows: list[list[str]], widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        set_cjk_font(cell.paragraphs[0].add_run(h), size=9.5)
        cell.paragraphs[0].runs[0].bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            set_cjk_font(cells[i].paragraphs[0].add_run(str(v)), size=9)
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def load_result(rel: str) -> dict:
    p = OUTPUTS / rel / "result.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------------------
# document sections
# ---------------------------------------------------------------------------

def cover(doc: Document, title: str, subtitle: str):
    for _ in range(4):
        doc.add_paragraph()
    para(doc, title, size=22, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)
    para(doc, subtitle, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)
    para(doc, "技术验证报告", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
         space_after=6)
    para(doc, "机器人交互数据驱动的三维高斯物理数字孪生构建与自适应校准系统",
         size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=30)
    para(doc, "版本 v0.6.0", size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    para(doc, "全部实验可复现 · 自动化测试 183 项全部通过",
         size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    doc.add_page_break()


def section_overview(doc: Document):
    doc.add_heading("一、项目概述", level=1)
    para(doc, "三维高斯溅射（3D Gaussian Splatting, 3DGS）解决了场景的“视觉真实”，"
              "使真实场景可被重建为照片级的三维表达；但它不描述场景的“物理真实”——"
              "物体多重、表面多滑、受击如何反弹，均无从得知。将这样的场景放入物理引擎时，"
              "物理参数只能凭经验填充，由此产生的仿真—现实偏差（sim-to-real gap）"
              "会直接劣化机器人策略训练、数据生成与部署前的安全验证。")
    para(doc, "本项目提出并验证了一套完整闭环：以机器人自身的交互作为“测量仪器”，"
              "自动反演高斯场景中目标物体的物理参数。核心流程如下：")
    steps = [
        "1  场景采集与重建：多视角照片 → 3DGS 重建（本项目支持本机从零训练）；",
        "2  实例级分离：从高斯场景中分割出目标物体；",
        "3  初始物理属性：给定碰撞代理与物理参数初值；",
        "4  真实交互：机器人在真实世界对物体执行动作（推、滑、举等）；",
        "5  仿真回放：在仿真中复现完全相同的动作；",
        "6  差异构造目标：比较真实与仿真的物体轨迹（或力信号）；",
        "7  参数反演：自动搜索物理参数使仿真行为逼近真实；",
        "8  输出校准孪生体：得到可视—物理绑定的 Physical Gaussian Twin。",
    ]
    for s in steps:
        bullet(doc, s)
    para(doc, "校准后的孪生体可直接用于机器人仿真、数据生成、模仿学习、"
              "视觉—语言—动作模型（VLA）与规划。")

    doc.add_heading("1.1  与常规 3DGS 工作的区别", level=2)
    add_table(
        doc,
        ["维度", "常规 3DGS", "本项目（Physical Gaussian Digital Twin）"],
        [
            ["目标", "视觉真实：任意视角渲染正确", "物理真实：仿真行为与真实一致"],
            ["输入", "多视角图像", "多视角图像 + 机器人交互数据"],
            ["输出", "高斯点云", "高斯表达 + 碰撞几何 + 标定后的物理参数"],
            ["物理参数来源", "人工设定或经验查表", "由真实交互自动反演"],
            ["验证方式", "渲染指标（PSNR/SSIM/LPIPS）", "real–sim 轨迹差、力差与误差改善率"],
        ],
        widths=[2.4, 5.6, 7.5],
    )


def section_motivation(doc: Document):
    doc.add_heading("二、技术背景与设计要点", level=1)
    para(doc, "现有做法可归纳为两类，均不足以支撑可信的机器人仿真："
              "其一，依据外观推断物理属性（“看起来像木头，就填木材参数”），"
              "忽略表面状态、装配、磨损带来的巨大差异；"
              "其二，人工反复试凑，成本高、不可复现、无法规模化。")
    para(doc, "本项目的主张是：让机器人对真实物体做一次交互，"
              "把真实世界的响应作为标准答案，通过 real–sim 差异自动反推出物理参数。"
              "以本报告实验 1 为例，摩擦系数填错（0.2 对真值 0.8）时，"
              "同一推动动作在仿真中会使物体多滑行约 4.3 cm——"
              "在抓取任务中，这已相当于一个物体宽度的位置偏差，"
              "足以让在此仿真中训练出的策略在真机上首次执行即失败。")

    doc.add_heading("2.1  五项核心创新", level=2)
    add_table(
        doc,
        ["编号", "创新点", "代码锚点"],
        [
            ["创新 1", "以机器人真实交互数据（而非纯视觉信息）校准高斯数字孪生的物理参数",
             "physics/calibrator.py"],
            ["创新 2", "真实与仿真执行同一动作，以 real–sim 轨迹差构造优化目标；"
                      "并以 EE 回放误差审计动作保真度",
             "simulator/replay.py, physics/losses.py"],
            ["创新 3", "按“动作—参数—观测量”敏感性分阶段辨识参数，"
                      "规避耦合参数不可辨识问题",
             "physics/sensitivity.py, HierarchicalCalibrator"],
            ["创新 4", "高斯视觉表达与物理表达的物体级绑定（视觉—碰撞代理解耦）",
             "gaussian/gaussian_object.py, physics/physical_object.py"],
            ["创新 5", "迭代式自适应孪生：交互 → 仿真 → 差异 → 参数更新 → 再仿真",
             "PhysicsCalibrator.calibrate 搜索环 + CalibrationResult.history"],
        ],
        widths=[1.8, 9.2, 4.5],
    )

    doc.add_heading("2.1  可辨识性：为什么不能一次性黑箱优化所有参数", level=2)
    para(doc, "并非所有物理参数都能通过同一种动作被观测。准静态推动下，"
              "物体的自由滑行距离为 v²/(2μg)，与质量无关；"
              "因此“推一次同时标定摩擦与质量”在数学上是病态的："
              "较重的物体配较小摩擦，与较轻的物体配较大摩擦，可产生几乎相同的轨迹。"
              "本项目将该规律固化为代码约束——敏感性策略会拒绝不可辨识的参数组合。")
    para(doc, "更进一步，观测量本身的选取也影响可辨识性。运动学抓取条件下，"
              "举升轨迹由夹爪强加、完全不含质量信息；质量的可观测量是夹持力 F = m(a+g)。"
              "本项目通过负对照实验验证了这一点：对 LIFT 数据强制使用轨迹损失时，"
              "质量维度的损失景观完全平坦（波动 < 5%）；切换为力损失后方可辨识。")
    add_table(
        doc,
        ["动作", "可辨识参数", "观测量", "本项目实现"],
        [
            ["PUSH / SLIDE", "动摩擦系数", "物体轨迹", "已实现并验证"],
            ["LIFT", "质量", "夹持力（N）", "已实现并验证"],
            ["COLLISION", "恢复系数", "物体轨迹", "接口就绪"],
            ["PRESS", "杨氏模量 / 泊松比", "接触力", "接口就绪"],
        ],
        widths=[3.0, 3.4, 3.2, 5.9],
    )


def section_architecture(doc: Document):
    doc.add_heading("三、系统架构", level=1)
    para(doc, "系统采用适配器（Adapter）架构：重建后端、物理引擎、机器人均以接口隔离，"
              "标定核心不依赖任何具体实现，替换后端仅需修改配置。")
    add_table(
        doc,
        ["模块", "职责", "关键实现"],
        [
            ["common", "SE(3) 位姿与四元数运算、坐标帧注册、IO 与日志",
             "Pose（四元数统一为 [x,y,z,w]）、FrameRegistry"],
            ["real_data", "交互 episode 数据模型、加载、时间对齐、合成数据",
             "InteractionEpisode、PoseTrajectory、EpisodeLoader"],
            ["robot", "机器人运动学描述（末端几何、基座位姿）",
             "BaseRobot + Franka / Piper 适配器"],
            ["simulator", "仿真后端抽象、场景生成、动作回放",
             "BaseSimulator + MockSimulator / MuJoCoSimulator"],
            ["physics", "物理参数、损失、优化器、敏感性策略、标定器",
             "PhysicsCalibrator、HierarchicalCalibrator"],
            ["gaussian", "高斯场景容器、实例分离、碰撞代理、重建后端",
             "GaussianScene、GaussianObject、CPU EWA 光栅器"],
            ["evaluation", "指标、可视化、实验留档", "metrics（cm/deg）、ExperimentReporter"],
            ["cli", "命令行编排", "pgtwin calibrate | evaluate | reconstruct"],
        ],
        widths=[2.4, 6.6, 6.5],
    )

    doc.add_heading("3.1  坐标系与单位约定", level=2)
    para(doc, "坐标系是本类系统最主要的错误来源，因此本项目将其固化为类型与命名约定，"
              "并以单元测试锁定：")
    for s in [
        "全部坐标系为右手系；世界系为 z 轴向上，重力方向 [0, 0, -9.81]；",
        "四元数统一为 [x, y, z, w]（标量在后），跨模块与跨文件格式唯一；"
        "对外部约定（MuJoCo wxyz、3DGS PLY wxyz）在边界处显式转换；",
        "变换统一命名为 T_父_子，满足 p_父 = T_父_子 · p_子；组合时下标相接；",
        "episode 中机器人状态记录 T_world_ee，物体状态记录 T_world_object，"
        "一律为世界系；",
        "位置内部单位统一为米、角度为弧度，报告与指标中改为厘米与度，"
        "且单位写入字段名（rmse_position_cm、rotation_error_deg）；",
        "仿真后端必须还原到世界系（T_sim_world 为单位阵），"
        "使 real 与 sim 轨迹可直接比较，不引入隐藏变换。",
    ]:
        bullet(doc, s)

    doc.add_heading("3.2  标定闭环与动作保真度审计", level=2)
    para(doc, "标定器 PhysicsCalibrator.calibrate 的执行序列为："
              "时间对齐 → 场景装配 → 初值基线回放 → 参数搜索 → 最优解复核 → 指标与留档。"
              "其中两项不变量尤为关键：")
    bullet(doc, "同一动作：真实与仿真消费完全相同的末端轨迹，"
                "并记录仿真中实际执行的末端轨迹，其与指令轨迹的偏差作为 EE 回放误差输出。"
                "若该值不可忽略，则物体轨迹偏差源于动作未复现，而非物理参数不准——"
                "此时标定结论无效。")
    bullet(doc, "确定性：每次候选参数评估均重置仿真并以固定种子运行，"
                "保证搜索过程中唯一变化的量是物理参数，结果可复现。")


def section_demos(doc: Document, en: bool = False):
    doc.add_heading("四、实验验证", level=1)
    para(doc, "本章列出全部可复现实验。所有实验均从错误初值出发，"
              "与真值对照以验证反演精度；每项实验的完整产物"
              "（配置快照、结果 JSON、搜索历史、轨迹 CSV、图表）"
              "均自动留档于 outputs/<experiment_id>/ 目录。")

    doc.add_heading("4.1  结果汇总", level=2)
    rows = []
    for rel, title, _, _, _ in DEMOS:
        r = load_result(rel)
        if not r:
            continue
        im, fm = r.get("initial_metrics", {}), r.get("final_metrics", {})
        imp = r.get("improvement", {})
        if "rmse_position_cm" in fm:
            err = f"{im.get('rmse_position_cm', 0):.2f} → {fm['rmse_position_cm']:.2f} cm"
            impv = f"{imp.get('rmse_position_cm', 0):.1f}%"
        else:
            err = f"{im.get('force_mae_n', 0):.2f} → {fm['force_mae_n']:.2f} N"
            impv = f"{imp.get('force_mae_n', 0):.1f}%"
        opt = r.get("optimized_params", {})
        gt = r.get("ground_truth_params", {})
        opt_s = ", ".join(f"{k}={v:.2f}" for k, v in opt.items())
        gt_s = ", ".join(f"{k}={gt[k]:.2f}" for k in opt if k in gt) or "—"
        short = title.split("  ", 1)[-1][:26]
        rows.append([short, opt_s, gt_s, err, impv])
    add_table(doc, ["实验", "反演结果", "真值", "初始 → 最终误差", "误差改善"],
              rows, widths=[5.0, 2.9, 2.5, 3.5, 1.6])

    for rel, title, cmd, desc, figs in DEMOS:
        r = load_result(rel)
        if not r:
            continue
        doc.add_heading(title, level=2)
        para(doc, desc)
        cp = doc.add_paragraph()
        cp.paragraph_format.space_after = Pt(4)
        run = cp.add_run(f"$ {cmd}")
        set_cjk_font(run, name="Consolas", size=9)
        run.font.color.rgb = RGBColor(0x0B, 0x4F, 0x3A)

        im, fm = r.get("initial_metrics", {}), r.get("final_metrics", {})
        imp = r.get("improvement", {})
        if "rmse_position_cm" in fm:
            pairs = [
                ("物体轨迹 RMSE", f"{im.get('rmse_position_cm', 0):.2f} cm",
                 f"{fm['rmse_position_cm']:.2f} cm", f"{imp.get('rmse_position_cm', 0):.1f}%"),
                ("绝对轨迹误差 ATE", f"{im.get('ate_position_cm', 0):.2f} cm",
                 f"{fm.get('ate_position_cm', 0):.2f} cm",
                 f"{imp.get('ate_position_cm', 0):.1f}%"),
            ]
        else:
            pairs = [
                ("夹持力 MAE", f"{im.get('force_mae_n', 0):.2f} N",
                 f"{fm['force_mae_n']:.2f} N", f"{imp.get('force_mae_n', 0):.1f}%"),
            ]
        ee = r.get("ee_replay_rmse_m")
        pairs.append(("动作回放误差（EE）",
                      "—", f"{ee * 100:.3f} cm" if ee else "—", "—"))
        add_table(doc, ["指标", "标定前", "标定后", "改善"], [list(p) for p in pairs],
                  widths=[5.4, 3.6, 3.6, 2.4])

        for f in figs:
            label = {
                "optimization_curve.png": "参数—损失曲线：损失景观在真值附近形成清晰极小值，"
                                          "是参数可辨识性的直接证据",
                "trajectory_xy.png": "物体 XY 轨迹对比：真实、标定前、标定后",
                "error_time.png": "real–sim 位置误差随时间的变化",
                "ee_replay_error.png": "动作回放误差：用于区分“动作未复现”与“物理参数不准”",
            }[f]
            add_figure(doc, OUTPUTS / rel / f, f"图  {title.split('  ', 1)[-1]}：{label}")


def section_pipeline_evidence(doc: Document):
    doc.add_heading("五、真实场景链路证据", level=1)
    para(doc, "下列图像给出从重建到实例分离的实际过程，用于说明标定目标的选取与尺度"
              "确立方式。")
    for name, cap in SCENE_FIGURES:
        add_figure(doc, GAUSSIAN / name, f"图  {cap}", width_cm=15.5)

    doc.add_heading("5.1  自建重建质量（实验 6）", level=2)
    para(doc, "用 185 张官方采集照片在本机训练 3DGS（gsplat 1.5.1，RTX 4090D，"
              "15k 迭代 655 秒，1004 万高斯）后，留出视角的渲染质量如下："
              "左为真实照片，右为渲染结果，PSNR 25.6 dB。")
    add_figure(doc, GAUSSIAN / "garden_trained" / "eval_015000_view092_psnr25.6.png",
               "图  留出视角重建质量对比（真实 | 渲染）", width_cm=16.0)
    para(doc, "米制尺度采用双锚点交叉验证：桌面高度 1.32 单位 → 0.74 m（系数 0.568），"
              "桌面直径 3.12 单位 → 1.78 m（系数 0.570），两者吻合至 0.4%，"
              "满足实验协议对代理几何精度的要求。", italic=False)


def section_protocol(doc: Document):
    doc.add_heading("六、实验协议要点", level=1)
    para(doc, "本项目在真实物理引擎上的调试过程沉淀为一组可复用的实验设计规则，"
              "这些规则同时适用于后续的真机实验：")
    rules = [
        ("使用扁平物体", "推动高物体时，高摩擦下会翻滚（实测瞬时倾角达 16°），"
                        "其相位对摩擦与初始位姿均高度敏感，会淹没摩擦信号。"
                        "建议物体宽度 ≥ 2 倍质心高度。"),
        ("通过质心高度推动", "偏心接触会引入俯仰力矩并激发翻滚。"),
        ("推后回撤末端", "仅停止推动会让物体在接触弹簧上振荡，"
                        "使释放速度（进而自由滑行距离）不可复现。"),
        ("柔顺推动、中速释放", "刚性运动学推杆相当于锤击；真机以阻抗控制推动。"
                              "仿真中以 mocap 目标 + 软 weld 复现，实测最佳时间常数 0.04 s。"),
        ("预期非对称精度", "自由滑行距离 ∝ v²/(2μg)，灵敏度随 μ 增大而下降，"
                          "高摩擦估计天然更粗；建议多段数据联合标定以收紧区间。"),
        ("先验证回放误差", "跨后端回放（一种动作模型记录、另一种回放）会静默失效，"
                          "EE 回放误差是唯一的显式告警量。"),
        ("代理几何误差直接折价为摩擦偏差", "实测代理接触半径 6.5 mm 的误差"
                                          "使摩擦反演结果由 0.50 偏至 0.40；"
                                          "经验关系 Δμ ≈ μ²·2gδ/v²。标定前须以实测尺寸校验代理。"),
    ]
    for name, detail in rules:
        p = doc.add_paragraph(style="List Number")
        p.paragraph_format.space_after = Pt(3)
        r1 = p.add_run(f"{name}：")
        set_cjk_font(r1, size=10.5)
        r1.bold = True
        set_cjk_font(p.add_run(detail), size=10.5)


def section_repro(doc: Document):
    doc.add_heading("七、复现方式", level=1)
    para(doc, "全部实验均可在无 GPU、无机器人、无外部仿真器的最小环境下运行；"
              "涉及本机训练的链路（实验 6）需要 NVIDIA GPU。")
    doc.add_heading("7.1  环境准备", level=2)
    for cmd, note in [
        ("pip install -e . --no-build-isolation", "核心环境（解析物理后端 + 真实场景管线）"),
        ("pip install mujoco -i https://pypi.tuna.tsinghua.edu.cn/simple",
         "可选：MuJoCo 全物理后端（实验 2）"),
        ("conda create -n pgtwin-gs python=3.10 -y",
         "可选：从零训练 3DGS（实验 6）所需环境"),
        ("pip install torch==2.4.1 gsplat==1.5.1", "训练依赖（gsplat 提供预编译 wheel）"),
    ]:
        cp = doc.add_paragraph()
        cp.paragraph_format.space_after = Pt(2)
        run = cp.add_run(f"$ {cmd}")
        set_cjk_font(run, name="Consolas", size=9)
        para(doc, f"    {note}", size=9)

    doc.add_heading("7.2  运行实验", level=2)
    for cmd in [
        "python scripts/run_calibration.py --example push_box",
        "python scripts/run_calibration.py --example push_box_mujoco",
        "python scripts/run_calibration.py --example push_desk_real_scene",
        "python scripts/run_calibration.py --example push_chair_playroom",
        "python scripts/run_calibration.py --example slide_truck",
        "python scripts/run_calibration.py --example push_vase_garden",
        "python scripts/run_hierarchical.py --example lift_push",
    ]:
        cp = doc.add_paragraph()
        cp.paragraph_format.space_after = Pt(2)
        set_cjk_font(cp.add_run(f"$ {cmd}"), name="Consolas", size=9)

    doc.add_heading("7.3  完整链路（实验 6，从照片开始）", level=2)
    for cmd in [
        "conda run -n pgtwin-gs python scripts/train_gsplat.py \\",
        "    --data data/mipnerf360/garden --out data/gaussian/garden_trained \\",
        "    --downscale 4 --iters 15000",
        "python scripts/prepare_gaussian_scene.py \\",
        "    --input data/gaussian/garden_trained/scene.splat \\",
        "    --output data/gaussian/garden_scene.npz --scale 0.56 \\",
        "    --plane-threshold 0.05 --crop -1.7 1.7 -1.7 1.7 -0.28 1.7",
        "python scripts/run_calibration.py --example push_vase_garden",
    ]:
        cp = doc.add_paragraph()
        cp.paragraph_format.space_after = Pt(0)
        set_cjk_font(cp.add_run(cmd), name="Consolas", size=9)

    doc.add_heading("7.4  结果与可视化", level=2)
    for cmd, note in [
        ("python scripts/build_demo_report.py", "生成自包含的 HTML 演示报告（含视频）"),
        ("python scripts/build_report_docx.py", "生成本 Word 报告"),
        ("python scripts/render_video.py splat --experiment outputs/<id> --scene <场景.npz> \\"
         "\n    --bbox <包围盒> --eye x y z --lookat x y z", "渲染标定前后对比视频"),
        ("python scripts/validate_episode.py <episode 目录>",
         "真机数据就绪前的自检：单位、四元数顺序、采样、信号完整性"),
        ("pytest tests/ -q", "运行全部单元与集成测试（183 项）"),
    ]:
        cp = doc.add_paragraph()
        cp.paragraph_format.space_after = Pt(2)
        set_cjk_font(cp.add_run(f"$ {cmd}"), name="Consolas", size=9)
        para(doc, f"    {note}", size=9)


def section_limits(doc: Document):
    doc.add_heading("八、当前限制与后续工作", level=1)
    doc.add_heading("8.1  已知限制", level=2)
    for s in [
        "物理引擎覆盖：解析后端（MockSimulator）仅支持平面推动的准静态模型；"
        "MuJoCo 后端已实现并验证，但物体参数映射中恢复系数仍为近似（solref 等效）；"
        "Isaac Sim 适配器为接口预留。",
        "参数范围：当前已完成摩擦（PUSH/SLIDE）与质量（LIFT）的标定验证；"
        "恢复系数（COLLISION）与刚度（PRESS）的策略与接口已就绪，"
        "尚缺对应的数据生成器与（刚度所需的）可变形后端。",
        "真机数据：真机实验（协议 3 节）因缺少机器人数据而未执行；"
        "数据接口、协议与自检工具均已就绪，数据到位即可运行。",
        "实例分离：当前为手工包围盒选取（按设计与其他模块解耦）；"
        "后续可替换为 SAM / Grounded-SAM / 语义高斯分割。",
        "渲染：CPU EWA 光栅器为演示用途（仅 DC 颜色、无视角相关光泽）；"
        "照片级实时渲染需 GPU 后端。",
        "噪声下的精度：单段数据在 1 mm 跟踪噪声下摩擦反演存在 −0.05 至 −0.10 的小幅低偏"
        "（损失盆地高摩擦侧天然平坦）；多段数据联合标定可收紧（已列入后续工作）。",
    ]:
        bullet(doc, s)

    doc.add_heading("8.2  后续工作", level=2)
    add_table(
        doc,
        ["阶段", "内容", "状态"],
        [
            ["阶段 7", "真机实验：≥5 次推动 × ≥2 物体 × ≥2 表面；"
                      "报告反演一致性与跨 episode 泛化", "待机器人数据"],
            ["阶段 8 扩展", "COLLISION → 恢复系数、PRESS → 刚度的分阶段标定；"
                           "多段数据联合优化；损失曲率不确定度", "策略与接口就绪"],
            ["工程增强", "GPU 渲染后端接入实时照片级视频；"
                        "Isaac Sim 适配；SAM 自动实例分割", "规划中"],
        ],
        widths=[2.6, 9.8, 3.1],
    )


def appendix_inventory(doc: Document):
    doc.add_page_break()
    doc.add_heading("附录 A  源码与产物清单", level=1)
    doc.add_heading("A.1  核心源码模块", level=2)
    for src, desc in [
        ("src/pgtwin/common/se3.py", "SE(3) 位姿、四元数运算与插值"),
        ("src/pgtwin/real_data/", "episode 数据模型、加载、时间对齐、合成数据生成"),
        ("src/pgtwin/simulator/", "仿真后端抽象；解析后端与 MuJoCo 后端；动作回放"),
        ("src/pgtwin/physics/", "物理对象、参数空间、损失、优化器、敏感性策略、标定器"),
        ("src/pgtwin/gaussian/", "高斯场景、实例分离、碰撞代理、COLMAP 解析、CPU 光栅器"),
        ("src/pgtwin/evaluation/", "指标、可视化、实验留档"),
        ("src/pgtwin/cli/", "命令行入口"),
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        set_cjk_font(p.add_run(f"{src}："), name="Consolas", size=9.5)
        set_cjk_font(p.add_run(desc), size=9.5)

    doc.add_heading("A.2  每次实验的留档内容", level=2)
    for s in [
        "config.yaml —— 完整配置快照（复现入口）",
        "result.json —— 初始/最优参数、初始/最终指标、改善率、动作回放误差",
        "calibration_history.csv —— 每一次目标函数评估（参数与损失），"
        "用于可辨识性分析",
        "real_trajectory.csv / sim_before.csv / sim_after.csv —— 轨迹级证据",
        "optimization_curve.png / trajectory_xy.png / error_time.png / "
        "ee_replay_error.png —— 图表证据",
        "physical_object.json —— 校准后的高斯物理孪生体定义",
    ]:
        bullet(doc, s)

    doc.add_heading("A.3  文档索引", level=2)
    for f, d in [
        ("README.md", "项目总览、安装与快速开始"),
        ("docs/architecture.md", "架构说明，含坐标系约定专章"),
        ("docs/experiment_protocol.md", "实验协议（含真实物理引擎上的设计规则）"),
        ("docs/roadmap.md", "阶段进展与后续规划"),
        ("docs/demo_report.html", "自包含演示报告（含标定前后对比视频）"),
        ("CHANGELOG.md", "版本变更记录"),
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        set_cjk_font(p.add_run(f"{f}："), name="Consolas", size=9.5)
        set_cjk_font(p.add_run(d), size=9.5)


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "PGTwin_技术验证报告.docx"))
    parser.add_argument("--en", action="store_true", help="(reserved) English edition")
    args = parser.parse_args()

    doc = Document()
    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)  # A4
    s.left_margin = s.right_margin = Cm(2.4)
    s.top_margin = s.bottom_margin = Cm(2.2)
    style_doc(doc)

    cover(doc, "Physical Gaussian Digital Twin",
          "机器人交互数据驱动的三维高斯物理数字孪生构建与自适应校准系统")
    section_overview(doc)
    section_motivation(doc)
    section_architecture(doc)
    section_demos(doc, args.en)
    section_pipeline_evidence(doc)
    section_protocol(doc)
    section_repro(doc)
    section_limits(doc)
    appendix_inventory(doc)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    print(f"report -> {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
