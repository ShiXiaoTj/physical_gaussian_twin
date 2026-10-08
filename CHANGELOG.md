# Changelog

All notable changes to Physical Gaussian Digital Twin are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.1] - 2026-10-08 — Word technical report generator

`scripts/build_report_docx.py` assembles a 31-page A4 Word report from the
experiment artifacts themselves (not hand-written): cover, project overview,
technical background with the key design points, architecture and frame
conventions, the eight verified experiments with live metrics tables and
figures, rebuild evidence, the protocol rules distilled from the debugging,
reproduction steps, limits and roadmap, and an inventory appendix.

- 14 tables / 29 embedded figures, all pulled from `outputs/*/result.json`
  and the experiment PNGs, so the numbers cannot drift from the results.
- CJK fonts set explicitly (ascii/hAnsi/eastAsia) so Chinese renders in Word
  on any platform; verified by LibreOffice → PDF render pass.
- `pytest tests/` → **183 passed**.

## [0.6.0] - 2026-10-08 — S2 for real: photos -> trained 3DGS -> calibrated twin

The last borrowed component is gone. The garden scene is trained from the
official 185-photo capture on this machine's RTX 4090D with gsplat 1.5.1:
15k iters, 1297x840, **655 s** -> 10.0M gaussians, held-out PSNR **25.06 dB**
(GT|render pairs in `data/gaussian/garden_trained/eval_*.png`).

### Added
- `gaussian/colmap_io.py`: dependency-free COLMAP binary reader
  (cameras/images/points3D), verified against the garden model — PINHOLE
  f≈3845, 185 poses, 138,766 sparse points.
- `scripts/train_gsplat.py`: gsplat training loop (DefaultStrategy densify/
  prune, L1+SSIM, SH degree warm-up), held-out PSNR eval, and export
  straight into the pgtwin `GaussianScene` (.npz + .splat).
- `SyntheticPushConfig.table_height`: tabletop episodes — object rest
  height, EE height and the simulator contact plane all follow it.
- `configs/experiments/push_vase_garden.yaml` + `--example push_vase_garden`.

### Verified
- Metric scale **double-anchored** on the scene's own furniture:
  table-top 1.32 u -> 0.74 m (=0.568) and table diameter 3.12 u -> 1.78 m
  (=0.570) agree to 0.4% — protocol rule 7 applied to a self-trained scene.
- Vase instance (10,322 gaussians, 0.76–1.40 m) with an explicitly measured
  body-cylinder proxy (r=0.09 m, h=0.26 m); dried flowers excluded from
  physics because they never contact the table.
- Push calibration on the table at 0.74 m: friction 0.2 -> **0.5500**
  (truth 0.55), object RMSE **2.34 -> 0.18 cm** (92.4%), EE replay ~0.
- EWA video `outputs/videos/push_vase_garden_001_splat.mp4`: before, the
  sim vase is blue-smeared up to 2.6 cm; after, it overlaps the ghost at
  0.04 cm.
- `pytest tests/` → **183 passed**.

### Fixed
- Deleted `garden_trained/scene.npz` by mistake while preparing scenes;
  recovered losslessly from the 321 MB `scene.splat` written at export time
  (splat carries float32 positions/scales/xyzw-quats — nothing was retrained).

## [0.5.4] - 2026-09-30 — True gaussian-splatting videos (CPU EWA rasterizer)

`gaussian/cpu_splatter.py`: a numpy EWA rasterizer with the same math as the
3DGS renderer — per-gaussian 3D covariance (scales+quats) projected through
the perspective Jacobian to 2D screen covariances, back-to-front alpha
compositing, 0.3px anti-alias floor. DC colors (no view-dependent SH). No
GPU/torch needed. Photoreal-level output: the playroom alphabet mat and book
spines, the classroom blackboard writing, and the truck's turquoise cab and
wooden stake bed all render as continuous surfaces.

- `render_video.py splat` now uses this engine: static background rasterized
  once per camera, only the moving instance (sim copy + blue ghost at the
  real pose, quats rotated per frame) re-splatted per frame.
- Rigid gaussian transforms rotate per-gaussian quaternions (not just
  positions), so anisotropic splats stay correctly oriented while moving.
- 3 videos re-rendered; the before/after story is now visible on the object
  itself (e.g. truck: blue-smeared at 11.5 cm error before, true colors at
  0.17 cm after).

### Verified
- `pytest tests/` → **183 passed**.

## [0.5.3] - 2026-09-30 — Gaussian point-splat videos: the object itself, moving

New `render_video.py splat` mode: a CPU perspective point-splat renderer
(pinhole projection, depth-sorted alpha scatter, ~340k background + instance
points per frame) that shows the segmented object's OWN gaussians moving
through the real scene — sim rollout in true colors, blue ghost at the real
pose, live error readout. No GPU/torch required.

- 3 videos: playroom chair (4.4→0.15 cm), truck (11.5→0.22 cm, full vehicle
  silhouette readable), classroom desk (3.6→0.24 cm).
- Demo report video section now leads with these object-level renders
  (chart-style `anim` videos remain available as a mode).

### Verified
- `pytest tests/` → **183 passed**.

## [0.5.2] - 2026-09-30 — Demo videos: before/after calibration, visible

`scripts/render_video.py` renders the calibration story as video (H.264):

- **mujoco3d**: 3D-rendered side-by-side rollouts at initial vs calibrated
  friction, with a translucent blue "ghost" kinematically following the REAL
  object trajectory — before: the sim box slides past the ghost; after: they
  overlap. Headless EGL rendering; CJK captions.
- **anim**: 2D top-down before/after animation over the real gaussian scene
  as background, live per-frame position-error readout (e.g. playroom chair:
  4.23 cm → 0.04 cm at the same timestamp).
- 4 videos generated (mujoco box 3D, playroom chair, truck, classroom desk),
  embedded into the demo report (self-contained, now 9.8 MB).

### Verified
- `pytest tests/` → **183 passed**.

## [0.5.1] - 2026-09-29 — Two more real 3DGS scenes: playroom & truck

Pipeline generality demonstrated on the official 3DGS benchmark scenes
(downloaded from ModelScope `Voxel51/gaussian_splatting`, Apache-2.0 card):

- **playroom** (Deep Blending, 1.92M gaussians): metric scale 0.58
  double-anchored (ceiling 4.15 u → 2.4 m AND desk plane 1.28 u → 0.74 m);
  segmented the red kids' chair (0.84×0.79 m, 0.55 m tall); friction
  0.2 → **0.6000** (truth 0.6), RMSE 3.84 → 0.18 cm (95.4%).
- **truck** (Tanks&Temples, 2.06M gaussians): reconstruction ≈ metric
  (truck length 6.7 u ≈ 6.7 m); segmented the whole vehicle (165k solid
  gaussians, 6.7×3.7×2.6 m proxy); locked-wheel SLIDE calibration
  0.2 → **0.7000** (truth 0.7), RMSE 10.44 → 0.35 cm (96.7%) — scale
  generality from a 6 cm box to a 6.7 m vehicle.

### Added
- `fit_ground_plane(rank=N)`: skip the N most dominant planes (indoor walls
  can out-vote the floor); `--plane-rank` in prepare_gaussian_scene.py; test.
- `SyntheticPushConfig.action_type` ("push" | "slide").
- Experiment configs + one-command examples `push_chair_playroom`,
  `slide_truck`; highlight-splat exports for both scenes; demo report now
  covers 7 demos with reference photos of the real scenes.

### Verified
- `pytest tests/` → **183 passed**.

## [0.5.0] - 2026-09-29 — Phase 8: mass calibration & hierarchical staging; Phase 7 deferred

Staged calibration is now executable
end-to-end: LIFT identifies mass through the grip-force signal, PUSH then
identifies friction through the trajectory signal, with stage results
threaded and provenance recorded. Phase 7 (real-robot experiments) is
explicitly deferred for lack of hardware data; a readiness validator ships
so real episodes can be checked the day they exist.

### Added — the observable dimension of sensitivity
- `sensitivity.ACTION_OBSERVABLE` + `observable_for_action()`: each action
  names the *measurement signal* that carries its parameters. Physics fact
  encoded: under a kinematic grasp a LIFT trajectory is imposed by the
  gripper and contains **no mass information** — the observable is the grip
  force `F = m (a + g_up)`; PUSH/SLIDE observables are object trajectories.
- `PhysicsCalibrator(signal="auto")` dispatches the loss per episode:
  trajectory loss for pose observables, `ForceLoss` (mean |ΔF|, newtons) for
  force observables; force-mode results report `force_mae_n`.

### Added — LIFT support
- `BaseSimulator.attach_object() / detach_object()` (kinematic grasp);
  MockSimulator implements grasping, EE accel tracking, and the grip-force
  model `F = m (a + g_up)` (None when nothing is held).
- `ReplayResult.ee_forces`: grip-force series recorded during replay.
- `SyntheticLiftGenerator` / `SyntheticLiftConfig`: cosine-ramp lift + hold,
  force + position noise, 100 Hz force sampling; episodes round-trip through
  the standard CSV format (fx,fy,fz columns).
- `physics/estimators.py::estimate_mass_from_lift`: closed-form hold-window
  estimate `m = mean(F_z)/g` with automatic steady-window selection — the
  direct cross-check for the search-based result.

### Added — hierarchical execution
- `HierarchicalCalibrator` stages record `stage_input_physics` provenance;
  `scripts/run_hierarchical.py --example lift_push` runs the two-stage demo
  and writes per-stage experiment records + a combined
  `hierarchical_summary.json` + the final calibrated twin.
- `configs/experiments/lift_push_hierarchical.yaml`.

### Added — Phase 7 readiness (deferred phase)
- `scripts/validate_episode.py`: pre-calibration checks for real episodes —
  schema/NaN/monotonicity (via the loader), unit plausibility, quaternion
  norm and xyzw-order heuristics, sampling gaps, action-specific signal
  checks (push displacement, EE approach distance, force presence for
  LIFT/PRESS, lift height). Exit 0/1 for scripting.

### Verified
- `pytest tests/` → **180 passed** (18 new: grasp mechanics, force loss
  units, direct estimator incl. CSV round-trip, mass recovery, signal
  dispatch, trajectory-blindness of mass under grasp, two-stage recovery,
  stage threading, stage ordering).
- Hierarchical demo: mass 1.0 → **0.4000 kg** (truth 0.42, grid step 0.05;
  direct estimate 0.4198 kg), friction 0.2 → **0.8000** (truth 0.8); force
  MAE 5.68 → 0.20 N.
- Negative control: with `signal="object_pose"` forced on a LIFT episode,
  the mass loss landscape is flat (spread < 5%) — confirming the observable
  table is not just documentation but a measured property of the system.

### Decisions
- **Phase 7 deferred, not skipped**: it is a data-execution phase, not a
  code phase. Everything it needs ships and is tested against synthetic
  stand-ins; `docs/experiment_protocol.md` §3 + `validate_episode.py` are
  the entry path once real Franka/Piper data exists.

## [0.4.0] - 2026-09-28 — Phase 5+6: real 3DGS scene ingestion & gaussian-derived collision proxies

The full S2→S7 chain now runs on a **real, internet-downloaded 3DGS scene**:
a high-school classroom (2.3M gaussians) is aligned to the metric world
frame, the teacher's desk is segmented as an instance, its collision proxy
is derived from the real gaussians, and push calibration recovers the
episode's ground-truth friction exactly.

### Added
- `gaussian/alignment.py`: RANSAC ground-plane fitting (`fit_ground_plane`),
  z-up alignment with plane at z=0 (`align_scene_to_ground`), manual metric
  scale anchoring (`estimate_metric_scale`).
- `GaussianScene.metric_scale` + alignment persisted through NPZ round-trips;
  `positions_world()` now returns metric, z-up coordinates.
- `gaussian/splat_io.py`: dependency-free `.splat` loader (antimatter15 /
  SuperSplat format) with quaternion renormalization; `GaussianScene.load`
  dispatches on `.ply` / `.splat` / `.npz`.
- `scripts/prepare_gaussian_scene.py`: load → align → scale → crop →
  floater-filter → NPZ + top-down/side preview plots.
- Config support for explicit instance selection
  (`gaussian.instance_bbox: {min, max}`) and gaussian-instance provenance in
  the calibrated `physical_object.json`.
- `SyntheticEpisodeGenerator.generate(obj=...)`: push an externally-defined
  object (e.g. a gaussian-proxy object) with configured ground-truth physics.
- `configs/experiments/push_desk_real_scene.yaml` + `--example
  push_desk_real_scene`: pushes the segmented desk instance at its true scene
  location with a gaussian-derived proxy.
- `tests/test_scene_alignment.py` (10) and
  `tests/test_real_scene_integration.py` (7, auto-skip without the scene).

### Changed
- Collision proxies are now expressed in the **object body frame** (origin at
  the instance centroid) so one proxy is valid at any pose; world placement
  comes from `PhysicalObject.initial_pose`.

### Verified
- Raw 574 MB 3DGS PLY (2,316,374 gaussians, 62 properties) parses in ~1 s
  with the built-in reader; `.splat` variant parses and matches.
- Desk-plane RANSAC alignment: the dominant plane in the classroom is the
  desk surface — serendipitously the right z=0 for tabletop manipulation;
  metric scale 0.78 anchored on desk/room heights.
- `--example push_desk_real_scene`: instance 6,545 gaussians → bbox proxy
  0.57×0.44×0.11 m → friction 0.2 → **0.5000** (truth 0.5), RMSE
  **3.45 → 0.18 cm** (94.9%).
- **Geometry-friction coupling measured**: with a 6.5 mm proxy-radius
  mismatch between episode and calibration geometry, recovered friction
  biased 0.50 → 0.40 — quantitatively explained by the extra slide needed to
  compensate the contact-offset difference. Proxy accuracy is a first-order
  requirement for push calibration (protocol rule #7).
- `pytest tests/` → **162 passed**.

### Data
- Scene: ModelScope `Lengyd/high-school-classroom-3dgs` (CC-BY-NC-4.0),
  research/non-commercial use; assets stay out of git (see the experiment
  config header for download + prepare commands).

## [0.3.0] - 2026-09-28 — Phase 3: MuJoCo backend validated

First full-physics backend: the same grid calibration that passed on
MockSimulator now recovers friction in MuJoCo 3.14, and the experiment
design that makes friction *identifiable* in a real contact engine is
captured in code and docs.

### Added
- `configs/experiments/push_box_mujoco.yaml` and the `push_box_mujoco`
  one-command example (`python scripts/run_calibration.py --example push_box_mujoco`).
- Compliant pusher in the MJCF push scene: mocap target + soft weld
  (`ee_weld_solref`, default 0.04) with the tip's *actual* pose reported as the
  robot state — the EE replay error is now a physically meaningful tracking lag.
- `SimulatorConfig.contact_solref / contact_dampratio / ee_weld_solref`, threaded
  through the scene builder and exposed in YAML.
- `SyntheticPushConfig.backend` (mock | mujoco) and `sim_options` passthrough:
  ground-truth episodes can be rolled out in any backend; `object_size`,
  `push_height_ratio` and `retract_distance` are now configurable.
- `tests/test_mujoco_adapter.py` (12 tests, skipped when mujoco is absent):
  contract, compliant-pusher behavior, friction identifiability, noise-free and
  noisy synthetic recovery, cross-backend actuation guardrail.
- `tests/test_imports.py`: fresh-interpreter import test for every entry package.

### Changed — experiment design (the Phase 3 findings)
Push-based friction identification in a full contact engine required four
design changes, each fixing a measured failure mode (docs/experiment_protocol.md §2.1):
- **Object geom `priority="1"`** — MuJoCo combines pair friction by element-wise
  max; the table's friction (1.0) masked the calibrated value entirely.
- **EE geom `priority="2"` with fixed friction** — the pusher-object friction is
  a different physical parameter than object-table friction; without this the
  calibrated μ leaked into the pusher contact.
- **Squat object (8×8×4 cm), center-height push, EE retraction** — tall objects
  rock/tip at high μ (16° transient tilts measured), an instantly-stopping pusher
  leaves the object oscillating on the contact spring, and both bury the friction
  signal in release-state chaos.
- **Compliant pusher + overdamped contact (`dampratio=2`) + elliptic cone,
  `impratio=10`, dt=0.001** — a rigid kinematic pusher at 0.5 m/s hammers the
  object; contact-spring surging made rollouts chaotic in both friction and the
  (noisy) initial pose. The soft weld reproduces impedance-controlled pushing.
- `SyntheticEpisodeGenerator` defaults updated accordingly (squat box,
  `push_height_ratio=1.0`, `retract_distance=0.08`).
- MuJoCo mass updates now scale only the object body (was `mj_setTotalmass`,
  which rescaled the whole model).

### Fixed
- Circular import: `pgtwin.physics.__init__` eagerly imported `calibrator`
  (which imports the simulator layer), so importing any `simulator` module
  first raised ImportError. Calibrator exports are now lazy (PEP 562).

### Verified
- `pytest tests/` → **147 passed** (135 core + 12 MuJoCo).
- Same-backend gate: noise-free recovery **exact (0.8000)**; with 1 mm noise,
  seeds 0/1/2 recover **0.75 / 0.75 / 0.70** (truth 0.8), all within the 0.7–0.9
  acceptance band.
- `--example push_box_mujoco`: friction 0.2 → **0.75**, object RMSE
  **2.71 → 0.18 cm** (93.3%), EE replay RMSE 1.41 cm (compliant-tip lag,
  reported honestly).
- Cross-backend probe (mock-recorded action replayed in MuJoCo): recovery does
  **not** transfer (0.20 / 1.50 mirror bias) because the effective action
  differs — and the EE replay error metric flags exactly this (1.5 cm vs
  ~1e-17 m same-backend). Closing the actuation gap is Phase 4 scope.

### Known limitations
- Recovered friction carries a small low bias (−0.05…−0.10) under noise: the
  loss basin is asymmetric (flat high-μ side, since slide length ∝ 1/μ) —
  within typical real-world friction repeatability; multi-episode joint
  calibration is the planned tightener (Phase 7/8).
- Isaac Sim adapter remains a stub; restitution→solref mapping is approximate.

## [0.2.0] - 2026-09-28 — Phase 2 Milestone: synthetic friction recovery

The first end-to-end closed loop runs: a synthetic "real" push episode with known friction
is replayed in simulation, and grid search recovers the ground-truth coefficient.

### Added — simulation (Phase 2)
- `simulator/base_simulator.py`: `BaseSimulator` interface (`reset`, `load_scene`,
  `load_robot`, `load_object`, `set_physics_parameters`, `set_object_pose`,
  `set_robot_state`, `step`, `get_object_state`, `get_robot_state`,
  `apply_ee_trajectory`, `close`), plus `SimulatorConfig` and `RolloutResult`.
- `simulator/mock_adapter.py`: `MockSimulator`, an analytic quasi-static planar push model
  with Coulomb ground friction, a bounded tangential contact impulse, and yaw from
  off-center contact. Deterministic; runs with no external engine.
- `simulator/mujoco_adapter.py`: `MuJoCoSimulator` generating MJCF via
  `scene_builder.py`. Import-guarded — raises with install guidance when `mujoco` is
  absent; an Isaac backend is a documented factory stub.
- `simulator/scene_builder.py`: backend-neutral MJCF scene generation (support plane,
  free object from its `CollisionProxy`, mocap-driven EE) — unit-testable without MuJoCo.
- `simulator/replay.py`: `PushReplayController` for kinematic EE-pose replay, recording the
  *actual* simulated EE trajectory so action-replay error is measured separately from
  object error.
- `simulator/__init__.py`: `create_simulator(cfg)` factory with graceful backend errors.

### Added — physics and calibration (Phase 2)
- `physics/physical_object.py`: `PhysicsParameters`, `CollisionProxy`, `PhysicalObject`;
  JSON round-trip for the `G_phy` record.
- `physics/parameter_space.py`: `ParameterSpace` / `ParameterSpec` with grid enumeration,
  random sampling, clipping, and bounds from YAML.
- `physics/losses.py`: `PositionLoss`, `RotationLoss`, `VelocityLoss`, `ContactLoss`,
  `TrajectoryLoss` (weighted sum). Units documented: position in metres, rotation in radians
  internally and degrees in reports.
- `physics/optimizer.py`: `GridSearchOptimizer`, `RandomSearchOptimizer`, a
  `BayesianOptimizer` (activates when scikit-optimize is installed), and a
  `DifferentiableOptimizer` interface stub.
- `physics/calibrator.py`: `PhysicsCalibrator` (the S6 inversion step) and
  `HierarchicalCalibrator` for staged, action-driven identification.
- `physics/sensitivity.py`: the action → parameter sensitivity table and
  `ParameterCalibrationPolicy`, which decides *which* parameters an episode may identify.

### Added — evaluation and reporting
- `evaluation/metrics.py`: ATE (cm), rotation error (deg), final displacement error,
  trajectory RMSE, and `improvement_percent`.
- `evaluation/visualization.py`: XY and XYZ trajectory plots, loss-vs-parameter curve,
  optimization history, before/after comparison, per-time position error, EE replay error.
- `evaluation/report.py`: writes the full `outputs/<experiment_id>/` bundle — `config.yaml`,
  `result.json`, `calibration_history.csv`, real/sim trajectory CSVs, and all figures.

### Added — synthetic data
- `real_data/synthetic.py`: `SyntheticEpisodeGenerator` producing a ground-truth push
  episode at a chosen friction, with optional measurement noise, so calibration can be
  validated without hardware.

### Added — core data model (Phase 1)
- `common/se3.py`: `Pose` on SE(3) with quaternion order fixed to `[x, y, z, w]`;
  `compose`, `inverse`, `relative_to`, `interpolate` (SLERP), `rotation_distance`,
  4x4 homogeneous conversion.
- `common/transforms.py`: `FrameRegistry` transform graph over the canonical frames
  (`world`, `robot`, `object`, `ee`, `camera`, `gaussian`, `sim`) with BFS resolution.
- `common/io.py`, `common/logging.py`: JSON/YAML/CSV helpers and configured logging.
- `real_data/episode.py`: `ActionType`, `RobotState`, `ObjectState`, `EpisodeMetadata`,
  `InteractionEpisode` with structural validation.
- `real_data/pose_utils.py`: `PoseTrajectory` with vectorized interpolation, resampling,
  displacement, and velocity estimation.
- `real_data/trajectory_loader.py`: `EpisodeLoader` — schema checks, timestamp sorting,
  resampling, coordinate normalization.
- `real_data/time_alignment.py`: linear position interpolation + quaternion SLERP onto a
  common timeline, constant clock-offset estimation; DTW reserved as a documented stub.

### Added — Gaussian representation (Phase 5 groundwork)
- `gaussian/gaussian_object.py`: `GaussianObject` binding gaussian indices, instance id,
  transform, visual properties, and physical properties.
- `gaussian/scene_loader.py`: `GaussianScene` (PLY load/save, transform, object extraction,
  collision-proxy export) behind a backend-neutral interface.
- `gaussian/instance_manager.py`: `ManualInstanceSelector` (bbox / id / mask) — segmentation
  kept decoupled from calibration.
- `gaussian/reconstruction.py`: `GaussianReconstructionBackend` adapter interface with
  `PlyImportBackend` (works today) and gsplat / Splatfacto stubs (Phase 5).
- `gaussian/ply_io.py`: minimal dependency-free PLY reader (ASCII + binary_little_endian,
  3DGS attribute conventions: log-scales, wxyz rotations, SH DC colors).

### Added — robot adapters, CLI, configs, tests
- `robot/base_robot.py`, `robot/franka_adapter.py`, `robot/piper_adapter.py`: kinematic
  descriptions (EE contact radius, base pose, joint metadata) behind one interface, with a
  `create_robot` factory.
- `cli/calibrate.py`, `cli/evaluate.py`, `cli/reconstruct.py`, `cli/main.py`: the `pgtwin`
  entry point.
- `scripts/run_calibration.py` (`--example push_box` runs with no data),
  `scripts/replay_episode.py`, `scripts/evaluate_result.py`,
  `scripts/generate_synthetic_episode.py`.
- `configs/`: `default.yaml`, `robot/{franka,piper}.yaml`, `scene/tabletop.yaml`,
  `calibration/friction_grid.yaml`, `experiments/push_box.yaml`.
- `tests/`: 126 tests across SE(3), frame registry, episodes, alignment, losses, parameter space,
  optimizers, metrics, Gaussian scene/instances, mock simulator, and the full pipeline.

### Verified
- `pytest tests/ -q` → **126 passed**.
- `python scripts/run_calibration.py --example push_box` → friction 0.2 → **0.8000**
  (ground truth 0.8), object RMSE **4.30 cm → 0.18 cm**, improvement **95.9%**, EE replay
  RMSE ~0 m, 13 artifacts written.

### Fixed
- `evaluation/report.py` imported `CalibrationResult` at module scope, creating a cycle with
  `physics/calibrator.py`; moved under `TYPE_CHECKING`.
- `MockSimulator` applied only a normal contact impulse, so an off-center push produced no
  yaw. Added a Coulomb-bounded tangential contact impulse and its torque about the center.
- `PhysicsParameters.from_dict` passed `young_modulus` through unconverted, so YAML 1.1
  parsing of `1.0e6` leaked a string into `physical_object.json`. Values are now coerced to
  float and the config uses `1.0e+6`.

### Known limitations
- MuJoCo and Isaac adapters are unexecuted here (packages unavailable offline).
- Replay is kinematic; controller-based replay with IK is Phase 4.
- Only `friction` is calibrated; `mass` is deliberately fixed (identifiability).

## [0.1.0] - 2026-09-28 — Phase 0/1: skeleton and core data model

### Added
- Project skeleton, `pyproject.toml`, `requirements.txt`, package layout under `src/pgtwin/`.
- `CURRENT_STATE.md` recording the Phase 0 repository and environment survey: the directory
  was empty, and PyPI, MuJoCo, Isaac Sim, CUDA, and robot SDKs are all unavailable — which
  is why mock-first, adapter-everywhere is the architecture.
- `docs/architecture.md` (including the Coordinate Frame Convention),
  `docs/experiment_protocol.md`, `docs/roadmap.md`, `README.md`.
