# Roadmap

Status legend: ✅ done · 🔶 in progress · ⬜ planned

## Phase 0 — Repository & environment analysis ✅

Empty directory confirmed; environment surveyed (no MuJoCo / Isaac / CUDA / robot / 3DGS
stack; PyPI unreachable). Consequence: mock-first, adapter-everywhere architecture.
Output: `CURRENT_STATE.md`.

## Phase 1 — Core data model ✅

SE(3) `Pose` (quaternion xyzw, compose/inverse/SLERP), `FrameRegistry`,
`InteractionEpisode` + CSV/JSON episode format, `EpisodeLoader` (validation, sorting,
resampling), `PoseTrajectory`, time alignment (common timeline, offset estimation),
losses with explicit units. 60+ unit tests.

## Phase 2 — Synthetic closed loop ✅  ← **Milestone 1 passed**

`MockSimulator` (deterministic quasi-static planar push), `PushReplayController` with EE
replay auditing, `ParameterSpace` + grid/random search, `PhysicsCalibrator`,
`SyntheticEpisodeGenerator`, metrics/plots/experiment records, CLI + one-command demo.

**Milestone result**: truth μ=0.8, initial 0.2 → recovered **0.8000**; RMSE 4.30 → 0.18 cm
(95.9%); EE replay RMSE ≈ 0; deterministic; robust to 1 mm noise and across
μ ∈ {0.3, 0.6, 1.0}. 126 tests green.

## Phase 3 — Real simulator backend ✅  ← **MuJoCo validated (2026-09-28)**

MuJoCo 3.14 installed (Tsinghua PyPI mirror) and the adapter executed for the
first time. Grid calibration recovers friction in full physics: noise-free
recovery is exact (0.8000); with 1 mm measurement noise, seeds recover
0.70–0.75 for truth 0.8. `--example push_box_mujoco` runs end-to-end
(RMSE 2.71 → 0.18 cm, 93.3%). 12 backend tests, skipped gracefully without mujoco.

Findings that now live in code + protocol (each fixed a measured failure):

1. **Contact-parameter masking**: MuJoCo pair friction = element-wise max →
   object geom needs `priority="1"` or the table friction hides the calibrated μ.
2. **Parameter purity**: EE geom gets `priority="2"` + fixed friction so the
   calibrated μ affects only the object-table contact.
3. **Rocking/tipping**: tall objects tip at high μ (μ_tip ≈ half-width /
   push-height); squat object + center-height push keep the grid in the
   sliding regime.
4. **Release-state chaos**: an instantly-stopping rigid pusher leaves the object
   oscillating on the contact spring → EE retraction + compliant pusher
   (mocap + soft weld) + overdamped contact restore identifiability.
5. **Actuation gap is measurable**: cross-backend replay (mock-recorded action
   in MuJoCo) fails to transfer recovery and is flagged by the EE replay error
   (1.5 cm vs ~1e-17 same-backend) — the audit metric doing its job.

Remaining in-phase extras (deferred): Isaac Sim adapter (stub), tighter
restitution mapping.

## Phase 4 — Real push episode replay ⬜

Replay recorded Franka pushes: gripper-tip geometry from `configs/robot/franka.yaml`,
constant clock-offset estimation, EE replay error budget (< 20% of object error).
**Actuation matching is the known hard part** (Phase 3 cross-backend finding): the
compliant sim pusher must reproduce the real controller's tracking behavior —
options are a stiffer weld for slow real pushes (lag ≈ v·τ, mm-scale at 5–10 cm/s),
target-lead compensation, or controller-mode replay behind
`PushReplayController(mode="controller")`.

## Phase 5 — Real Gaussian scene ingestion ✅  ← **validated on a real scene (2026-09-28)**

Downloaded a real 3DGS classroom (ModelScope, CC-BY-NC-4.0; 2.3M gaussians, 574 MB
PLY + 74 MB .splat — huggingface/github unreachable, ModelScope mirror used). The
built-in PLY reader parses the full 62-property export in ~1 s; a `.splat` loader was
added. `gaussian/alignment.py` RANSAC-fits the dominant plane (here: the desk plane —
ideal for tabletop work), aligns it to z=0/z-up, and applies a manually anchored metric
scale (0.78 from desk/room heights; real experiments use a measured reference).
`scripts/prepare_gaussian_scene.py` packages align+crop+filter into one command.
Teacher's-desk instance segmented by world-frame bbox (6,545 gaussians).

## Phase 6 — Collision proxies from real gaussians ✅ (bbox/hull/sphere) / 🔶 (mesh)

Body-frame proxies derived from the real instance (bbox 0.57×0.44×0.11 m; convex hull
102 verts, 0.139 m³). End-to-end gate passed: pushing the gaussian-proxy object at its
true scene location recovers friction exactly (0.2 → 0.5000, truth 0.5, RMSE
3.45 → 0.18 cm). **Measured lesson**: a 6.5 mm proxy-size error biased friction by
−0.10 (geometry-friction coupling) — proxy fidelity is first-order for push calibration.
Remaining: watertight-mesh route (gaussian → mesh → MJCF mesh asset; currently hull →
bounding-cylinder approximation in MJCF).

## Phase 7 — Real-world calibration experiments ⏸ deferred (no robot data)

A data-execution phase, not a code phase: every ingredient ships and is tested against
synthetic stand-ins. When real Franka/Piper episodes exist: run
`scripts/validate_episode.py <dir>` (unit/quaternion/sampling/signal checks), then the
standard pipeline per `docs/experiment_protocol.md` §3 (≥ 5 pushes × ≥ 2 objects × ≥ 2
surfaces; report median ± IQR of recovered friction, before/after RMSE, cross-episode
generalization). The known hard part is actuation matching (Phase 3/4 finding) — budget
the EE replay error before trusting any parameter.

## Phase 8 — Mass & hierarchical calibration ✅  ← **validated (2026-09-29)**

The staged-identification innovation is executable and measured:

- **Observable dimension added to the sensitivity policy**: LIFT's mass information lives
  in the grip force (`F = m(a + g)`), not the trajectory (which a kinematic grasp imposes).
  Verified by a negative control — forcing the trajectory loss on a lift episode yields a
  flat (< 5% spread) landscape over mass.
- MockSimulator grasping + grip-force model; `ForceLoss` (newtons);
  `estimate_mass_from_lift` closed-form cross-check (0.4198 kg vs truth 0.42).
- `HierarchicalCalibrator.calibrate_auto`: LIFT→mass then PUSH→friction with threaded
  results and provenance; order enforced regardless of episode list order.
- Demo `scripts/run_hierarchical.py --example lift_push`: mass 1.0 → 0.4000 kg,
  friction 0.2 → 0.8000 (truths 0.42 / 0.8); 180 tests green.

Remaining for later: COLLISION→restitution and PRESS→stiffness stages need episode
generators + (for press) a deformable backend; multi-episode joint refinement;
loss-curvature uncertainty.

## Beyond (not scheduled)

Bayesian optimization (skopt), differentiable backends (Warp/Brax) behind
`DifferentiableOptimizer`, DTW alignment, SAM/semantic instance segmentation, deformable
objects (PRESS + young_modulus/poisson), multi-object scenes, online adaptation during
deployment.
