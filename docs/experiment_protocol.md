# Experiment Protocol

Standard operating procedure for calibration experiments — synthetic today, real robot in
Phase 7. The goal is that any run can be re-executed and audited from its `outputs/` record
alone.

## 1. Principles

1. **One episode directory = one interaction.** Never mix actions in one episode.
2. **Config over code.** An experiment is fully described by its YAML; if you edited code to
   change behavior, that is a new version, not a new experiment.
3. **Baseline first.** Every calibration records the uncalibrated rollout (`sim_before`) at
   the initial parameters; improvement claims are always relative to it.
4. **Audit replay before believing physics.** If `ee_replay_rmse` is not ≪ the object
   error, stop: the action did not reproduce, and the parameter estimate is invalid.
5. **Controlled randomness.** Simulator and samplers take explicit seeds from the config;
   two runs of the same config must produce identical results.

## 2. Synthetic protocol (Phase 2-3 — current)

Purpose: validate the inversion machinery with a known ground truth.

```bash
# 1. generate data at a known friction
python scripts/generate_synthetic_episode.py --out data/generated/push_A --friction 0.8 --noise 0.001
# 2. calibrate starting from a wrong guess (config initial: 0.2)
python scripts/run_calibration.py --episode data/generated/push_A --config configs/experiments/push_box.yaml
# 3. verify
pytest tests/test_calibration_pipeline.py -q
```

Acceptance criteria (enforced by tests):

| Check | Threshold |
| --- | --- |
| recovered friction (truth 0.8, grid 0.05) | within 0.75–0.85 |
| RMSE improvement over baseline | > 50% |
| calibrated RMSE (noise-free) | < 0.5 cm |
| EE replay RMSE | < 0.1 cm |
| repeat run | bit-identical parameters |

Recommended sweeps when touching the simulator or the loss: friction truth ∈
{0.3, 0.6, 1.0}, noise ∈ {0, 1 mm}, lateral offset ∈ {0, 2 cm}.

### 2.1 Push design rules for friction identifiability (Phase 3 findings)

Each rule below fixed a *measured* failure mode in the MuJoCo backend; they
apply equally to real experiments.

1. **Use a squat object** (width ≥ 2 × COM height, e.g. 8×8×4 cm). A pushed
   cube starts rocking/tipping near μ ≈ half-width / push-height (16° transient
   tilts measured at μ ≥ 0.75); rocking phase is chaotic in μ and buries the
   friction signal.
2. **Push through the center of mass height.** Off-center contact injects pitch
   torque and excites rocking.
3. **Retract the pusher after the push.** A pusher that merely stops leaves the
   object oscillating against it (contact compliance), making the release
   velocity — and therefore the friction-dependent slide — chaotic. Retraction
   gives a clean separation at a well-defined velocity.
4. **Push compliantly at moderate speed (~0.3–0.5 m/s effective release).**
   A rigid kinematic pusher is a hammer; real robots push with impedance
   control. In simulation this is the mocap-target + soft-weld pusher
   (`ee_weld_solref`); harder welds re-introduce chatter, softer ones lose
   tracking (measured sweet spot: 0.04 s with overdamped contacts).
5. **Expect asymmetric precision.** Free-slide length ∝ v²/(2μg): sensitivity
   falls as 1/μ², so high-μ estimates are inherently coarser. With 1 mm
   tracking noise and a 0.05 grid, single-episode recovery is ±0.05–0.10 with a
   small low bias; average multiple pushes for tighter bounds.
6. **Watch the EE replay error before trusting μ.** Cross-backend replay
   (action recorded under one actuation model, replayed under another) fails
   silently *except* for this metric — a 1.5 cm EE replay RMSE vs ~0 flags that
   the effective action differs and the estimate is not comparable.
7. **Collision-proxy geometry error trades off directly against friction.**
   Measured on the real classroom scene: a 6.5 mm error in the proxy's contact
   radius shifted the push-end object position by the same amount, and the
   optimizer compensated with a longer free slide — biasing recovered friction
   from 0.50 to 0.40. Rule of thumb: proxy contact-face error δ maps to a
   friction bias Δμ ≈ μ² · 2gδ / v² ; verify the proxy footprint against a
   measured dimension before calibrating.

## 3. Real-robot protocol (Phase 7 — planned)

### 3.1 Setup

- Table with a known-height flat surface; world frame marked physically (checkerboard or
  AprilTag board defining origin and axes, z-up).
- Object: rigid box/cylinder, weighed on a scale (ground-truth mass for validation),
  fiducials or mocap markers for 6-DoF tracking.
- Robot: Franka with a rigid pusher tip (known radius) instead of fingers.
- Cameras: calibrated (`T_camera_world` recorded into the episode `scene/`).

### 3.2 Per-episode checklist

1. Record object rest pose ≥ 0.5 s before contact (defines initial pose robustly).
2. Execute a straight push at constant speed (0.1–0.5 m/s) through the object center
   (Phase 1) — off-center pushes only after rotation loss is enabled.
3. Keep pushing distance ≥ 5 cm and let the object come fully to rest before stopping the
   recording (the free-slide segment carries the friction signal).
4. Verify time synchronization: robot and tracking timestamps on one clock, or estimate the
   constant offset with `time_alignment.estimate_time_offset` and store it in metadata.
5. Repeat ≥ 5 pushes per object; calibrate each independently; report median ± spread of
   recovered friction (consistency across episodes is the real-world validity check).

### 3.3 Data hygiene

- Store raw captures under `episode_*/camera/`, derived poses in the CSVs; never overwrite
  raw data.
- Note surface material, cleaning state, and object temperature in `metadata.extra` —
  friction drifts with all three.

## 4. Reporting

Each experiment directory is the report. For a paper figure set, use:

- `optimization_curve.png` — identifiability evidence (single clear minimum),
- `trajectory_xy.png` + `error_time.png` — before/after behavior match,
- `result.json` — quantitative claims (RMSE, improvement %),
- `calibration_history.csv` — full search trace for supplementary material.

Cross-experiment comparisons should be regenerated with `pgtwin evaluate` from the stored
CSVs rather than re-simulated, so figures stay consistent with the archived data.
