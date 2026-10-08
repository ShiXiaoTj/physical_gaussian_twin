# Architecture

## 1. What this system is

pgtwin is a **calibration framework**, not a renderer and not a physics engine. Its job is
to take a real robot interaction, replay it in a simulator, and invert the observed
real/sim discrepancy into physical parameters attached to a Gaussian object.

Everything that can be supplied by a mature external project — 3DGS training, rigid-body
dynamics, robot control — sits behind an adapter. The parts that are the contribution live
in `physics/` and `real_data/`.

## 2. Module layout and responsibilities

| Module | Responsibility | Must not |
| --- | --- | --- |
| `common/` | SE(3) `Pose`, frame registry, IO, logging | know about robots, sims, gaussians |
| `real_data/` | episode schema, loading, `PoseTrajectory`, time alignment, synthetic data | import a simulator |
| `robot/` | kinematic description per robot (EE contact radius, base pose, joints) | own control loops |
| `simulator/` | `BaseSimulator` + backends, scene building, EE replay | know about loss functions |
| `physics/` | parameters, losses, optimizers, sensitivity policy, calibrators | import a specific backend |
| `gaussian/` | scene IO, instance segmentation, collision proxies, reconstruction adapters | depend on `physics/` internals |
| `evaluation/` | metrics, plots, experiment records | mutate calibration state |
| `cli/` | argument parsing, config assembly, orchestration | contain algorithms |

Dependency direction is strictly downward: `cli → {physics, simulator, evaluation} →
{real_data, gaussian} → common`. `physics/calibrator.py` talks to a simulator only through
`BaseSimulator`, which is why swapping MuJoCo for Isaac changes one config key.

## 3. Coordinate Frame Convention

**This project fails on coordinate frames if it fails anywhere.** The rules below are
enforced by the type (`Pose`), the naming convention, and `tests/test_se3.py` +
`tests/test_metrics_and_transforms.py`.

### 3.1 Handedness and axes

- All frames are **right-handed**.
- The canonical `world` frame is **z-up**: `+x` forward (away from the robot base, along the
  usual push direction), `+y` left, `+z` up against gravity. Gravity is `[0, 0, -9.81]`.
- The support surface (table top) lies at a known `z = table_height`; object poses are
  expressed with the object's **geometric center**, not its contact face.

### 3.2 Quaternion order

Quaternions are **always `[x, y, z, w]`** (scalar last), unit norm, with the sign
canonicalized so `w ≥ 0`. This is the only representation crossing a module boundary or a
file format.

Two conversions exist because external tools disagree, and both are explicit:

- MuJoCo uses `[w, x, y, z]` → `simulator/mujoco_adapter.py` converts at the boundary only.
- 3DGS PLY rotations are stored `[w, x, y, z]` → `gaussian/ply_io.py` converts on load.

Never pass a scalar-first quaternion into `Pose`.

### 3.3 Transform naming

A transform is named `T_parent_child` and maps a point expressed in `child` into `parent`:

```
p_parent = T_parent_child · p_child
```

Composition reads left to right with matching inner names:

```
T_world_ee = T_world_robot · T_robot_ee
```

The frames this project uses, and who owns each:

| Transform | Meaning | Source |
| --- | --- | --- |
| `T_world_robot` | robot base in world | robot adapter / config |
| `T_robot_ee` | EE in robot base frame | robot FK (or measured) |
| `T_world_ee` | EE in world — **what episodes record** | `RobotState.ee_pose` |
| `T_world_object` | object center in world — **what episodes record** | `ObjectState.pose` |
| `T_camera_world` | extrinsics used during capture | capture rig / COLMAP |
| `T_gaussian_world` | 3DGS reconstruction frame → world | `GaussianScene.transform` |
| `T_sim_world` | simulator frame → world (identity by convention) | simulator adapter |

### 3.4 Rules

1. **Episodes store world-frame poses.** `robot_states.csv` holds `T_world_ee`,
   `object_states.csv` holds `T_world_object`. Any sensor-frame data is converted at load
   time, never later.
2. **`T_sim_world` is identity.** The simulator is built in world coordinates so sim and
   real trajectories are directly comparable with no hidden transform in the loss. A backend
   that cannot do this must expose its offset explicitly.
3. **Gaussian frames are aligned once.** 3DGS output typically has an arbitrary scale and
   orientation. `T_gaussian_world` is applied on load (`GaussianScene.transform`) so all
   downstream code sees world coordinates. Scale mismatch is a *metric* error and must be
   resolved here, not compensated in the physics.
4. **`FrameRegistry` resolves chains.** Register the edges you know
   (`reg.add("world", "robot", T_wr)`) and query any pair (`reg.get("world", "ee")`);
   resolution walks the graph and inverts edges as needed.
5. **Comparisons happen in world frame, on a common timeline.** Losses are computed only
   after `time_alignment` puts real and sim on identical timestamps.

### 3.5 Units

Position **metres** everywhere internally. Rotation **radians** internally. Reports convert
to **centimetres** and **degrees**, and the unit is in the field name
(`rmse_position_cm`, `rotation_error_deg`) so a mixed-unit comparison is visible in review.
Time is **seconds**; mass **kg**; friction dimensionless.

## 4. Data model

```
InteractionEpisode
├── EpisodeMetadata  (episode_id, action_type, robot, object_id, sampling_rate, ...)
├── list[RobotState]   (timestamp, ee_pose=T_world_ee, joints, gripper, ee_force)
└── list[ObjectState]  (timestamp, pose=T_world_object, linear/angular velocity)
```

`PoseTrajectory` is the vectorized view used by every numerical path: `(N,)` timestamps,
`(N,3)` positions, `(N,4)` quaternions, with SLERP interpolation, resampling, displacement,
and finite-difference velocities.

```
PhysicalObject
├── object_id, initial_pose
├── CollisionProxy   (box | convex_hull | sphere | mesh)   ← physics geometry
├── PhysicsParameters (mass, friction_static/dynamic, restitution, young_modulus, poisson)
├── gaussian_asset    (path/reference only)                 ← visual, loaded lazily
└── calibration_info  (method, episode, metrics)
```

The binding is by reference, which is the decoupling that matters: physics never needs
gaussians in memory, and rendering never needs the simulator.

## 5. The calibration loop

```
PhysicsCalibrator.calibrate(episode, object, parameter_space)
  1. align            real EE + object trajectories onto a common timeline (30 Hz)
  2. setup scene      support plane + object collision proxy + EE proxy
  3. baseline rollout replay real EE trajectory at the initial parameters
  4. search           for each candidate θ: set_physics_parameters → replay → loss
  5. select           θ* = argmin L(θ)
  6. verify           re-run at θ*, compute metrics + improvement + EE replay error
  7. record           CalibrationResult (params, losses, history, metrics)
```

Two invariants:

- **Same action.** Both real and sim consume the *same* recorded EE trajectory. The
  simulated EE trajectory is recorded back and its deviation from the commanded one is
  reported as `ee_replay_rmse`. If that number is not small, an object-trajectory mismatch
  is a replay problem, not a physics problem, and the calibration result is not meaningful.
  In the MuJoCo backend the pusher is *compliant* (mocap target + soft weld, mimicking
  impedance control), so `ee_replay_rmse` is a real tracking lag (~1.4 cm at 0.5 m/s) that
  is consistent between generation and calibration; a Phase 3 cross-backend probe showed
  that when the actuation models differ, recovery does not transfer and this metric is
  what exposes it.
- **Determinism.** The simulator is reset and re-seeded per candidate, so the only thing
  varying across the search is θ.

## 6. Action → Parameter Sensitivity

Not every parameter is observable from every action. Optimizing an insensitive parameter
fits noise; optimizing two coupled parameters from one action is unidentifiable.

| Action | Identifiable | Weakly sensitive | Not identifiable |
| --- | --- | --- | --- |
| `PUSH` | friction_dynamic | mass (coupled with friction), restitution | stiffness |
| `SLIDE` | friction_dynamic, friction_static | — | mass, restitution |
| `LIFT` | mass | friction_static (grasp slip) | restitution, stiffness |
| `COLLISION` | restitution | mass | friction |
| `PRESS` | young_modulus | poisson_ratio | friction, restitution |

The friction/mass coupling in a quasi-static push is the canonical trap: sliding
deceleration is `μg`, independent of mass, so mass barely moves the trajectory and a joint
search trades one against the other arbitrarily. **Phase 1 therefore calibrates friction
only, with mass fixed** — and `ParameterCalibrationPolicy` (`physics/sensitivity.py`)
enforces this, refusing parameters an action cannot identify.

`HierarchicalCalibrator` is the consequence: run LIFT to fix mass, then PUSH/SLIDE for
friction with mass held, then COLLISION for restitution, then PRESS for stiffness. Staged,
action-driven, sensitivity-ordered — never one joint black-box fit.

## 7. Backends and fallback

| Layer | Available now | Adapter interface | Planned |
| --- | --- | --- | --- |
| Simulator | `MockSimulator` (analytic planar push), **MuJoCo (validated)** | `BaseSimulator` | Isaac Sim |
| Robot | Franka, Piper (kinematic) | `BaseRobot` | real SDK control |
| Reconstruction | PLY / NPZ import, synthetic scenes | `GaussianReconstructionBackend` | gsplat, Splatfacto, PhysGaussian |
| Optimizer | grid, random | `Optimizer` | Bayesian (skopt), differentiable |

`MockSimulator` models a quasi-static planar push: Coulomb ground friction decelerating the
object at `μg`, an inelastic normal contact against the EE proxy, a Coulomb-bounded
tangential contact impulse, and yaw from off-center contact. It is deterministic and needs
no external engine, so **the full pipeline runs with nothing installed** — that is the
fallback guarantee from the requirements, and it is also what makes the synthetic recovery
test a trustworthy regression anchor.

## 8. Reproducibility

Every run writes `outputs/<experiment_id>/` containing the resolved config, `result.json`
(initial/optimized parameters, initial/final metrics, improvement), the full search history
as CSV, real and sim trajectories before and after calibration, all figures, and the
calibrated `physical_object.json`. Nothing needed to re-run or audit an experiment lives
only in the console.
