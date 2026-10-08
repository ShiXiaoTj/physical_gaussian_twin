# CURRENT_STATE

Phase 0 report — repository and environment survey, recorded before any code was written.

## Repository state at start

`/home/linlan/physical_gaussian_twin` was an **empty directory**: no git repository, no
files, no prior sources. Nothing was overwritten or deleted; the project skeleton was
created from scratch.

```
/home/linlan/physical_gaussian_twin/    (empty, not a git repo)
```

Because there was no existing code, the "read before you break" branch of Phase 0 did not
apply. The tree now in the repository is described in `docs/architecture.md`.

## Environment survey

Probed with `python3 -c "import ..."` and `pip download`:

| Dependency | Status | Consequence for the design |
| --- | --- | --- |
| Python 3.13.9 (anaconda) | available | target is 3.10+, satisfied |
| numpy, scipy, pandas, matplotlib, pyyaml | available | used directly |
| pytest | available | test suite runs |
| PyPI network access | pypi.org blocked; **Tsinghua/Aliyun mirrors reachable** | install extras with `pip install <pkg> -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| mujoco | **3.14.0 installed** (Tsinghua mirror, Phase 3) | `MuJoCoSimulator` validated: synthetic friction recovery passes |
| Isaac Sim / omni.isaac | not installed | `IsaacAdapter` left as a documented stub |
| CUDA / torch | not present | no learned or differentiable component in Phase 1-2 |
| 3DGS backends (gsplat, nerfstudio, diff-gaussian-rasterization) | not installed | `GaussianScene` loads PLY directly; backends behind `GaussianBackend` |
| Franka / Piper SDK, real robot | not connected | robot adapters are kinematic descriptions only |

## What this implies (and what was built because of it)

The environment cannot run a real physics engine or a real 3DGS pipeline, so the
architecture had to make those *replaceable back ends* rather than foundations. That
constraint is the reason for three decisions:

1. **`MockSimulator` is a first-class citizen, not a test double.** It implements an
   analytic quasi-static push model (Coulomb friction, tangential contact impulse,
   rotation from off-center contact) through the same `BaseSimulator` interface as
   MuJoCo. The whole calibration pipeline is validated against it, so swapping in
   MuJoCo later changes one config line (`simulation.backend`) and nothing else.
2. **`SyntheticEpisodeGenerator` replaces the missing robot.** It produces a ground-truth
   episode at a known friction, which makes the recovery test (0.2 → 0.8) the project's
   correctness anchor instead of unavailable hardware data.
3. **PLY loading with a synthetic fallback replaces the missing 3DGS stack.** No
   reconstruction code is vendored; `GaussianScene` consumes whatever a backend produced.

Every default path in the repository therefore runs offline, on CPU, with no robot and no
simulator installed: `python scripts/run_calibration.py --example push_box` works on a
clean checkout.

## Verified status

- `pytest tests/ -q` — 180 passed (incl. MuJoCo + real-scene suites).
- `python scripts/run_calibration.py --example push_box` — recovers friction 0.8000 from an
  initial guess of 0.2, object RMSE 4.30 cm → 0.18 cm (95.9% improvement), EE replay error
  ~0 (kinematic replay), 13 artifacts written to `outputs/push_box_001/`.

## Known limitations at this point

- Physics fidelity is bounded by `MockSimulator`'s quasi-static planar model: it covers
  push/slide on a flat support, not toppling, multi-body contact, or deformation.
- MuJoCo backend validated (Phase 3). Isaac adapter remains a factory stub with clear
  install guidance; no Isaac code paths run.
- Robot adapters do not do IK; Phase 1-2 replay is kinematic (EE pose is commanded
  directly). Controller-based replay is Phase 4.
- Collision proxies are box / convex-hull / sphere fits from Gaussian centers; a
  Gaussian → mesh → collision-mesh route is Phase 6.
- Only `friction` is calibrated. `mass` is held fixed on purpose to avoid the
  friction-mass identifiability problem described in `docs/architecture.md`.
