# Example: push_box

The Phase-2 milestone experiment: recover a box's friction coefficient from a single push
episode.

## Run it

```bash
python scripts/run_calibration.py --example push_box
```

No data or simulator installation required. The script:

1. generates a synthetic "real" episode at ground-truth friction **0.8** (with 1 mm
   measurement noise) into `data/generated/push_box_001/`,
2. calibrates with `configs/experiments/push_box.yaml` (initial guess **0.2**, grid
   0.05–1.5, step 0.05, MockSimulator backend),
3. writes the full record to `outputs/push_box_001/`.

Expected output: optimized friction ≈ **0.80**, RMSE 4.3 cm → < 0.3 cm, improvement > 90%.

## Vary the experiment

```bash
# different ground truth / noise / off-center push
python scripts/generate_synthetic_episode.py --out data/generated/push_B \
    --friction 0.45 --noise 0.001 --lateral-offset 0.015
python scripts/run_calibration.py --episode data/generated/push_B \
    --config configs/experiments/push_box.yaml

# regenerate the default episode
python scripts/run_calibration.py --example push_box --regenerate
```

## Use your own real episode

Record data in the episode format (see README §6), then:

```bash
pgtwin calibrate --episode data/real/my_push_001 --config configs/experiments/push_box.yaml
```

Check `ee_replay_error.png` first: if the EE replay error is comparable to the object
error, fix the replay (frames, timing) before trusting the friction estimate.
