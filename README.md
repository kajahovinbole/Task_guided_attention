# Task-Guided Attention

Research code for studying how human cues, especially pointing direction, can guide visual attention in AbSViT. The repository contains data preparation scripts, MediaPipe landmark extraction, pointing-prior generation, model experiments, and visual evaluation tools.

## Project layout

- `models/AbSViT/`: Vision Transformer and AbSViT model implementations.
- `models/mediapipe/`: MediaPipe task model files.
- `scripts/data_processing/`: dataset export, merging, filtering, and file organization.
- `scripts/run_mediapipe/`: hand and pose landmark extraction and combination.
- `scripts/prior/`: pointing-hand scoring and numeric pointing-prior generation.
- `scripts/visualisering/`: visual checks for labels, pointing priors, and predicted directions.
- `scripts/test_absvit_baseline.py`: checkpoint loading and a minimal forward pass.
- `scripts/test_absvit_prior.py`: AbSViT experiment using a spatial pointing prior.
- `scripts/evaluate_bbox.py`: compares model focus maps with ground-truth boxes.
- `data/`, `results/`, and `checkpoints/`: local inputs and generated artifacts. These folders are ignored by Git and need to be present locally.
- `main.py` and `src/`: reserved for a future unified entry point and reusable pipeline code.

## Setup

The project uses `uv`. Python 3.12 is selected by `.python-version`.

```bash
uv sync
```

Model checkpoints are not stored in Git. Place the required checkpoint files in `checkpoints/`. The original AbSViT Tiny checkpoint can be downloaded with:

```bash
mkdir -p checkpoints
curl -L "https://berkeley.box.com/shared/static/4z5y181r02b8034r2b8109u8x2c15t14.pth" -o checkpoints/absvit_tiny.pth
```

The prior experiment uses a separately trained/configured patch-8 GAP checkpoint named `checkpoints/absvit_tiny_patch8_224_gap.pth`. It is not included in the repository; the baseline checkpoint above is not a substitute for it.

## Current workflows

Run commands from the repository root. Local datasets and checkpoints must exist at the paths described below.

### Experiment with reach-adaptive cone length

Before changing the full pipeline, compare a fixed cone with a cone whose length is scaled from 2D elbow and shoulder/torso pose angles:

```bash
uv run python scripts/experiment_reach_adaptive_cone.py path/to/reach_video.mov --show
```

The script writes a side-by-side video and per-frame measurements to `results/reach_cone_experiment/<video-name>/`. The reach score is an experimental 2D pose proxy, not a calibrated distance estimate. This experiment uses Holistic's pose and hand landmarks directly; it tests the reach cue, not the combined hand-detector fallback. Adjust `--min-cone-length`, `--max-cone-length`, `--elbow-weight`, `--shoulder-weight`, and `--score-alpha` to compare mappings. The production video pipeline is not changed by this experiment.

### MediaPipe scripts

The preferred MediaPipe pipeline to build on is `combining_holisitc_pose_hand.py`. It combines Holistic pose/hand detections with the standalone Hand Landmarker and crop fallback, then selects final left/right hand landmarks. Its output folder is named after the input dataset folder. With the current input, the ensemble JSON, visualizations, fallback crops, scores, priors, and derived visualizations are grouped under `results/combined_holistic_pose_hand/1_2_3_andrea_merged/`:

```bash
uv run python scripts/run_mediapipe/combining_holisitc_pose_hand.py
```

The other scripts in `scripts/run_mediapipe/` are retained as earlier approaches and experiments. They are useful for comparison and reference; the ensemble script above is the current choice for further development.

`pose_crop&handlandmarker.py` is an earlier, separate approach that also estimates a 2D pointing direction. It is retained for comparison and reference; it is not an input to the current prior workflow.

### Create and inspect geometric pointing priors

The prior generator reads the ensemble JSON produced by `combining_holisitc_pose_hand.py`. It estimates a 2D direction from each final hand's index-finger MCP landmark (5) to its fingertip (8), then writes `.npz` priors and a `manifest.json` alongside the ensemble run:

```bash
uv run python scripts/prior/create_numeric_prior.py
uv run python scripts/visualisering/visualize_pointing_prior.py
```

The visualization is saved in `results/combined_holistic_pose_hand/1_2_3_andrea_merged/pointing_prior_visualizations/`.

### Score pointing hands and draw the selected direction

These scripts use the ensemble JSON and images in `data/3_exported_from_fiftyone/1_2_3_andrea_merged/data/`:

```bash
uv run python scripts/prior/score_pointing_hand.py
uv run python scripts/visualisering/draw_arrow.py
```

The score visualizations and `scores.json` are written to `results/combined_holistic_pose_hand/1_2_3_andrea_merged/pointing_hand_scores/`. Arrow visualizations are written to `results/combined_holistic_pose_hand/1_2_3_andrea_merged/pointing_direction_visualizations/`.

### Check the AbSViT checkpoint and run the pointing-prior experiment

The baseline script uses `checkpoints/absvit_tiny.pth`. It reports any missing or unexpected checkpoint weights; shape mismatches are printed and stop execution.

```bash
uv run python scripts/test_absvit_baseline.py
```

The prior experiment expects images under `data/funker_bra_20okt/`, priors under `results/combined_holistic_pose_hand/1_2_3_andrea_merged/numeric_pointing_prior/`, and `checkpoints/absvit_tiny_patch8_224_gap.pth`:

```bash
uv run python scripts/test_absvit_prior.py
uv run python scripts/evaluate_bbox.py
```

The experiment writes maps and figures under `data/results/absvit_prior_test_28x28_lay9/`. The bounding-box evaluator reads `data/funker_bra_20okt/labels.json` and those maps, then writes its figures and CSV summary under `data/results/bbox_eval_28x28_lay9/`.

## Notes

- Data, checkpoints, and generated results are intentionally excluded from Git. Keep copies elsewhere if they need to be preserved.
- Some dataset preparation scripts use paths tied to particular local exports. Check their path settings before using them with a different dataset.
- `main.py` is currently a placeholder. A single end-to-end pipeline entry point can be added after the current branch work is integrated.
