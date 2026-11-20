# Task-Guided Attention

Research code for studying how human cues, especially pointing direction, can guide visual attention in AbSViT. The repository contains data preparation scripts, MediaPipe landmark extraction, pointing-prior generation, model experiments, and visual evaluation tools.

## Project layout

- `models/AbSViT/`: Vision Transformer and AbSViT model implementations.
- `models/mediapipe/`: MediaPipe task model files.
- `scripts/data_processing/`: dataset export, merging, filtering, and file organization.
- `scripts/run_mediapipe/`: hand and pose landmark extraction and combination.
- `scripts/prior/`: pointing-hand scoring and numeric pointing-prior generation.
- `scripts/full_pipeline.py`: end-to-end video pipeline.
- `scripts/visualisering/`: visual checks for labels, pointing priors, and predicted directions.
- `configs/full_pipeline.py`: video-pipeline model, detection, scoring, and reach-adaptation settings.
- `scripts/test_absvit_baseline.py`: checkpoint loading and a minimal forward pass.
- `scripts/absvit_prior.py`: AbSViT experiment using a spatial pointing prior and reusable helpers.
- `scripts/evaluate_bbox.py`: compares model focus maps with ground-truth boxes.
- `data/`, `results/`, and `checkpoints/`: local inputs and generated artifacts. These folders are ignored by Git and need to be present locally.
- `main.py`: command-line entry point for the video pipeline.
- `src/`: reusable video hand detection, temporal object selection, and reach-estimation components.

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

The prior experiment and video pipeline use the AbSViT Tiny patch-8 GAP checkpoint. Download the checkpoint used by the official AbSViT demo with:

```bash
curl -fL "https://berkeley.box.com/shared/static/7415yz4d1l5z0ur6x32k35f8y99zgynq.pth" -o checkpoints/absvit_tiny_patch8_224_gap.pth
```

This is a different architecture from `absvit_tiny.pth` above, which is patch-16, so the files cannot be substituted for each other.

## Current workflows

Run commands from the repository root. Local datasets and checkpoints must exist at the paths described below.

### Video pipeline

The full pipeline takes a video file (or camera index), finds hands with the combined MediaPipe approach, selects a pointing hand, conditions AbSViT with the cone prior, detects objects with YOLO, and ranks YOLO boxes using AbSViT focus and YOLO confidence. A focus heatmap is drawn on every frame of the verification video: it is prior-conditioned when a pointing hand is available and bottom-up otherwise. Object selection still requires a valid pointing prior. Optional reach adaptation keeps the cone's length fixed and adjusts how quickly heat fades along it: a more extended arm keeps more heat toward the far end.

The pipeline reuses the pointing-hand score from `scripts/prior/score_pointing_hand.py`, pointing geometry from `scripts/prior/create_numeric_prior.py`, AbSViT preprocessing/prior/model helpers from `scripts/absvit_prior.py`, and YOLO frame inference from `scripts/yolo_detection.py`. This keeps image experiments and video processing on the same implementations where their workflows overlap.

```bash
uv run python main.py path/to/video.mp4 --show
```

For a camera, pass its index, usually `0`. The patch-8 GAP checkpoint must be present at `checkpoints/absvit_tiny_patch8_224_gap.pth`; Ultralytics loads `yolo26n.pt` by default.

Tune the pipeline in one place: `configs/full_pipeline.py`. It contains the model and device, YOLO inference size and confidence thresholds, pointing-hand and MediaPipe thresholds, cone geometry, AbSViT prior strength and focus layer, signal weights, temporal smoothing and switching, output defaults, and heatmap opacity. By default, object choice combines normalized AbSViT saliency and YOLO confidence with weights `0.80` and `0.20`. Eligible boxes must also pass `min_yolo_confidence`; `yolo_inference_confidence` controls which lower-confidence boxes YOLO returns for verification.

YOLO detections are associated across frames by class and IoU. Each track gets a smoothed score and bbox; a new object must beat the current one by `track_switch_margin` for `switch_confirmation_frames` before the selection changes. The selected bbox is held briefly through detector dropouts. Adjust `score_smoothing_alpha` and `bbox_smoothing_alpha` to trade smoothness for responsiveness.

The default output folder is `results/video_pipeline/<video-name>/` and contains:

- `selected_object.mp4`: the selected object bounding box per frame.
- `verification.mp4`: all YOLO boxes and confidences, selected hand landmarks, and a continuous AbSViT focus heatmap.
- `frames.jsonl`: one metadata record and one record per processed frame, including hand scores and all object detections.

Use `--max-frames 100` for a short run or `--output-dir PATH` to choose another output folder. Add `--show` to display both outputs while processing. Device and other tunable model settings are in `configs/full_pipeline.py`.

### Experiment with reach-adaptive heat

Compare uniform heat with reach-adaptive heat inside the same fixed cone:

```bash
uv run python scripts/experiment_reach_adaptive_cone.py path/to/reach_video.mov --show
```

The script writes a side-by-side video and per-frame measurements to `results/reach_heat_experiment/<video-name>/`. It uses the same combined hand detector and reach-measurement functions as the full pipeline. The reach score is an experimental 2D pose proxy, not a calibrated distance estimate. Adjust `--near-heat-extent`, `--far-heat-extent`, `--elbow-weight`, `--shoulder-weight`, and `--score-alpha` to compare mappings.

Set `use_reach_adaptive_heat = True` in `configs/full_pipeline.py` to enable it; the heat falloff range, signal weights, smoothing, and missing-pose hold are configurable there. The cone length remains `cone_length` for every frame.

### MediaPipe scripts

The preferred MediaPipe pipeline to build on is `combining_holisitc_pose_hand.py`. Both this image-folder script and `scripts/full_pipeline.py` use the shared `VideoHandDetector` implementation in `src/video_hand_detector.py`. It combines Holistic pose/hand detections with the standalone Hand Landmarker and crop fallback, then selects final left/right hand landmarks. The image script uses MediaPipe image mode; the video pipeline uses video mode for frame-to-frame tracking. Its output folder is named after the input dataset folder. With the current input, the ensemble JSON, visualizations, fallback crops, scores, priors, and derived visualizations are grouped under `results/combined_holistic_pose_hand/1_2_3_andrea_merged/`:

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
uv run python scripts/absvit_prior.py
uv run python scripts/evaluate_bbox.py
```

The experiment writes maps and figures under `data/results/absvit_prior_test_28x28_lay9/`. The bounding-box evaluator reads `data/funker_bra_20okt/labels.json` and those maps, then writes its figures and CSV summary under `data/results/bbox_eval_28x28_lay9/`.

## Notes

- Data, checkpoints, and generated results are intentionally excluded from Git. Keep copies elsewhere if they need to be preserved.
- Some dataset preparation scripts use paths tied to particular local exports. Check their path settings before using them with a different dataset.
- Reach-adaptive heat uses 2D pose geometry as a relative reach proxy; it is not a calibrated physical-distance estimate.
