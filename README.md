# Task-Guided Attention

This project uses a person's pointing gesture to guide visual attention and select an object in a video. The main entry point is `main.py`; model and selection settings are in `configs/full_pipeline.py`.

## Main video pipeline

For each video frame, the pipeline:

1. Detects hands and pose landmarks with MediaPipe Holistic and Hand Landmarker. A crop fallback uses pose landmarks to look for a hand if neither model finds one.
2. Scores detected hands as pointing gestures. The score combines index-finger straightness (65%) with the average curl of the middle, ring, and little fingers (35%); the default minimum score is `0.60`.
3. Builds a pointing prior from the index finger, then runs AbSViT with top-down feedback. Without a valid pointing prior, AbSViT produces a bottom-up focus map, but no object is selected.
4. Runs YOLO26n on the full frame and scores each candidate box using the mean focus value inside it.
5. Tracks candidates across frames and draws the selected box in the output video.

### Requirements

Use Python 3.12 and `uv`:

```bash
uv sync
```

The MediaPipe task files are included in `models/mediapipe/`. Download the AbSViT-Tiny patch-8 GAP checkpoint required by the video pipeline:

```bash
mkdir -p checkpoints
curl -fL "https://berkeley.box.com/shared/static/7415yz4d1l5z0ur6x32k35f8y99zgynq.pth" -o checkpoints/absvit_tiny_patch8_224_gap.pth
```

Ultralytics loads `yolo26n.pt` on first use. Checkpoints and generated results are excluded from Git.

### Run

Run the full video:

```bash
uv run main.py path/to/video.mp4
```

To show the output windows while processing, add `--show`. For a camera, use its index, usually `0`. For example:

```bash
uv run main.py 0 --show --max-frames 300
```

By default, outputs are written to `results/video_pipeline/<video-name>/`:

- `selected_object.mp4`: the selected object box over the original video.
- `verification.mp4`: focus heatmap, hand landmarks, YOLO boxes, and scores.
- `frames.jsonl`: run configuration and per-frame hand, detection, and selection data.

Use `--output-dir PATH` to choose another output directory. Use `--max-frames N` to limit processing.

### Object selection and tracking

The current defaults are in `configs/full_pipeline.py`:

- A candidate needs YOLO confidence of at least `0.25`, mean focus score of at least `0.30`, and a valid pointing prior.
- Candidate ranking uses focus alone (`saliency_weight = 1`, `yolo_confidence_weight = 0`). YOLO confidence is a filter, not part of the ranking score.
- Detections are matched across frames by class and bounding-box overlap (IoU; threshold `0.20`). Scores and box positions are smoothed over time.
- A new candidate must exceed the current track's smoothed score by `0.08` for four consecutive frames before the selection switches. While a valid pointing prior remains available, the selected box is held for up to three frames during a detection gap.

The optional reach and person-scale adaptations are enabled by default. Reach changes the heat falloff along a cone of fixed length; torso size adjusts the cone's width. These are 2D image-based proxies, not measurements of physical distance. All thresholds, weights, cone settings, model paths, and tracking parameters can be changed in `configs/full_pipeline.py`.

## Evaluation and supporting workflows

To evaluate object selection against the FiftyOne export:

```bash
uv run scripts/evaluate_pipeline_bbox.py --max-images 5 --save-overlays
```

Remove `--max-images 5` to process the full export. The script expects labels and images under `data/3_exported_from_fiftyone/1_2_3_andrea_merged/` and writes metrics and overlays to `results/pipeline_bbox_eval/`.

The repository also contains scripts for preparing datasets, running the MediaPipe image ensemble, generating and visualizing pointing priors, and experimenting with AbSViT focus maps and reach-adaptive heat. These are supporting research workflows; they are not required to run `main.py`. Some use dataset paths configured inside the individual scripts. Earlier MediaPipe approaches are retained in `scripts/run_mediapipe/` for comparison; `combining_holisitc_pose_hand.py` is the current image-folder ensemble.

## Project layout

- `main.py`: command-line entry point for video processing.
- `configs/full_pipeline.py`: pipeline defaults and thresholds.
- `scripts/full_pipeline.py`: per-frame orchestration.
- `src/`: shared hand detection, reach estimation, and temporal object tracking.
- `scripts/prior/`: pointing-hand score and prior geometry.
- `scripts/evaluate_pipeline_bbox.py`: evaluation against annotated images.
- `models/AbSViT/` and `models/mediapipe/`: model implementations and MediaPipe task files.
- `data/`, `checkpoints/`, and `results/`: local inputs, model weights, and generated outputs; these are not stored in Git.
