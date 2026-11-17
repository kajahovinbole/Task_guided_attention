import argparse
from pathlib import Path

from scripts.full_pipeline import ROOT, run_full_pipeline


def parse_video_source(value):
    """Accept either a video path or an integer camera index."""
    if value.isdecimal():
        return int(value)
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run hand detection, prior-conditioned AbSViT, and YOLO on a video."
        )
    )
    parser.add_argument(
        "source",
        type=parse_video_source,
        help="Video file path or camera index, for example video.mp4 or 0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder (default: results/video_pipeline/<source-name>)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "absvit_tiny_patch8_224_gap.pth",
        help="AbSViT Tiny patch-8 GAP checkpoint",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolo26n.pt",
        help="Ultralytics model name or checkpoint path (default: yolo26n.pt)",
    )
    parser.add_argument(
        "--min-pointing-score",
        type=float,
        default=0.6,
        help="Minimum hand score required to create a pointing prior",
    )
    parser.add_argument(
        "--min-yolo-confidence",
        type=float,
        default=0.25,
        help="Minimum YOLO confidence for a box to be considered for selection",
    )
    parser.add_argument(
        "--yolo-inference-confidence",
        type=float,
        default=0.05,
        help="Lower YOLO threshold so rejected boxes also appear in verification output",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="AbSViT device (default: use MPS, CUDA, then CPU when available)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show selected-object and verification windows while processing",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional frame limit, useful for a short initial run",
    )
    return parser


def main():
    args = build_parser().parse_args()
    run_full_pipeline(
        source=args.source,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        yolo_model_name=args.yolo_model,
        min_pointing_score=args.min_pointing_score,
        min_yolo_confidence=args.min_yolo_confidence,
        yolo_inference_confidence=args.yolo_inference_confidence,
        device_name=args.device,
        show=args.show,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
