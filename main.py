import argparse
from pathlib import Path

from configs.full_pipeline import DEFAULT_CONFIG
from scripts.full_pipeline import run_full_pipeline


def parse_video_source(value):
    """Accept either a video path or an integer camera index."""
    if value.isdecimal():
        return int(value)
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the configured hand-guided AbSViT and YOLO video pipeline. "
            "Tune model and scoring parameters in configs/full_pipeline.py."
        )
    )
    parser.add_argument(
        "source",
        type=parse_video_source,
        help="Video file path or camera index, for example data/video.mov or 0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the output folder set in configs/full_pipeline.py",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=None,
        help="Show the selected-object and verification windows",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Override the frame limit set in configs/full_pipeline.py",
    )
    return parser


def main():
    args = build_parser().parse_args()
    run_full_pipeline(
        source=args.source,
        config=DEFAULT_CONFIG,
        output_dir=args.output_dir,
        show=args.show,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
