"""Visual experiment: compare fixed and reach-adaptive heat in a fixed cone.

This is deliberately separate from the production video pipeline. It uses
2D pose landmarks to inspect whether elbow and shoulder angles are useful
proxies for near/far pointing and compare heat falloff profiles.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prior.create_numeric_prior import (
    CONE_LENGTH,
    create_cone_heatmap,
    get_pointing_data,
)
from scripts.prior.score_pointing_hand import score_pointing_hand
from src.reach_estimator import (
    ReachScoreSmoother,
    adaptive_heat_extent,
    make_reach_score,
    measure_arm,
)
from src.video_hand_detector import VideoHandDetector


def parse_source(value):
    return int(value) if value.isdecimal() else value


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed-cone heat profiles using arm reach on a video. "
            "This experiment does not modify the full pipeline."
        )
    )
    parser.add_argument("source", type=parse_source, help="Video path or camera index")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder (default: results/reach_heat_experiment/<source-name>)",
    )
    parser.add_argument("--max-frames", type=int, help="Optional frame limit")
    parser.add_argument("--show", action="store_true", help="Show the comparison live")
    parser.add_argument(
        "--near-heat-extent",
        type=float,
        default=0.35,
        help="Near-reach heat decay distance, as a fraction of fixed cone length",
    )
    parser.add_argument(
        "--far-heat-extent",
        type=float,
        default=1.50,
        help="Far-reach heat decay distance, as a fraction of fixed cone length",
    )
    parser.add_argument(
        "--elbow-weight",
        type=float,
        default=0.70,
        help="Weight of elbow extension in the reach score",
    )
    parser.add_argument(
        "--shoulder-weight",
        type=float,
        default=0.30,
        help="Weight of shoulder-to-torso angle in the reach score",
    )
    parser.add_argument(
        "--score-alpha",
        type=float,
        default=0.35,
        help="EMA weight for the current reach score; higher reacts faster",
    )
    parser.add_argument(
        "--min-pointing-score",
        type=float,
        default=0.60,
        help="Mark hands below this hand-shape score as not pointing",
    )
    return parser


def choose_hand(hands):
    """Return the highest hand-shape score, even when below pointing threshold."""
    scored = []
    for hand in hands:
        score = score_pointing_hand(hand["landmarks"])
        hand["pointing_score"] = score["score"]
        hand["pointing_components"] = score["components"]
        scored.append(hand)
    return max(scored, key=lambda hand: hand["pointing_score"]) if scored else None


def make_cone(tip, direction, width, height, heat_extent=None):
    grid_width = 448
    grid_height = max(1, round(grid_width * height / width))
    cone = create_cone_heatmap(
        tip,
        direction,
        grid_width,
        grid_height,
        cone_length_fraction=CONE_LENGTH,
        longitudinal_decay_fraction=heat_extent,
    )
    return cv2.resize(cone, (width, height), interpolation=cv2.INTER_LINEAR)


def overlay_cone(frame, cone, opacity=0.45):
    color = cv2.applyColorMap(
        np.uint8(np.clip(cone, 0.0, 1.0) * 255), cv2.COLORMAP_JET
    )
    blended = cv2.addWeighted(frame, 1.0 - opacity, color, opacity, 0)
    mask = cone > 0.01
    output = frame.copy()
    output[mask] = blended[mask]
    return output


def annotate_panel(image, title, hand, arm, raw_score, smooth_score,
                   heat_extent, pointing_threshold):
    cv2.putText(
        image, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
        0.85, (255, 255, 255), 2, cv2.LINE_AA,
    )
    if hand is None:
        cv2.putText(
            image, "No hand detected", (18, 68), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 255), 2, cv2.LINE_AA,
        )
        return image

    is_pointing = hand["pointing_score"] >= pointing_threshold
    rows = [
        f"Hand: {hand['side']} | shape score: {hand['pointing_score']:.2f} "
        f"({'pointing' if is_pointing else 'below threshold'})",
    ]
    if arm is not None:
        rows.extend([
            f"Elbow angle: {arm['elbow_angle_degrees']:.1f} deg",
            f"Shoulder/torso: {arm['shoulder_torso_angle_degrees']:.1f} deg",
            f"Reach score raw/smoothed: {raw_score:.2f}/{smooth_score:.2f}",
            (
                "Uniform heat along cone" if heat_extent is None else
                f"Heat decay extent: {heat_extent:.2f} x cone length"
            ),
        ])
    else:
        rows.append("Pose arm measurements unavailable")

    for row_index, text in enumerate(rows, start=2):
        cv2.putText(
            image, text, (18, 32 * row_index), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return image


def _source_name(source):
    return f"camera_{source}" if isinstance(source, int) else Path(source).stem


def run_experiment(args):
    if not 0.0 < args.score_alpha <= 1.0:
        raise ValueError("--score-alpha must be greater than 0 and at most 1.")
    if args.elbow_weight < 0 or args.shoulder_weight < 0:
        raise ValueError("Score weights cannot be negative.")
    if args.elbow_weight + args.shoulder_weight <= 0:
        raise ValueError("At least one arm-angle weight must be greater than zero.")
    if not 0.0 <= args.min_pointing_score <= 1.0:
        raise ValueError("--min-pointing-score must be between 0 and 1.")
    if not 0 < args.near_heat_extent <= args.far_heat_extent:
        raise ValueError("Heat extent must satisfy 0 < near <= far.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be greater than zero.")

    output_dir = args.output_dir or (
        ROOT / "results" / "reach_heat_experiment" / _source_name(args.source)
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "reach_heat_comparison.mp4"
    records_path = output_dir / "measurements.jsonl"

    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    detector = VideoHandDetector(ROOT / "models" / "mediapipe")
    writer = None
    frame_index = 0
    last_timestamp_ms = -1
    reach_smoother = ReachScoreSmoother(
        alpha=args.score_alpha, missing_hold_frames=0
    )
    try:
        with records_path.open("w", encoding="utf-8") as records:
            records.write(json.dumps({
                "type": "metadata",
                "source": str(args.source),
                "fps": fps,
                "fixed_cone_length": CONE_LENGTH,
                "cone_length": CONE_LENGTH,
                "near_heat_extent": args.near_heat_extent,
                "far_heat_extent": args.far_heat_extent,
                "elbow_weight": args.elbow_weight,
                "shoulder_weight": args.shoulder_weight,
                "score_alpha": args.score_alpha,
                "min_pointing_score": args.min_pointing_score,
                "reach_score_note": (
                    "2D pose proxy from elbow opening and shoulder angle; "
                    "not a calibrated physical-distance estimate"
                ),
            }, ensure_ascii=False) + "\n")

            while args.max_frames is None or frame_index < args.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (2 * width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create {video_path}")

                timestamp_ms = max(
                    last_timestamp_ms + 1,
                    int(round(frame_index * 1000.0 / fps)),
                )
                last_timestamp_ms = timestamp_ms
                hands = detector.detect(frame, timestamp_ms)
                hand = choose_hand(hands)
                arm = measure_arm(
                    detector.last_pose_landmarks,
                    hand["side"] if hand else None,
                    detector.last_pose_visibility,
                )
                raw_reach_score = make_reach_score(
                    arm, args.elbow_weight, args.shoulder_weight
                )
                smooth_reach_score = reach_smoother.update(raw_reach_score)
                adaptive_extent = adaptive_heat_extent(
                    smooth_reach_score,
                    args.near_heat_extent,
                    args.far_heat_extent,
                )

                pointing = None
                if (
                    hand is not None
                    and hand["pointing_score"] >= args.min_pointing_score
                ):
                    pointing = get_pointing_data(
                        {"final_landmarks": hand["landmarks"]}, width, height
                    )

                fixed_view = frame.copy()
                adaptive_view = frame.copy()
                cone_tip = None
                cone_direction = None
                if pointing is not None:
                    cone_tip, cone_direction = pointing
                    fixed_cone = make_cone(
                        cone_tip, cone_direction, width, height
                    )
                    fixed_view = overlay_cone(frame, fixed_cone)
                    if adaptive_extent is not None:
                        adaptive_cone = make_cone(
                            cone_tip, cone_direction, width, height, adaptive_extent
                        )
                        adaptive_view = overlay_cone(frame, adaptive_cone)

                fixed_view = annotate_panel(
                    fixed_view,
                    "FIXED CONE, UNIFORM HEAT",
                    hand,
                    arm,
                    raw_reach_score,
                    smooth_reach_score,
                    None,
                    args.min_pointing_score,
                )
                adaptive_view = annotate_panel(
                    adaptive_view,
                    "FIXED CONE, REACH-ADAPTIVE HEAT",
                    hand,
                    arm,
                    raw_reach_score,
                    smooth_reach_score,
                    adaptive_extent,
                    args.min_pointing_score,
                )
                if pointing is None:
                    for panel in (fixed_view, adaptive_view):
                        cv2.putText(
                            panel,
                            "NO VALID POINTING DIRECTION",
                            (18, panel.shape[0] - 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                comparison = np.concatenate((fixed_view, adaptive_view), axis=1)
                writer.write(comparison)

                records.write(json.dumps({
                    "type": "frame",
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "hand_side": hand["side"] if hand else None,
                    "pointing_score": hand["pointing_score"] if hand else None,
                    "is_pointing": (
                        hand["pointing_score"] >= args.min_pointing_score
                        if hand else False
                    ),
                    "arm_measurements": arm,
                    "reach_score_raw": raw_reach_score,
                    "reach_score_smoothed": smooth_reach_score,
                    "cone_length": CONE_LENGTH,
                    "adaptive_heat_extent": adaptive_extent,
                    "cone_tip": cone_tip,
                    "cone_direction": cone_direction,
                }, ensure_ascii=False) + "\n")

                if args.show:
                    cv2.imshow("Fixed cone: uniform vs adaptive heat", comparison)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_index += 1
                if frame_index % 30 == 0:
                    print(f"Processed {frame_index} frames")
    finally:
        capture.release()
        detector.close()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if frame_index == 0:
        raise RuntimeError("Videokilden returnerte ingen frames.")
    print(f"Done: {frame_index} frames")
    print(f"Comparison video: {video_path}")
    print(f"Measurements: {records_path}")


def main():
    args = build_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
