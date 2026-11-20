"""Visual experiment: compare a fixed cone with one scaled by arm reach.

This is deliberately separate from the production video pipeline. It uses
2D pose landmarks to inspect whether elbow and shoulder angles are useful
proxies for near/far pointing before the cone is changed in full_pipeline.py.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import mediapipe as mp
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


POSE_IDS = {
    "left": {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23},
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24},
}


class HolisticVideoDetector:
    """Run Holistic in video mode and expose pose plus both hand landmark sets."""

    def __init__(self, model_path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Fant ikke MediaPipe-modellen: {model_path}")
        vision = mp.tasks.vision
        options = vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.3,
            min_pose_landmarks_confidence=0.3,
            min_hand_landmarks_confidence=0.3,
        )
        self.model = vision.HolisticLandmarker.create_from_options(options)
        self.last_pose_landmarks = None
        self.last_pose_visibility = None

    @staticmethod
    def _pixels(landmarks, width, height):
        if not landmarks:
            return None
        return [
            {"x": int(point.x * width), "y": int(point.y * height)}
            for point in landmarks
        ]

    def detect(self, frame_bgr, timestamp_ms):
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.model.detect_for_video(image, timestamp_ms)
        pose = result.pose_landmarks
        self.last_pose_landmarks = self._pixels(pose, width, height)
        self.last_pose_visibility = (
            [float(getattr(point, "visibility", 1.0)) for point in pose]
            if pose else None
        )
        hands = []
        for side in ("left", "right"):
            points = self._pixels(
                getattr(result, f"{side}_hand_landmarks", None), width, height
            )
            if points:
                hands.append({"side": side, "landmarks": points})
        return hands

    def close(self):
        self.model.close()


def parse_source(value):
    return int(value) if value.isdecimal() else value


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed and arm-reach-adaptive pointing cones on a video. "
            "This experiment does not modify the full pipeline."
        )
    )
    parser.add_argument("source", type=parse_source, help="Video path or camera index")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder (default: results/reach_cone_experiment/<source-name>)",
    )
    parser.add_argument("--max-frames", type=int, help="Optional frame limit")
    parser.add_argument("--show", action="store_true", help="Show the comparison live")
    parser.add_argument(
        "--min-cone-length",
        type=float,
        default=0.20,
        help="Adaptive cone length for a fully bent/near reach, as frame-diagonal fraction",
    )
    parser.add_argument(
        "--max-cone-length",
        type=float,
        default=CONE_LENGTH,
        help="Adaptive cone length for a fully extended/far reach, as frame-diagonal fraction",
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


def _distance(point_a, point_b):
    return float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))


def _angle_degrees(point_a, vertex, point_b):
    """Return the smaller 2D angle at vertex in the range [0, 180]."""
    a = np.asarray(point_a, dtype=float) - np.asarray(vertex, dtype=float)
    b = np.asarray(point_b, dtype=float) - np.asarray(vertex, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-8:
        return None
    cosine = np.clip(np.dot(a, b) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def measure_arm(pose_landmarks, side, pose_visibility=None):
    """Measure elbow and shoulder/torso angles from pixel pose landmarks."""
    if side not in POSE_IDS or pose_landmarks is None:
        return None
    ids = POSE_IDS[side]
    if len(pose_landmarks) <= max(ids.values()):
        return None
    if pose_visibility is not None and any(
        pose_visibility[index] < 0.1 for index in ids.values()
    ):
        return None

    points = {
        name: np.array(
            [pose_landmarks[index]["x"], pose_landmarks[index]["y"]],
            dtype=float,
        )
        for name, index in ids.items()
    }
    elbow_angle = _angle_degrees(
        points["shoulder"], points["elbow"], points["wrist"]
    )
    shoulder_torso_angle = _angle_degrees(
        points["hip"], points["shoulder"], points["elbow"]
    )
    torso_length = _distance(points["shoulder"], points["hip"])
    shoulder_wrist_ratio = (
        _distance(points["shoulder"], points["wrist"]) / torso_length
        if torso_length > 1e-6 else None
    )

    if elbow_angle is None or shoulder_torso_angle is None:
        return None

    # Map 90..180 degrees of elbow opening and 15..90 degrees of arm lift
    # to [0,1]. These are experiment defaults, not calibrated thresholds.
    elbow_extension = float(np.clip((elbow_angle - 90.0) / 90.0, 0.0, 1.0))
    shoulder_extension = float(
        np.clip((shoulder_torso_angle - 15.0) / 75.0, 0.0, 1.0)
    )
    return {
        "elbow_angle_degrees": elbow_angle,
        "shoulder_torso_angle_degrees": shoulder_torso_angle,
        "shoulder_wrist_to_torso_ratio": shoulder_wrist_ratio,
        "elbow_extension_score": elbow_extension,
        "shoulder_extension_score": shoulder_extension,
    }


def choose_hand(hands):
    """Return the highest hand-shape score, even when below pointing threshold."""
    scored = []
    for hand in hands:
        score = score_pointing_hand(hand["landmarks"])
        hand["pointing_score"] = score["score"]
        hand["pointing_components"] = score["components"]
        scored.append(hand)
    return max(scored, key=lambda hand: hand["pointing_score"]) if scored else None


def make_reach_score(arm_measurements, elbow_weight, shoulder_weight):
    if arm_measurements is None:
        return None
    weight_total = elbow_weight + shoulder_weight
    if weight_total <= 0:
        return None
    return (
        elbow_weight * arm_measurements["elbow_extension_score"]
        + shoulder_weight * arm_measurements["shoulder_extension_score"]
    ) / weight_total


def make_cone(tip, direction, width, height, length):
    grid_width = 448
    grid_height = max(1, round(grid_width * height / width))
    cone = create_cone_heatmap(
        tip,
        direction,
        grid_width,
        grid_height,
        cone_length_fraction=length,
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
                   cone_length, pointing_threshold):
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
            f"Cone length: {cone_length:.3f} x frame diagonal",
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
        raise ValueError("--score-alpha må være større enn 0 og maks 1.")
    if args.elbow_weight < 0 or args.shoulder_weight < 0:
        raise ValueError("Score-vektene kan ikke være negative.")
    if args.elbow_weight + args.shoulder_weight <= 0:
        raise ValueError("Minst én armvinkel-vekt må være større enn null.")
    if not 0.0 <= args.min_pointing_score <= 1.0:
        raise ValueError("--min-pointing-score må være mellom 0 og 1.")
    if not 0 < args.min_cone_length <= args.max_cone_length:
        raise ValueError("Kjeglelengdene må oppfylle 0 < min <= maks.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames må være større enn null.")

    output_dir = args.output_dir or (
        ROOT / "results" / "reach_cone_experiment" / _source_name(args.source)
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "reach_cone_comparison.mp4"
    records_path = output_dir / "measurements.jsonl"

    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Kunne ikke åpne videokilde: {args.source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    detector = HolisticVideoDetector(
        ROOT / "models" / "mediapipe" / "holistic_landmarker.task"
    )
    writer = None
    frame_index = 0
    last_timestamp_ms = -1
    smooth_reach_score = None
    try:
        with records_path.open("w", encoding="utf-8") as records:
            records.write(json.dumps({
                "type": "metadata",
                "source": str(args.source),
                "fps": fps,
                "fixed_cone_length": CONE_LENGTH,
                "min_cone_length": args.min_cone_length,
                "max_cone_length": args.max_cone_length,
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
                        raise RuntimeError(f"Kunne ikke opprette {video_path}")

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
                if raw_reach_score is None:
                    smooth_reach_score = None
                    adaptive_length = None
                else:
                    smooth_reach_score = (
                        raw_reach_score if smooth_reach_score is None else
                        args.score_alpha * raw_reach_score
                        + (1.0 - args.score_alpha) * smooth_reach_score
                    )
                    adaptive_length = (
                        args.min_cone_length
                        + smooth_reach_score
                        * (args.max_cone_length - args.min_cone_length)
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
                fixed_length = CONE_LENGTH
                cone_tip = None
                cone_direction = None
                if pointing is not None:
                    cone_tip, cone_direction = pointing
                    fixed_cone = make_cone(
                        cone_tip, cone_direction, width, height, fixed_length
                    )
                    fixed_view = overlay_cone(frame, fixed_cone)
                    if adaptive_length is not None:
                        adaptive_cone = make_cone(
                            cone_tip, cone_direction, width, height, adaptive_length
                        )
                        adaptive_view = overlay_cone(frame, adaptive_cone)

                fixed_view = annotate_panel(
                    fixed_view,
                    f"FIXED CONE ({fixed_length:.2f})",
                    hand,
                    arm,
                    raw_reach_score,
                    smooth_reach_score,
                    fixed_length,
                    args.min_pointing_score,
                )
                adaptive_view = annotate_panel(
                    adaptive_view,
                    "REACH-ADAPTIVE CONE",
                    hand,
                    arm,
                    raw_reach_score,
                    smooth_reach_score,
                    adaptive_length,
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
                    "fixed_cone_length": fixed_length,
                    "adaptive_cone_length": adaptive_length,
                    "cone_tip": cone_tip,
                    "cone_direction": cone_direction,
                }, ensure_ascii=False) + "\n")

                if args.show:
                    cv2.imshow("Fixed vs reach-adaptive cone", comparison)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_index += 1
                if frame_index % 30 == 0:
                    print(f"Behandlet {frame_index} frames")
    finally:
        capture.release()
        detector.close()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if frame_index == 0:
        raise RuntimeError("Videokilden returnerte ingen frames.")
    print(f"Ferdig: {frame_index} frames")
    print(f"Sammenligningsvideo: {video_path}")
    print(f"Målinger: {records_path}")


def main():
    args = build_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
