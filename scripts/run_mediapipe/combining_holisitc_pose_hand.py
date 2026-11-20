"""Run the shared hand-detection ensemble over an image folder."""

from pathlib import Path
import json

import cv2

from scripts.prior.score_pointing_hand import HAND_CONNECTIONS
from src.video_hand_detector import VideoHandDetector


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "data/3_exported_from_fiftyone/1_2_3_andrea_merged/data"
RUN_NAME = IMAGE_DIR.parent.name
MODEL_DIR = ROOT / "models/mediapipe"
OUTPUT_DIR = ROOT / "results/combined_holistic_pose_hand" / RUN_NAME
VIS_DIR = OUTPUT_DIR / "visualizations"
CROP_DIR = OUTPUT_DIR / "fallback_crops"
JSON_PATH = OUTPUT_DIR / "ensemble_results.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def as_xy(points):
    if points is None:
        return None
    return [[point["x"], point["y"]] for point in points]


def draw_hand(image, points, color, thickness=2):
    if not points:
        return
    height, width = image.shape[:2]
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (x, y), 3, color, -1)
    for start, end in HAND_CONNECTIONS:
        p1, p2 = tuple(points[start]), tuple(points[end])
        if all(0 <= x < width and 0 <= y < height for x, y in (p1, p2)):
            cv2.line(image, p1, p2, color, thickness)


def main():
    for directory in (OUTPUT_DIR, VIS_DIR, CROP_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Fant ikke bildemappen: {IMAGE_DIR}")

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"Fant ingen bilder i {IMAGE_DIR}")

    records = []
    with VideoHandDetector(
        MODEL_DIR, running_mode="image", crop_output_dir=CROP_DIR
    ) as detector:
        for index, path in enumerate(image_paths, start=1):
            image = cv2.imread(str(path))
            if image is None:
                print(f"Kunne ikke lese {path.name}")
                continue
            height, width = image.shape[:2]
            detected_hands = detector.detect(
                image, timestamp_ms=index, frame_name=path.stem
            )
            visualization = image.copy()
            for point in detector.last_pose_landmarks or []:
                x, y = point["x"], point["y"]
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(visualization, (x, y), 3, (255, 0, 255), -1)
            combined = {
                side: {
                    "final_landmarks": None,
                    "holistic_landmarks": None,
                    "hand_landmarker_landmarks": None,
                    "fallback_landmarks": None,
                    "decision": "not_found",
                    "confidence_label": "none",
                    "disagreement_normalized": None,
                    "crop_attempts": [],
                }
                for side in ("left", "right")
            }

            for hand in detected_hands:
                side = hand["side"]
                final_points = as_xy(hand["landmarks"])
                holistic_points = as_xy(hand.get("holistic_landmarks"))
                direct_points = as_xy(hand.get("hand_landmarker_landmarks"))
                fallback_points = as_xy(hand.get("fallback_landmarks"))

                if holistic_points:
                    draw_hand(
                        visualization,
                        holistic_points,
                        (0, 255, 0) if side == "left" else (0, 255, 255),
                    )
                if direct_points:
                    draw_hand(visualization, direct_points, (255, 100, 0))
                if fallback_points:
                    draw_hand(visualization, fallback_points, (255, 255, 0))
                if final_points:
                    for x, y in final_points:
                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(visualization, (x, y), 5, (255, 255, 255), 1)

                combined[side] = {
                    "final_landmarks": final_points,
                    "holistic_landmarks": holistic_points,
                    "hand_landmarker_landmarks": direct_points,
                    "fallback_landmarks": fallback_points,
                    "decision": hand["decision"],
                    "confidence_label": hand["confidence_label"],
                    "disagreement_normalized": hand["disagreement_normalized"],
                    "crop_attempts": [],
                }
                print(
                    f"  {side}: {hand['decision']}, "
                    f"disagreement={hand['disagreement_normalized']}"
                )

            cv2.imwrite(str(VIS_DIR / path.name), visualization)
            records.append({
                "image": path.name,
                "width": width,
                "height": height,
                "pose_found": detector.last_pose_found,
                "hands": combined,
            })
            print(f"[{index}/{len(image_paths)}] {path.name}")

    with JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)
    print("\nFerdig.")
    print(f"Visualiseringer: {VIS_DIR}")
    print(f"Fallback-crops: {CROP_DIR}")
    print(f"JSON-resultater: {JSON_PATH}")


if __name__ == "__main__":
    main()
