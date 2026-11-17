import json
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "data" / "3_exported_from_fiftyone" / "1_2_3_andrea_merged" / "data"
RUN_NAME = IMAGE_DIR.parent.name
ENSEMBLE_RUN_DIR = ROOT / "results" / "combined_holistic_pose_hand" / RUN_NAME
SCORES_PATH = ENSEMBLE_RUN_DIR / "pointing_hand_scores" / "scores.json"
LANDMARKS_PATH = ENSEMBLE_RUN_DIR / "ensemble_results.json"
OUTPUT_DIR = ENSEMBLE_RUN_DIR / "pointing_direction_visualizations"

# Arrow length as a fraction of the image diagonal
ARROW_LENGTH = 0.25

# --------------------------------------------------
# Helpers
# --------------------------------------------------


def find_image(image_name):
    """Find the original image by filename."""

    candidates = [
        IMAGE_DIR / image_name,
        IMAGE_DIR / Path(image_name).name,
    ]

    for path in candidates:
        if path.is_file():
            return path

    matches = list(IMAGE_DIR.rglob(Path(image_name).name))
    return matches[0] if matches else None


def calculate_direction(landmarks):
    """
    Estimate pointing direction from landmark 5 to landmark 8.

    Landmark 5: index finger MCP (base knuckle)
    Landmark 8: index fingertip

    Returns the fingertip and a normalized direction vector.
    """

    if not isinstance(landmarks, list) or len(landmarks) != 21:
        return None

    try:
        base = np.array(
            [landmarks[5]["x"], landmarks[5]["y"]],
            dtype=float,
        )
        tip = np.array(
            [landmarks[8]["x"], landmarks[8]["y"]],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError):
        return None

    direction = tip - base
    length = np.linalg.norm(direction)

    if length < 1e-6:
        return None

    return tip, direction / length


def draw_direction(image, landmarks, side, score):
    """Draw the selected hand's estimated pointing direction."""

    result = calculate_direction(landmarks)

    if result is None:
        return False

    tip, direction = result

    height, width = image.shape[:2]
    diagonal = np.hypot(width, height)
    arrow_length = ARROW_LENGTH * diagonal

    start = tuple(np.round(tip).astype(int))
    end = tuple(np.round(tip + direction * arrow_length).astype(int))

    # Draw the estimated direction from the fingertip.
    cv2.arrowedLine(
        image,
        start,
        end,
        (0, 255, 0),
        thickness=3,
        tipLength=0.12,
    )

    # Highlight the fingertip.
    cv2.circle(image, start, 6, (0, 255, 0), -1)

    label = f"{side} hand | score={score:.2f}"

    label_position = (
        max(5, start[0] - 80),
        max(25, start[1] - 15),
    )

    cv2.putText(
        image,
        label,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return True


# --------------------------------------------------
# Main
# --------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SCORES_PATH.open(encoding="utf-8") as f:
        scores_data = json.load(f)

    with LANDMARKS_PATH.open(encoding="utf-8") as f:
        landmarks_data = json.load(f)

    # Index landmark records by image filename.
    landmarks_by_image = {
        Path(entry["image"]).name: entry
        for entry in landmarks_data
    }

    images_processed = 0
    arrows_drawn = 0
    skipped = 0

    for score_entry in scores_data:
        image_name = Path(score_entry["image"]).name
        selected_side = score_entry.get("selected_side")

        # No hand passed the pointing-score threshold.
        if selected_side not in ("left", "right"):
            skipped += 1
            continue

        landmark_entry = landmarks_by_image.get(image_name)

        if landmark_entry is None:
            print(f"No landmarks found for {image_name}")
            skipped += 1
            continue

        hand_data = landmark_entry.get("hands", {}).get(
            selected_side, {}
        )
        landmarks = hand_data.get("final_landmarks")

        # Retrieve the selected hand's score.
        selected_score = next(
            (
                item["score"]
                for item in score_entry.get("scores", [])
                if item.get("side") == selected_side
            ),
            None,
        )

        if selected_score is None:
            print(f"No score found for {image_name}")
            skipped += 1
            continue

        image_path = find_image(image_name)

        if image_path is None:
            print(f"Original image not found: {image_name}")
            skipped += 1
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read image: {image_path}")
            skipped += 1
            continue

        success = draw_direction(
            image,
            landmarks,
            selected_side,
            float(selected_score),
        )

        if not success:
            print(f"Invalid landmarks for {image_name}")
            skipped += 1
            continue

        output_path = OUTPUT_DIR / image_name

        if cv2.imwrite(str(output_path), image):
            arrows_drawn += 1
        else:
            print(f"Could not save: {output_path}")

        images_processed += 1

    print("\nFinished.")
    print(f"Images processed: {images_processed}")
    print(f"Pointing directions drawn: {arrows_drawn}")
    print(f"Skipped: {skipped}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
