
import json
from pathlib import Path

import cv2
import numpy as np


"""
Grønn: valgt pekehånd.
Rød: hånd som ikke ble valgt.
Score: vises ved hvert håndledd.
Ingen valgt hånd: dersom ingen hånd passerer terskelen.
"""


# --------------------------------------------------
# Paths and settings
# --------------------------------------------------

DATA_PATH = Path(
    "results/combined_holistic_pose_hand/tom_scene28_2/ensemble_results.json"
)

IMAGE_DIR = Path(
    "/Users/kajahovinbole/Task_guided_attention/"
    "data/3_exported_from_fiftyone/1_2_3_andrea_merged/data"
)

OUTPUT_DIR = Path(
    "results/combined_holistic_pose_hand/pointing_hand_scores"
)

THRESHOLD = 0.6

FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# --------------------------------------------------
# Pointing score
# --------------------------------------------------

def distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def finger_straightness(landmarks, finger_ids):
    points = [
        np.array(
            [landmarks[i]["x"], landmarks[i]["y"]],
            dtype=float,
        )
        for i in finger_ids
    ]

    path_length = sum(
        distance(points[i], points[i + 1])
        for i in range(len(points) - 1)
    )

    direct_distance = distance(points[0], points[-1])

    if path_length < 1e-6:
        return 0.0

    return float(
        np.clip(direct_distance / path_length, 0.0, 1.0)
    )


def score_pointing_hand(landmark_list):
    """
    Score a list of 21 landmarks.
    Each landmark must contain x and y coordinates.
    """

    if not isinstance(landmark_list, list) or len(landmark_list) != 21:
        return {
            "score": 0.0,
            "is_pointing": False,
            "components": {},
        }

    try:
        landmarks = {
            i: {
                "x": float(lm["x"]),
                "y": float(lm["y"]),
            }
            for i, lm in enumerate(landmark_list)
        }
    except (KeyError, TypeError, ValueError):
        return {
            "score": 0.0,
            "is_pointing": False,
            "components": {},
        }

    index_score = finger_straightness(
        landmarks, FINGERS["index"]
    )

    curled_score = float(np.mean([
        1.0 - finger_straightness(landmarks, FINGERS[name])
        for name in ("middle", "ring", "pinky")
    ]))

    wrist = [landmarks[0]["x"], landmarks[0]["y"]]
    index_mcp = [landmarks[5]["x"], landmarks[5]["y"]]
    index_tip = [landmarks[8]["x"], landmarks[8]["y"]]

    palm_length = distance(wrist, index_mcp)
    finger_length = distance(index_mcp, index_tip)

    if palm_length < 1e-6:
        extension_score = 0.0
    else:
        ratio = finger_length / palm_length
        extension_score = float(
            np.clip((ratio - 0.5) / 0.8, 0.0, 1.0)
        )

    score = (
        0.60 * index_score
        + 0.25 * curled_score
        + 0.15 * extension_score
    )

    return {
        "score": round(score, 3),
        "is_pointing": score >= THRESHOLD,
        "components": {
            "index_straightness": round(index_score, 3),
            "other_fingers_curled": round(curled_score, 3),
            "index_extension": round(extension_score, 3),
        },
    }


# --------------------------------------------------
# Find original image
# --------------------------------------------------

def find_image(image_name):
    """Find the corresponding original image by filename."""

    candidates = [
        IMAGE_DIR / image_name,
        IMAGE_DIR / Path(image_name).name,
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = list(IMAGE_DIR.rglob(Path(image_name).name))
    return matches[0] if matches else None


# --------------------------------------------------
# Visualization
# --------------------------------------------------

def draw_hand(image, hand, score, selected):
    points = hand.get("landmarks", [])

    if len(points) != 21:
        return

    # OpenCV uses BGR colors.
    color = (0, 200, 0) if selected else (0, 0, 255)

    by_id = {p["id"]: p for p in points}

    # Draw hand skeleton.
    for a, b in HAND_CONNECTIONS:
        if a in by_id and b in by_id:
            p1 = (
                round(by_id[a]["x"]),
                round(by_id[a]["y"]),
            )
            p2 = (
                round(by_id[b]["x"]),
                round(by_id[b]["y"]),
            )

            cv2.line(image, p1, p2, color, 3)

    # Draw landmarks.
    for point in points:
        x = round(point["x"])
        y = round(point["y"])
        cv2.circle(image, (x, y), 4, color, -1)

    # Add score near wrist.
    wrist = by_id.get(0)

    if wrist:
        x = round(wrist["x"])
        y = round(wrist["y"])

        label = (
            f"{hand['wrist_side']} "
            f"{'SELECTED' if selected else 'NOT SELECTED'} "
            f"score={score['score']:.2f}"
        )

        cv2.putText(
            image,
            label,
            (max(5, x - 100), max(25, y - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Fant ikke JSON-filen: {DATA_PATH.resolve()}"
        )

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Fant ikke bildemappen: {IMAGE_DIR}"
        )

    with DATA_PATH.open(encoding="utf-8") as f:
        images = json.load(f)

    summary = []

    total_images = 0
    images_with_hands = 0
    images_without_hands = 0
    images_with_selected_hand = 0
    images_without_selected_hand = 0

    total_hands_scored = 0
    total_hands_selected = 0

    for item in images:
        image_name = item.get("image", "unknown")
        hands_data = item.get("hands", {})

        total_images += 1
        scored = []

        # The JSON stores hands under "left" and "right".
        for side in ("left", "right"):
            hand_data = hands_data.get(side, {})
            landmarks = hand_data.get("final_landmarks")

            # None means that no final hand was detected.
            if not isinstance(landmarks, list) or len(landmarks) != 21:
                continue

            score = score_pointing_hand(landmarks)

            # Add landmark IDs for visualization.
            hand = {
                "wrist_side": side,
                "landmarks": [
                    {
                        "id": i,
                        "x": float(lm["x"]),
                        "y": float(lm["y"]),
                    }
                    for i, lm in enumerate(landmarks)
                ],
            }

            scored.append((side, hand, score))

        total_hands_scored += len(scored)

        if scored:
            images_with_hands += 1
        else:
            images_without_hands += 1
            print(f"{image_name}: ingen hender funnet")

        # Select the highest-scoring hand if it passes threshold.
        if scored:
            best = max(scored, key=lambda entry: entry[2]["score"])

            selected_side = (
                best[0]
                if best[2]["score"] >= THRESHOLD
                else None
            )
        else:
            selected_side = None

        if selected_side is not None:
            images_with_selected_hand += 1
            total_hands_selected += 1
        else:
            images_without_selected_hand += 1

        # Find and load original image.
        image_path = find_image(image_name)

        if image_path is None:
            print(f"Fant ikke originalbildet: {image_name}")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Kunne ikke lese bildet: {image_path}")
            continue

        # Draw each detected hand.
        for side, hand, score in scored:
            selected = side == selected_side

            draw_hand(
                image,
                hand,
                score,
                selected=selected,
            )

            status = "SELECTED" if selected else "not selected"

            print(
                f"{image_name} | {side} hand | "
                f"score={score['score']:.3f} | {status}"
            )

        # Add message if no hand passes the threshold.
        if selected_side is None:
            cv2.putText(
                image,
                "NO CONFIDENT POINTING HAND",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        output_path = OUTPUT_DIR / Path(image_name).name

        if not cv2.imwrite(str(output_path), image):
            print(f"Kunne ikke lagre visualisering: {output_path}")

        summary.append({
            "image": image_name,
            "selected_side": selected_side,
            "threshold": THRESHOLD,
            "scores": [
                {
                    "side": side,
                    **score,
                }
                for side, hand, score in scored
            ],
        })

    # Save score results.
    summary_path = OUTPUT_DIR / "scores.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary.
    print("\n" + "=" * 45)
    print("OPPSUMMERING")
    print("=" * 45)
    print(f"Bilder i JSON: {total_images}")
    print(f"Bilder med minst én hånd: {images_with_hands}")
    print(f"Bilder uten detekterte hender: {images_without_hands}")
    print(f"Bilder med valgt pekehånd: {images_with_selected_hand}")
    print(f"Bilder uten valgt pekehånd: {images_without_selected_hand}")
    print(f"Totalt antall hender scoret: {total_hands_scored}")
    print(f"Totalt antall valgte hender: {total_hands_selected}")
    print(f"\nVisualiseringer: {OUTPUT_DIR.resolve()}")
    print(f"Scorefil: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
