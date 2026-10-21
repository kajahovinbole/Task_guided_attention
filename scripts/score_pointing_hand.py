
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

DATA_PATH = Path("data/results/mediapipe_wrist_hand/hand_landmarks.json")
IMAGE_DIR = Path("data/exported/merged/data/")
OUTPUT_DIR = Path("data/results/pointing_hand_scores/")

THRESHOLD = 0.6

FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def finger_straightness(landmarks, finger_ids):
    points = [
        np.array([landmarks[i]["x"], landmarks[i]["y"]], dtype=float)
        for i in finger_ids
    ]

    path_length = sum(
        distance(points[i], points[i + 1])
        for i in range(len(points) - 1)
    )
    direct_distance = distance(points[0], points[-1])

    if path_length < 1e-6:
        return 0.0

    return float(np.clip(direct_distance / path_length, 0.0, 1.0))


def score_pointing_hand(hand):
    landmark_list = hand.get("landmarks", [])

    if len(landmark_list) != 21:
        return {"score": 0.0, "is_pointing": False}

    landmarks = {lm["id"]: lm for lm in landmark_list}

    if not all(i in landmarks for i in range(21)):
        return {"score": 0.0, "is_pointing": False}

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


def find_image(image_name):
    """Find the corresponding original image by filename."""
    candidates = [
        IMAGE_DIR / image_name,
        IMAGE_DIR / Path(image_name).name,
    ]

    for path in candidates:
        if path.exists():
            return path

    # Fallback if the JSON stores a relative path from another folder.
    matches = list(IMAGE_DIR.rglob(Path(image_name).name))
    return matches[0] if matches else None


def draw_hand(image, hand, score, selected):
    # Landmarks use original-image pixel coordinates.
    points = hand.get("landmarks", [])
    if not points:
        return

    color = (0, 200, 0) if selected else (0, 0, 255)

    # Draw hand skeleton.
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]

    by_id = {p["id"]: p for p in points}

    for a, b in connections:
        if a in by_id and b in by_id:
            p1 = (round(by_id[a]["x"]), round(by_id[a]["y"]))
            p2 = (round(by_id[b]["x"]), round(by_id[b]["y"]))
            cv2.line(image, p1, p2, color, 3)

    for point in points:
        x, y = round(point["x"]), round(point["y"])
        cv2.circle(image, (x, y), 4, color, -1)

    wrist = by_id.get(0)
    if wrist:
        x, y = round(wrist["x"]), round(wrist["y"])
        label = (
            f"{'SELECTED' if selected else 'NOT SELECTED'} "
            f"score={score['score']:.2f}"
        )

        # Keep text inside the image when possible.
        y_text = max(25, y - 20)
        cv2.putText(
            image, label, (max(5, x - 100), y_text),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
            cv2.LINE_AA,
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with DATA_PATH.open() as f:
        images = json.load(f)

    summary = []

    for item in images:
        hands = item.get("hands", [])
        if not hands:
            continue

        scored = [
            (i, hand, score_pointing_hand(hand))
            for i, hand in enumerate(hands)
        ]

        # Only select a hand if the highest score passes the threshold.
        best = max(scored, key=lambda entry: entry[2]["score"])
        selected_index = (
            best[0] if best[2]["score"] >= THRESHOLD else None
        )

        image_path = find_image(item["image"])
        if image_path is None:
            print(f"Image not found: {item['image']}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        for i, hand, score in scored:
            draw_hand(
                image,
                hand,
                score,
                selected=(i == selected_index),
            )

            print(
                f"{item['image']} | hand {i} "
                f"({hand.get('wrist_side', '?')}) | "
                f"score={score['score']:.3f} | "
                f"{'SELECTED' if i == selected_index else 'not selected'}"
            )

        if selected_index is None:
            cv2.putText(
                image, "NO CONFIDENT POINTING HAND",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2, cv2.LINE_AA,
            )

        output_path = OUTPUT_DIR / Path(item["image"]).name
        cv2.imwrite(str(output_path), image)

        summary.append({
            "image": item["image"],
            "selected_hand_index": selected_index,
            "scores": [
                {
                    "hand_index": i,
                    "wrist_side": hand.get("wrist_side"),
                    **score,
                }
                for i, hand, score in scored
            ],
        })

    with (OUTPUT_DIR / "scores.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nVisualizations saved to: {OUTPUT_DIR}")
    print(f"Summary saved to: {OUTPUT_DIR / 'scores.json'}")


if __name__ == "__main__":
    main()
