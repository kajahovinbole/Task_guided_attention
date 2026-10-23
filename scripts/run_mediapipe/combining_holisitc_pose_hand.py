
from pathlib import Path
import json

import cv2
import mediapipe as mp
import numpy as np

"""
Lilla:	        Pose-landemerker
Grønn:	        Holistic venstre hånd
Gul:	        Holistic høyre hånd
Blå/oransje:	Hand Landmarker fra hele bildet
Turkis:	        Hånd funnet via crop-fallback
Hvite sirkler:	Endelig valgt prediksjon
"""


# ============================================================
# Paths and configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent.parent

IMAGE_DIR = (
    ROOT / "data/3_exported_from_fiftyone/1_2_3_andrea_merged/data"
)

MODEL_DIR = ROOT / "models/mediapipe"
HOLISTIC_PATH = MODEL_DIR / "holistic_landmarker.task"
HAND_PATH = MODEL_DIR / "hand_landmarker.task"

OUTPUT_DIR = ROOT / "results/combined_holistic_pose_hand/tom_scene28"
VIS_DIR = OUTPUT_DIR / "visualizations"
CROP_DIR = OUTPUT_DIR / "fallback_crops"

for directory in (OUTPUT_DIR, VIS_DIR, CROP_DIR):
    directory.mkdir(parents=True, exist_ok=True)

JSON_PATH = OUTPUT_DIR / "ensemble_results.json"

CROP_SIZE = 500
AGREEMENT_THRESHOLD = 0.08  # Normalized image diagonal
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

for model_path in (HOLISTIC_PATH, HAND_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Fant ikke modellen: {model_path}")

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Fant ikke bildemappen: {IMAGE_DIR}")


# ============================================================
# MediaPipe options
# ============================================================

Vision = mp.tasks.vision
BaseOptions = mp.tasks.BaseOptions

holistic_options = Vision.HolisticLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HOLISTIC_PATH)
    ),
    running_mode=Vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
    min_pose_landmarks_confidence=0.5,
    min_hand_landmarks_confidence=0.5,
)

hand_options = Vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_PATH)
    ),
    running_mode=Vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)


# ============================================================
# Helpers
# ============================================================

POSE_INDEX = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}


def normalized_landmarks_to_pixels(landmarks, width, height):
    return [
        [
            int(lm.x * width),
            int(lm.y * height),
        ]
        for lm in landmarks
    ]


def hand_center(points):
    if not points:
        return None

    # Use the wrist landmark as the hand's reference point.
    return np.array(points[0], dtype=float)


def normalized_distance(point_a, point_b, width, height):
    if point_a is None or point_b is None:
        return float("inf")

    delta = np.array(point_a, dtype=float) - np.array(
        point_b, dtype=float
    )

    return float(np.linalg.norm(delta) / np.hypot(width, height))


def pose_point(pose, name, width, height):
    if not pose:
        return None

    lm = pose[POSE_INDEX[name]]

    visibility = getattr(lm, "visibility", None)
    if visibility is not None and visibility < 0.1:
        return None

    return [int(lm.x * width), int(lm.y * height)]


def estimate_crop_centers(pose, side, width, height):
    """Estimate fallback crop centers from the arm."""

    shoulder = pose_point(
        pose, f"{side}_shoulder", width, height
    )
    elbow = pose_point(
        pose, f"{side}_elbow", width, height
    )
    wrist = pose_point(
        pose, f"{side}_wrist", width, height
    )

    candidates = []

    if wrist is not None:
        candidates.append(("wrist", wrist))

    if elbow is not None and wrist is not None:
        direction = np.array(wrist) - np.array(elbow)
        length = np.linalg.norm(direction)

        if length > 1:
            center = np.array(wrist) + direction / length * length
            candidates.append(
                ("beyond_wrist", center.astype(int).tolist())
            )

    if wrist is None and shoulder is not None and elbow is not None:
        direction = np.array(elbow) - np.array(shoulder)
        length = np.linalg.norm(direction)

        if length > 1:
            center = np.array(elbow) + direction / length * length
            candidates.append(
                ("beyond_elbow", center.astype(int).tolist())
            )

    # Remove near-duplicate centers.
    unique = []
    for name, center in candidates:
        if all(
            np.linalg.norm(np.array(center) - np.array(existing[1])) > 30
            for existing in unique
        ):
            unique.append((name, center))

    return unique


def make_crop(image, center, crop_size):
    """Create a square crop and return its origin in original coordinates."""

    h, w = image.shape[:2]
    cx, cy = map(int, center)
    half = crop_size // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = image[y1:y2, x1:x2].copy()
    ch, cw = crop.shape[:2]

    top = (crop_size - ch) // 2
    bottom = crop_size - ch - top
    left = (crop_size - cw) // 2
    right = crop_size - cw - left

    crop = cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    # Original-image coordinate of the padded crop's top-left corner.
    origin = (x1 - left, y1 - top)

    return crop, origin


def hand_result_to_pixels(hand_landmarks, width, height, origin=(0, 0)):
    ox, oy = origin
    points = []

    for lm in hand_landmarks:
        points.append(
            [
                int(lm.x * width) + ox,
                int(lm.y * height) + oy,
            ]
        )

    return points


def draw_hand(image, points, color, thickness=2):
    if not points:
        return

    h, w = image.shape[:2]

    for x, y in points:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), 3, color, -1)

    for connection in Vision.HandLandmarksConnections.HAND_CONNECTIONS:
        start = connection.start
        end = connection.end

        if start >= len(points) or end >= len(points):
            continue

        p1 = tuple(points[start])
        p2 = tuple(points[end])

        if (
            0 <= p1[0] < w and 0 <= p1[1] < h
            and 0 <= p2[0] < w and 0 <= p2[1] < h
        ):
            cv2.line(image, p1, p2, color, thickness)


def draw_pose(image, pose):
    if not pose:
        return

    points = normalized_landmarks_to_pixels(
        pose, image.shape[1], image.shape[0]
    )

    for x, y in points:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 3, (255, 0, 255), -1)


def handedness_label(handedness_list, index):
    """Read the handedness label when the API provides one."""

    try:
        return handedness_list[index][0].category_name.lower()
    except (IndexError, TypeError, AttributeError):
        return "unknown"


def find_nearest_detection(target_points, detections, width, height):
    """Find the full-image detection closest to a target hand."""

    target_center = hand_center(target_points)

    if target_center is None or not detections:
        return None, float("inf")

    best_detection = None
    best_distance = float("inf")

    for detection in detections:
        center = hand_center(detection["landmarks"])
        distance_value = normalized_distance(
            target_center, center, width, height
        )

        if distance_value < best_distance:
            best_distance = distance_value
            best_detection = detection

    return best_detection, best_distance


def serialize_points(points):
    if points is None:
        return None

    return [{"x": int(x), "y": int(y)} for x, y in points]


# ============================================================
# Find images
# ============================================================

image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
)

if not image_paths:
    raise FileNotFoundError(f"Fant ingen bilder i {IMAGE_DIR}")


# ============================================================
# Run ensemble
# ============================================================

records = []

with Vision.HandLandmarker.create_from_options(hand_options) as hand_model:

    for image_number, path in enumerate(image_paths, start=1):
        original = cv2.imread(str(path))

        if original is None:
            print(f"Kunne ikke lese {path.name}")
            continue

        h, w = original.shape[:2]
        visualization = original.copy()

        rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # Recreate Holistic for each image to avoid the segmentation
        # smoothing dimension-mismatch issue between differently sized images.
        with Vision.HolisticLandmarker.create_from_options(
            holistic_options
        ) as holistic_model:
            holistic = holistic_model.detect(mp_image)

        pose = holistic.pose_landmarks
        draw_pose(visualization, pose)

        # --------------------------------------------
        # A. Holistic hand detections
        # --------------------------------------------

        holistic_hands = {}

        for side in ("left", "right"):
            landmarks = getattr(
                holistic, f"{side}_hand_landmarks", None
            )

            if landmarks:
                holistic_hands[side] = normalized_landmarks_to_pixels(
                    landmarks, w, h
                )
            else:
                holistic_hands[side] = None

        # --------------------------------------------
        # B. Hand Landmarker on full image
        # --------------------------------------------

        full_result = hand_model.detect(mp_image)

        direct_hands = []

        for index, landmarks in enumerate(full_result.hand_landmarks):
            points = normalized_landmarks_to_pixels(
                landmarks, w, h
            )

            label = handedness_label(full_result.handedness, index)

            direct_hands.append({
                "landmarks": points,
                "handedness": label,
                "source": "hand_landmarker_full_image",
            })

        # --------------------------------------------
        # C. Crop fallback when Holistic misses a hand
        # --------------------------------------------

        fallback_hands = {"left": None, "right": None}
        crop_metadata = {"left": [], "right": []}

        for side in ("left", "right"):
            if holistic_hands[side] is not None:
                continue

            for candidate_name, center in estimate_crop_centers(
                pose, side, w, h
            ):
                crop, origin = make_crop(
                    original, center, CROP_SIZE
                )

                if crop is None:
                    continue

                crop_name = (
                    f"{path.stem}_{side}_{candidate_name}.jpg"
                )
                cv2.imwrite(str(CROP_DIR / crop_name), crop)

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=crop_rgb,
                )

                crop_result = hand_model.detect(crop_image)

                crop_metadata[side].append({
                    "crop": crop_name,
                    "candidate": candidate_name,
                    "detected": bool(crop_result.hand_landmarks),
                })

                if not crop_result.hand_landmarks:
                    continue

                crop_h, crop_w = crop.shape[:2]

                crop_points = hand_result_to_pixels(
                    crop_result.hand_landmarks[0],
                    crop_w,
                    crop_h,
                    origin,
                )

                fallback_hands[side] = {
                    "landmarks": crop_points,
                    "source": f"hand_landmarker_crop:{candidate_name}",
                }

                break

        # --------------------------------------------
        # D. Compare and combine
        # --------------------------------------------

        combined = {}

        for side in ("left", "right"):
            holistic_points = holistic_hands[side]

            # The direct model may detect multiple hands.
            nearest_direct, disagreement = find_nearest_detection(
                holistic_points, direct_hands, w, h
            )

            if holistic_points is not None and nearest_direct is not None:
                is_agreement = disagreement <= AGREEMENT_THRESHOLD

                if is_agreement:
                    # Both models detected a nearby hand. Keep Holistic's
                    # 21 landmarks as the final set and preserve both raw sets.
                    final_points = holistic_points
                    decision = "agreement"
                    confidence_label = "higher_evidence"
                else:
                    # The detections differ substantially. Avoid averaging
                    # them blindly; keep the Holistic result and flag it.
                    final_points = holistic_points
                    decision = "disagreement"
                    confidence_label = "uncertain"

                direct_points = nearest_direct["landmarks"]

            elif holistic_points is not None:
                final_points = holistic_points
                direct_points = None
                decision = "holistic_only"
                confidence_label = "single_model"

            elif nearest_direct is not None:
                final_points = nearest_direct["landmarks"]
                direct_points = final_points
                decision = "hand_landmarker_only"
                confidence_label = "single_model"

            elif fallback_hands[side] is not None:
                final_points = fallback_hands[side]["landmarks"]
                direct_points = final_points
                decision = "crop_fallback"
                confidence_label = "single_model"

            else:
                final_points = None
                direct_points = None
                decision = "not_found"
                confidence_label = "none"

            combined[side] = {
                "final_landmarks": serialize_points(final_points),
                "holistic_landmarks": serialize_points(holistic_points),
                "hand_landmarker_landmarks": serialize_points(direct_points),
                "fallback_landmarks": serialize_points(
                    fallback_hands[side]["landmarks"]
                    if fallback_hands[side] else None
                ),
                "decision": decision,
                "confidence_label": confidence_label,
                "disagreement_normalized": (
                    disagreement
                    if holistic_points is not None and nearest_direct is not None
                    else None
                ),
                "crop_attempts": crop_metadata[side],
            }

            # ----------------------------------------
            # E. Visualization
            # ----------------------------------------

            if holistic_points is not None:
                # Holistic: green for left, yellow for right.
                color = (0, 255, 0) if side == "left" else (0, 255, 255)
                draw_hand(visualization, holistic_points, color)

            if direct_points is not None:
                # Direct Hand Landmarker: blue.
                draw_hand(visualization, direct_points, (255, 100, 0))

            if fallback_hands[side] is not None:
                # Crop fallback: cyan.
                draw_hand(
                    visualization,
                    fallback_hands[side]["landmarks"],
                    (255, 255, 0),
                )

            if final_points is not None:
                # Final selected landmarks: white circles.
                for x, y in final_points:
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(
                            visualization, (x, y), 5, (255, 255, 255), 1
                        )

            print(
                f"  {side}: {decision}, "
                f"disagreement={combined[side]['disagreement_normalized']}"
            )

        # --------------------------------------------
        # F. Save outputs
        # --------------------------------------------

        cv2.imwrite(str(VIS_DIR / path.name), visualization)

        records.append({
            "image": path.name,
            "width": w,
            "height": h,
            "pose_found": bool(pose),
            "hands": combined,
        })

        print(
            f"[{image_number}/{len(image_paths)}] {path.name}"
        )


with JSON_PATH.open("w", encoding="utf-8") as file:
    json.dump(records, file, indent=2, ensure_ascii=False)

print("\nFerdig.")
print(f"Visualiseringer: {VIS_DIR}")
print(f"Fallback-crops: {CROP_DIR}")
print(f"JSON-resultater: {JSON_PATH}")
