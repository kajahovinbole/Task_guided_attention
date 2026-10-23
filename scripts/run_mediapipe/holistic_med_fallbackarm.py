
from pathlib import Path
import json

import cv2
import mediapipe as mp


# ==================================================
# Paths and configuration
# ==================================================

ROOT = Path(__file__).resolve().parent.parent.parent

IMAGE_DIR = (
    ROOT / "data/3_exported_from_fiftyone/1_2_3_andrea_merged/data"
)

MODEL_DIR = ROOT / "models/mediapipe"
HOLISTIC_MODEL = MODEL_DIR / "holistic_landmarker.task"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"

OUTPUT_DIR = ROOT / "results/holistic_hand_fallback/tom_scene28"
VIS_DIR = OUTPUT_DIR / "visualizations"
CROP_DIR = OUTPUT_DIR / "fallback_crops"
JSON_PATH = OUTPUT_DIR / "landmarks.json"

VIS_DIR.mkdir(parents=True, exist_ok=True)
CROP_DIR.mkdir(parents=True, exist_ok=True)

# Crop size is relative to the estimated hand location.
CROP_SIZE = 400

# How far beyond the wrist to look, as a fraction of
# the shoulder-to-elbow / elbow-to-wrist segment length.
EXTRAPOLATION_FACTOR = 1.0

# Hand Landmarker confidence thresholds.
HAND_DETECTION_CONFIDENCE = 0.3
HAND_PRESENCE_CONFIDENCE = 0.3
HAND_TRACKING_CONFIDENCE = 0.3

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

for model_path in (HOLISTIC_MODEL, HAND_MODEL):
    if not model_path.exists():
        raise FileNotFoundError(f"Fant ikke modellen: {model_path}")

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Fant ikke bildemappen: {IMAGE_DIR}")


# ==================================================
# MediaPipe setup
# ==================================================

mp_tasks = mp.tasks
Vision = mp_tasks.vision
BaseOptions = mp_tasks.BaseOptions

holistic_options = Vision.HolisticLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HOLISTIC_MODEL)
    ),
    running_mode=Vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
    min_pose_landmarks_confidence=0.5,
    min_hand_landmarks_confidence=0.5,
)

hand_options = Vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL)
    ),
    running_mode=Vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
    min_hand_presence_confidence=HAND_PRESENCE_CONFIDENCE,
    min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
)


# ==================================================
# Landmark helpers
# ==================================================

# MediaPipe Pose landmark indices.
POSE = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}


def landmark_xy(landmarks, index, width, height):
    """Convert a normalized pose landmark to pixel coordinates."""

    if not landmarks or index >= len(landmarks):
        return None

    lm = landmarks[index]

    # Visibility is not guaranteed to exist in every result type.
    visibility = getattr(lm, "visibility", None)
    if visibility is not None and visibility < 0.1:
        return None

    x = int(lm.x * width)
    y = int(lm.y * height)

    # Allow a small margin outside the image.
    if not (-width <= x <= 2 * width and -height <= y <= 2 * height):
        return None

    return (x, y)


def distance(a, b):
    if a is None or b is None:
        return 0.0
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def estimate_hand_candidates(
    pose_landmarks,
    side,
    width,
    height,
):
    """
    Estimate possible hand locations from the arm.

    Returns candidate centers in original-image pixel coordinates.
    Candidates are tried in order.
    """

    shoulder = landmark_xy(
        pose_landmarks, POSE[f"{side}_shoulder"], width, height
    )
    elbow = landmark_xy(
        pose_landmarks, POSE[f"{side}_elbow"], width, height
    )
    wrist = landmark_xy(
        pose_landmarks, POSE[f"{side}_wrist"], width, height
    )

    candidates = []

    # Best candidate: directly around the wrist.
    if wrist is not None:
        candidates.append(("wrist", wrist))

    # Estimate where the hand may be if the wrist is missing,
    # or try an additional candidate beyond the wrist.
    if elbow is not None and wrist is not None:
        dx = wrist[0] - elbow[0]
        dy = wrist[1] - elbow[1]
        length = distance(elbow, wrist)

        if length > 1:
            estimated = (
                int(wrist[0] + dx / length * length * EXTRAPOLATION_FACTOR),
                int(wrist[1] + dy / length * length * EXTRAPOLATION_FACTOR),
            )
            candidates.append(("beyond_wrist", estimated))

    # If the wrist is missing, extrapolate from shoulder to elbow.
    if wrist is None and shoulder is not None and elbow is not None:
        dx = elbow[0] - shoulder[0]
        dy = elbow[1] - shoulder[1]
        length = distance(shoulder, elbow)

        if length > 1:
            estimated = (
                int(elbow[0] + dx / length * length),
                int(elbow[1] + dy / length * length),
            )
            candidates.append(("beyond_elbow", estimated))

    # Avoid duplicate candidate centers.
    unique = []
    seen = set()

    for label, center in candidates:
        key = (center[0] // 20, center[1] // 20)
        if key not in seen:
            unique.append((label, center))
            seen.add(key)

    return unique


def crop_around(image, center, crop_size):
    """Return a square crop and its top-left origin in the image."""

    height, width = image.shape[:2]
    half = crop_size // 2

    cx, cy = center

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(width, cx + half)
    y2 = min(height, cy + half)

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = image[y1:y2, x1:x2].copy()

    # Pad edge crops to a square, keeping the image content intact.
    ch, cw = crop.shape[:2]
    side_len = max(ch, cw)

    top = (side_len - ch) // 2
    bottom = side_len - ch - top
    left = (side_len - cw) // 2
    right = side_len - cw - left

    crop = cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    # Offset maps padded crop coordinates back to original image.
    origin_x = x1 - left
    origin_y = y1 - top

    return crop, (origin_x, origin_y)


def hand_landmarks_to_image(hand_landmarks, origin, crop_size):
    """Convert normalized crop landmarks to image pixel coordinates."""

    ox, oy = origin
    points = []

    for lm in hand_landmarks:
        points.append(
            (
                int(lm.x * crop_size) + ox,
                int(lm.y * crop_size) + oy,
            )
        )

    return points


# ==================================================
# Drawing
# ==================================================

def draw_points(image, points, color, connections=None):
    height, width = image.shape[:2]

    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (x, y), 3, color, -1)

    if connections is not None:
        for connection in connections:
            start = connection.start
            end = connection.end

            if start >= len(points) or end >= len(points):
                continue

            p1 = points[start]
            p2 = points[end]

            if (
                0 <= p1[0] < width
                and 0 <= p1[1] < height
                and 0 <= p2[0] < width
                and 0 <= p2[1] < height
            ):
                cv2.line(image, p1, p2, color, 2)


def draw_pose(image, pose_landmarks):
    if not pose_landmarks:
        return

    points = [
        (
            int(lm.x * image.shape[1]),
            int(lm.y * image.shape[0]),
        )
        for lm in pose_landmarks
    ]

    connections_enum = getattr(Vision, "PoseLandmarksConnections", None)

    if connections_enum is not None:
        connections = connections_enum.POSE_LANDMARKS
    else:
        connections = None

    draw_points(image, points, (255, 0, 255), connections)


def draw_hand(image, points, color):
    draw_points(
        image,
        points,
        color,
        Vision.HandLandmarksConnections.HAND_CONNECTIONS,
    )


# ==================================================
# Find images
# ==================================================

image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
)

if not image_paths:
    raise FileNotFoundError(f"Fant ingen bilder i {IMAGE_DIR}")


# ==================================================
# Run detection
# ==================================================

all_results = []

# Hand Landmarker has no segmentation-smoothing state.
# Holistic is recreated for each image to avoid the
# dimension-mismatch error seen when image sizes vary.
with Vision.HandLandmarker.create_from_options(hand_options) as hand_model:

    for image_index, path in enumerate(image_paths, start=1):
        original = cv2.imread(str(path))

        if original is None:
            print(f"Kunne ikke lese: {path.name}")
            continue

        image = original.copy()
        height, width = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # Run Holistic independently on this image.
        with Vision.HolisticLandmarker.create_from_options(
            holistic_options
        ) as holistic_model:
            holistic_result = holistic_model.detect(mp_image)

        pose = holistic_result.pose_landmarks

        draw_pose(image, pose)

        record = {
            "image": path.name,
            "width": width,
            "height": height,
            "pose_found": bool(pose),
            "hands": {},
        }

        # Keep Holistic detections where available.
        for side, color, holistic_hand in (
            ("left", (0, 255, 0), holistic_result.left_hand_landmarks),
            ("right", (0, 255, 255), holistic_result.right_hand_landmarks),
        ):
            if holistic_hand:
                points = [
                    (int(lm.x * width), int(lm.y * height))
                    for lm in holistic_hand
                ]

                draw_hand(image, points, color)

                record["hands"][side] = {
                    "found": True,
                    "source": "holistic",
                    "landmarks": [
                        {"x": x, "y": y} for x, y in points
                    ],
                }

                continue

            # Holistic missed this hand: try fallback crops.
            candidates = estimate_hand_candidates(
                pose, side, width, height
            )

            fallback_found = False

            for candidate_index, (candidate_name, center) in enumerate(
                candidates
            ):
                crop, origin = crop_around(image, center, CROP_SIZE)

                if crop is None:
                    continue

                crop_filename = (
                    f"{path.stem}_{side}_{candidate_name}_{candidate_index}.jpg"
                )
                crop_path = CROP_DIR / crop_filename
                cv2.imwrite(str(crop_path), crop)

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=crop_rgb,
                )

                hand_result = hand_model.detect(crop_mp_image)

                if not hand_result.hand_landmarks:
                    continue

                # Use the first hand returned for this candidate crop.
                crop_landmarks = hand_result.hand_landmarks[0]

                # Coordinates must use the actual padded crop size.
                crop_side = crop.shape[0]
                points = hand_landmarks_to_image(
                    crop_landmarks, origin, crop_side
                )

                draw_hand(image, points, (255, 255, 0))

                # Mark the candidate center used by the fallback.
                cv2.circle(image, center, 8, (0, 0, 255), 2)

                record["hands"][side] = {
                    "found": True,
                    "source": "hand_landmarker_fallback",
                    "candidate": candidate_name,
                    "crop": crop_filename,
                    "landmarks": [
                        {"x": x, "y": y} for x, y in points
                    ],
                }

                fallback_found = True
                break

            if not fallback_found:
                record["hands"][side] = {
                    "found": False,
                    "source": None,
                    "landmarks": [],
                    "candidates_tried": [
                        name for name, _ in candidates
                    ],
                }

        output_path = VIS_DIR / path.name
        cv2.imwrite(str(output_path), image)
        all_results.append(record)

        left = record["hands"].get("left", {}).get("source")
        right = record["hands"].get("right", {}).get("source")

        print(
            f"[{image_index}/{len(image_paths)}] {path.name}: "
            f"pose={bool(pose)}, left={left or 'not found'}, "
            f"right={right or 'not found'}"
        )


# ==================================================
# Save JSON
# ==================================================

with JSON_PATH.open("w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print("\nFerdig.")
print(f"Visualiseringer: {VIS_DIR}")
print(f"Fallback-crops: {CROP_DIR}")
print(f"Landemerker JSON: {JSON_PATH}")
