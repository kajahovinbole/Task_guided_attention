
from pathlib import Path
import json
import math

import cv2
import mediapipe as mp


# --------------------------------------------------
# Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent

IMAGE_DIR = ROOT / "data/3_exported_from_fiftyone/1_2_3_andrea_merged/data"

MODEL_DIR = ROOT / "models" / "mediapipe"
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"

OUTPUT_DIR = ROOT / "results/wrist_detect&handlandmarks/tom_scene28_3"
CROP_DIR = OUTPUT_DIR / "crops"
VIS_DIR = OUTPUT_DIR / "visualizations"
JSON_PATH = OUTPUT_DIR / "hand_landmarks.json"

CROP_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

MIN_WRIST_VISIBILITY = 0.3
MIN_ARM_VISIBILITY = 0.1
MAX_IMAGE_WIDTH = 2560

CROP_SCALE = 3.0
MIN_CROP_SIZE = 200
MAX_CROP_SIZE = 800

ARROW_EXTENSION = 350

# Candidates from different detection methods are merged
# if their landmark bounding boxes overlap sufficiently.
DUPLICATE_IOU_THRESHOLD = 0.25
DUPLICATE_CENTER_THRESHOLD = 0.35


# --------------------------------------------------
# Check models
# --------------------------------------------------

for model_path in (POSE_MODEL, HAND_MODEL):
    if not model_path.is_file():
        raise FileNotFoundError(f"Fant ikke modellen: {model_path}")


# --------------------------------------------------
# Initialize MediaPipe
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(POSE_MODEL)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=5,
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL)
    ),
    running_mode=VisionRunningMode.IMAGE,

    # Allow multiple hands when searching the whole image.
    num_hands=10,

    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(
    pose_options
)

hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(
    hand_options
)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def get_visibility(landmark):
    return float(getattr(landmark, "visibility", 1.0))


def resize_for_detection(image):
    """Resize while preserving aspect ratio."""
    if MAX_IMAGE_WIDTH is None:
        return image, 1.0

    height, width = image.shape[:2]

    if width <= MAX_IMAGE_WIDTH:
        return image, 1.0

    scale = MAX_IMAGE_WIDTH / width

    resized = cv2.resize(
        image,
        (
            int(round(width * scale)),
            int(round(height * scale)),
        ),
    )

    return resized, scale


def calculate_crop_size(
    shoulder,
    elbow,
    image_width,
    image_height,
    scale_factor=1.0,
):
    """Estimate a crop size from upper-arm length."""
    dx = (shoulder.x - elbow.x) * image_width
    dy = (shoulder.y - elbow.y) * image_height

    upper_arm_length = math.hypot(dx, dy)
    upper_arm_original = upper_arm_length / scale_factor

    crop_size = int(round(upper_arm_original * CROP_SCALE))

    return max(
        MIN_CROP_SIZE,
        min(crop_size, MAX_CROP_SIZE),
    )


def create_square_crop(image, center_x, center_y, size):
    """Create a square crop centered on a point."""
    image_h, image_w = image.shape[:2]

    crop_w = min(int(size), image_w)
    crop_h = min(int(size), image_h)

    x1 = int(round(center_x - crop_w / 2))
    y1 = int(round(center_y - crop_h / 2))

    x1 = max(0, min(x1, image_w - crop_w))
    y1 = max(0, min(y1, image_h - crop_h))

    crop = image[
        y1:y1 + crop_h,
        x1:x1 + crop_w,
    ].copy()

    return crop, x1, y1


def make_mp_image(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )


def landmarks_to_pixels(
    hand_landmarks,
    offset_x,
    offset_y,
    crop_w,
    crop_h,
):
    """Convert normalized landmarks to original-image pixels."""
    points = []

    for landmark in hand_landmarks:
        x = offset_x + landmark.x * crop_w
        y = offset_y + landmark.y * crop_h

        points.append((float(x), float(y)))

    return points


def get_bbox(points):
    """Return x, y, width, height for a set of points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    return (
        x1,
        y1,
        max(1.0, x2 - x1),
        max(1.0, y2 - y1),
    )


def bbox_iou(box1, box2):
    """Intersection over union for two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    intersection = max(0.0, ix2 - ix1) * max(
        0.0, iy2 - iy1
    )

    area1 = w1 * h1
    area2 = w2 * h2

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def is_duplicate(candidate, accepted):
    """Check whether two candidates likely represent the same hand."""
    box1 = candidate["bbox"]
    box2 = accepted["bbox"]

    if bbox_iou(box1, box2) >= DUPLICATE_IOU_THRESHOLD:
        return True

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    center1 = (x1 + w1 / 2, y1 + h1 / 2)
    center2 = (x2 + w2 / 2, y2 + h2 / 2)

    distance = math.hypot(
        center1[0] - center2[0],
        center1[1] - center2[1],
    )

    scale = max(
        math.hypot(w1, h1),
        math.hypot(w2, h2),
    )

    return (
        distance < DUPLICATE_CENTER_THRESHOLD * scale
    )


def add_candidates_from_result(
    result,
    offset_x,
    offset_y,
    input_w,
    input_h,
    source,
    crop_box,
    candidates,
):
    """Convert MediaPipe hand results into candidate records."""
    if not result.hand_landmarks:
        return

    for hand_landmarks in result.hand_landmarks:
        points = landmarks_to_pixels(
            hand_landmarks,
            offset_x,
            offset_y,
            input_w,
            input_h,
        )

        bbox = get_bbox(points)

        candidates.append({
            "points": points,
            "landmarks": hand_landmarks,
            "bbox": bbox,
            "source": source,
            "crop_box": crop_box,
            # Prefer the crop-based result if two detections
            # appear to describe the same hand.
            "priority": 2 if source == "wrist_crop" else 1,
        })


def deduplicate_candidates(candidates):
    """Keep one candidate for each likely unique hand."""
    # Prefer detections from the more zoomed-in wrist crops.
    candidates = sorted(
        candidates,
        key=lambda c: (
            c["priority"],
            c["bbox"][2] * c["bbox"][3],
        ),
        reverse=True,
    )

    accepted = []

    for candidate in candidates:
        duplicate_index = None

        for i, previous in enumerate(accepted):
            if is_duplicate(candidate, previous):
                duplicate_index = i
                break

        if duplicate_index is None:
            accepted.append(candidate)

    return accepted


def draw_candidate(image, candidate, hand_number):
    """Draw hand landmarks, skeleton, bounding box and direction."""
    points = candidate["points"]

    integer_points = [
        (int(round(x)), int(round(y)))
        for x, y in points
    ]

    # Draw hand skeleton.
    for connection in (
        mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    ):
        start_idx = connection.start
        end_idx = connection.end

        cv2.line(
            image,
            integer_points[start_idx],
            integer_points[end_idx],
            (255, 180, 0),
            2,
        )

    for point in integer_points:
        cv2.circle(image, point, 3, (0, 255, 0), -1)

    x, y, w, h = candidate["bbox"]

    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(image.shape[1] - 1, int(round(x + w)))
    y2 = min(image.shape[0] - 1, int(round(y + h)))

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (255, 0, 255),
        2,
    )

    # Direction from index-finger base (5) to fingertip (8).
    base_x, base_y = points[5]
    tip_x, tip_y = points[8]

    dx = tip_x - base_x
    dy = tip_y - base_y
    length = math.hypot(dx, dy)

    if length > 1:
        dx /= length
        dy /= length

        end_x = int(round(tip_x + dx * ARROW_EXTENSION))
        end_y = int(round(tip_y + dy * ARROW_EXTENSION))

        end_x = max(0, min(image.shape[1] - 1, end_x))
        end_y = max(0, min(image.shape[0] - 1, end_y))

        cv2.arrowedLine(
            image,
            (int(round(base_x)), int(round(base_y))),
            (end_x, end_y),
            (0, 0, 255),
            4,
            tipLength=0.04,
        )

    cv2.putText(
        image,
        f"hand {hand_number}: {candidate['source']}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        2,
    )


def make_hand_json(candidate, image_w, image_h, hand_number):
    points = candidate["points"]

    landmarks_json = []

    for landmark_id, (x, y) in enumerate(points):
        original_landmark = candidate["landmarks"][landmark_id]

        landmarks_json.append({
            "id": landmark_id,
            "x": round(x, 3),
            "y": round(y, 3),
            "x_norm": round(x / image_w, 6),
            "y_norm": round(y / image_h, 6),
            "z": round(float(original_landmark.z), 6),
        })

    base_x, base_y = points[5]
    tip_x, tip_y = points[8]

    dx = tip_x - base_x
    dy = tip_y - base_y
    length = math.hypot(dx, dy)

    if length > 0:
        direction = {
            "dx": round(dx / length, 6),
            "dy": round(dy / length, 6),
        }
    else:
        direction = {"dx": 0.0, "dy": 0.0}

    crop_x, crop_y, crop_w, crop_h = candidate["crop_box"]

    return {
        "hand_id": hand_number,
        "detection_source": candidate["source"],
        "bbox": {
            "x": round(candidate["bbox"][0], 3),
            "y": round(candidate["bbox"][1], 3),
            "width": round(candidate["bbox"][2], 3),
            "height": round(candidate["bbox"][3], 3),
        },
        "crop": {
            "x": crop_x,
            "y": crop_y,
            "width": crop_w,
            "height": crop_h,
        },
        "landmarks": landmarks_json,
        "pointing_direction": {
            "start": {
                "x": round(base_x, 3),
                "y": round(base_y, 3),
                "x_norm": round(base_x / image_w, 6),
                "y_norm": round(base_y / image_h, 6),
            },
            "tip": {
                "x": round(tip_x, 3),
                "y": round(tip_y, 3),
                "x_norm": round(tip_x / image_w, 6),
                "y_norm": round(tip_y / image_h, 6),
            },
            "direction_2d": direction,
        },
    }


# --------------------------------------------------
# Process images
# --------------------------------------------------

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(
        f"Fant ikke bildemappen: {IMAGE_DIR}"
    )

image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not image_paths:
    raise RuntimeError(f"Fant ingen bilder i {IMAGE_DIR}")

json_results = []
total_hands = 0
total_wrist_candidates = 0
total_direct_candidates = 0
images_processed = 0

try:
    for image_path in image_paths:
        original = cv2.imread(str(image_path))

        if original is None:
            print(f"Kunne ikke lese bildet: {image_path.name}")
            continue

        images_processed += 1

        original_h, original_w = original.shape[:2]
        visualization = original.copy()

        image_json = {
            "image": image_path.name,
            "image_width": original_w,
            "image_height": original_h,
            "hands": [],
        }

        candidates = []

        # Resize for pose estimation and direct hand detection.
        detection_image, scale = resize_for_detection(original)

        detection_h, detection_w = detection_image.shape[:2]

        # --------------------------------------------------
        # Method 1: Detect hands directly in the whole image.
        # This can find hands even if the wrist is not detected.
        # --------------------------------------------------

        full_image_result = hand_detector.detect(
            make_mp_image(detection_image)
        )

        before = len(candidates)

        add_candidates_from_result(
            result=full_image_result,
            offset_x=0,
            offset_y=0,
            input_w=detection_w / scale,
            input_h=detection_h / scale,
            source="full_image",
            crop_box=(0, 0, original_w, original_h),
            candidates=candidates,
        )

        total_direct_candidates += len(candidates) - before

        # --------------------------------------------------
        # Method 2: Use Pose Landmarker to propose wrist crops.
        # --------------------------------------------------

        pose_result = pose_detector.detect(
            make_mp_image(detection_image)
        )

        image_wrist_candidates = 0

        if pose_result.pose_landmarks:
            for person_idx, pose_landmarks in enumerate(
                pose_result.pose_landmarks
            ):
                # MediaPipe pose indices:
                # left shoulder/elbow/wrist: 11/13/15
                # right shoulder/elbow/wrist: 12/14/16
                arms = [
                    ("left", 11, 13, 15),
                    ("right", 12, 14, 16),
                ]

                for side, shoulder_idx, elbow_idx, wrist_idx in arms:
                    shoulder = pose_landmarks[shoulder_idx]
                    elbow = pose_landmarks[elbow_idx]
                    wrist = pose_landmarks[wrist_idx]

                    if (
                        get_visibility(wrist) < MIN_WRIST_VISIBILITY
                        or get_visibility(elbow) < MIN_ARM_VISIBILITY
                        or get_visibility(shoulder) < MIN_ARM_VISIBILITY
                    ):
                        continue

                    wrist_x = int(round(wrist.x * detection_w / scale))
                    wrist_y = int(round(wrist.y * detection_h / scale))

                    if not (
                        0 <= wrist_x < original_w
                        and 0 <= wrist_y < original_h
                    ):
                        continue

                    crop_size = calculate_crop_size(
                        shoulder,
                        elbow,
                        detection_w,
                        detection_h,
                        scale_factor=scale,
                    )

                    crop, crop_x, crop_y = create_square_crop(
                        original,
                        wrist_x,
                        wrist_y,
                        crop_size,
                    )

                    if crop.size == 0:
                        continue

                    crop_h, crop_w = crop.shape[:2]

                    crop_name = (
                        f"{image_path.stem}_person{person_idx}_"
                        f"{side}_wrist.jpg"
                    )

                    cv2.imwrite(str(CROP_DIR / crop_name), crop)

                    hand_result = hand_detector.detect(
                        make_mp_image(crop)
                    )

                    # Visualize the wrist crop and wrist point.
                    cv2.rectangle(
                        visualization,
                        (crop_x, crop_y),
                        (
                            crop_x + crop_w - 1,
                            crop_y + crop_h - 1,
                        ),
                        (255, 0, 255),
                        2,
                    )

                    cv2.circle(
                        visualization,
                        (wrist_x, wrist_y),
                        5,
                        (0, 0, 255),
                        -1,
                    )

                    before = len(candidates)

                    add_candidates_from_result(
                        result=hand_result,
                        offset_x=crop_x,
                        offset_y=crop_y,
                        input_w=crop_w,
                        input_h=crop_h,
                        source="wrist_crop",
                        crop_box=(
                            crop_x,
                            crop_y,
                            crop_w,
                            crop_h,
                        ),
                        candidates=candidates,
                    )

                    image_wrist_candidates += len(candidates) - before

        total_wrist_candidates += image_wrist_candidates

        # --------------------------------------------------
        # Merge detections of the same hand.
        # Do not discard candidates merely because their
        # crop rectangles overlap.
        # --------------------------------------------------

        accepted_hands = deduplicate_candidates(candidates)

        for hand_number, candidate in enumerate(
            accepted_hands, start=1
        ):
            draw_candidate(
                visualization,
                candidate,
                hand_number,
            )

            hand_json = make_hand_json(
                candidate,
                original_w,
                original_h,
                hand_number,
            )

            image_json["hands"].append(hand_json)
            total_hands += 1

        json_results.append(image_json)

        cv2.imwrite(
            str(VIS_DIR / image_path.name),
            visualization,
        )

        print(
            f"{image_path.name}: "
            f"{len(candidates)} kandidater, "
            f"{len(accepted_hands)} unike hender "
            f"({image_wrist_candidates} fra wrist-crops)"
        )

finally:
    pose_detector.close()
    hand_detector.close()


# --------------------------------------------------
# Save results
# --------------------------------------------------

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        json_results,
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\nFerdig.")
print(f"Bilder behandlet: {images_processed}")
print(f"Kandidater fra fullbilder: {total_direct_candidates}")
print(f"Kandidater fra wrist-crops: {total_wrist_candidates}")
print(f"Hender etter deduplisering: {total_hands}")
print(f"JSON: {JSON_PATH}")
print(f"Crops: {CROP_DIR}")
print(f"Visualiseringer: {VIS_DIR}")
