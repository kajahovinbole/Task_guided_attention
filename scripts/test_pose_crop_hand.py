
from pathlib import Path
import json
import math

import cv2
import mediapipe as mp


# --------------------------------------------------
# Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "data/exported/1_2_3_andrea_merged/data"

MODEL_DIR = ROOT / "models" / "mediapipe"
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"

OUTPUT_DIR = ROOT / "data" / "results" / "wrist_crops/1_2_3_andrea_merged"
CROP_DIR = OUTPUT_DIR / "crops"
VIS_DIR = OUTPUT_DIR / "visualizations"
JSON_PATH = OUTPUT_DIR / "hand_landmarks.json"

CROP_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

MIN_WRIST_VISIBILITY = 0.3
MIN_ARM_VISIBILITY = 0.3
MAX_IMAGE_WIDTH = 1280

# Crop size = upper-arm length * this factor.
CROP_SCALE = 2.5
MIN_CROP_SIZE = 160
MAX_CROP_SIZE = 800

# Extension beyond the fingertip, in original-image pixels.
ARROW_EXTENSION = 350


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
    base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=5,
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
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


def resize_for_pose(image):
    """Resize for pose detection while preserving aspect ratio."""
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
    """Calculate square crop size from upper-arm length."""
    dx = (shoulder.x - elbow.x) * image_width
    dy = (shoulder.y - elbow.y) * image_height

    upper_arm_length = math.hypot(dx, dy)
    upper_arm_original = upper_arm_length / scale_factor

    crop_size = int(round(upper_arm_original * CROP_SCALE))
    return max(MIN_CROP_SIZE, min(crop_size, MAX_CROP_SIZE))


def create_wrist_crop(image, wrist_x, wrist_y, crop_size):
    """Create a square crop around the wrist, inside image boundaries."""
    image_h, image_w = image.shape[:2]

    crop_w = min(int(crop_size), image_w)
    crop_h = min(int(crop_size), image_h)

    center_x = int(round(wrist_x))
    center_y = int(round(wrist_y))

    x1 = center_x - crop_w // 2
    y1 = center_y - crop_h // 2

    x1 = max(0, min(x1, image_w - crop_w))
    y1 = max(0, min(y1, image_h - crop_h))

    crop = image[y1:y1 + crop_h, x1:x1 + crop_w].copy()
    return crop, x1, y1


def landmark_to_original(landmark, crop_x, crop_y, crop_w, crop_h):
    """Convert normalized crop coordinates to original-image pixels."""
    x = crop_x + landmark.x * crop_w
    y = crop_y + landmark.y * crop_h
    return x, y


def draw_hand_landmarks_on_original(
    image, hand_landmarks, crop_x, crop_y, crop_w, crop_h
):
    """Draw hand landmarks and skeleton on original image."""
    points = []

    for landmark in hand_landmarks:
        x, y = landmark_to_original(
            landmark, crop_x, crop_y, crop_w, crop_h
        )
        point = (int(round(x)), int(round(y)))
        points.append(point)
        cv2.circle(image, point, 3, (0, 255, 0), -1)

    for connection in (
        mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    ):
        start_idx = connection.start
        end_idx = connection.end

        if start_idx < len(points) and end_idx < len(points):
            cv2.line(
                image,
                points[start_idx],
                points[end_idx],
                (255, 180, 0),
                2,
            )

    return points


def draw_pointing_direction(
    image, hand_landmarks, crop_x, crop_y, crop_w, crop_h
):
    """
    Draw arrow from index-finger base (5) through fingertip (8).
    Return the base and fingertip coordinates, without arrow extension.
    """
    base = hand_landmarks[5]
    tip = hand_landmarks[8]

    start = landmark_to_original(
        base, crop_x, crop_y, crop_w, crop_h
    )
    tip_point = landmark_to_original(
        tip, crop_x, crop_y, crop_w, crop_h
    )

    dx = tip_point[0] - start[0]
    dy = tip_point[1] - start[1]
    length = math.hypot(dx, dy)

    if length < 1:
        return start, tip_point

    dx /= length
    dy /= length

    end = (
        int(round(tip_point[0] + dx * ARROW_EXTENSION)),
        int(round(tip_point[1] + dy * ARROW_EXTENSION)),
    )

    height, width = image.shape[:2]
    end = (
        max(0, min(width - 1, end[0])),
        max(0, min(height - 1, end[1])),
    )

    cv2.arrowedLine(
        image,
        (int(round(start[0])), int(round(start[1]))),
        end,
        (0, 0, 255),
        thickness=5,
        tipLength=0.04,
    )

    # Return the actual finger base and tip, not the extended arrow endpoint.
    return start, tip_point


def make_hand_json(
    hand_landmarks,
    image_w,
    image_h,
    crop_x,
    crop_y,
    crop_w,
    crop_h,
    person_idx,
    wrist_side,
    hand_idx,
    pointing_start,
    pointing_tip,
):
    """Create JSON-ready hand data in original-image coordinates."""
    landmarks_json = []

    for landmark_id, landmark in enumerate(hand_landmarks):
        x_px, y_px = landmark_to_original(
            landmark, crop_x, crop_y, crop_w, crop_h
        )

        landmarks_json.append({
            "id": landmark_id,
            "x": round(x_px, 3),
            "y": round(y_px, 3),
            "x_norm": round(x_px / image_w, 6),
            "y_norm": round(y_px / image_h, 6),
            # MediaPipe hand z is relative to the hand landmark origin;
            # it is not a metric depth in the original image.
            "z": round(landmark.z, 6),
        })

    sx, sy = pointing_start
    tx, ty = pointing_tip

    dx = tx - sx
    dy = ty - sy
    length = math.hypot(dx, dy)

    if length > 0:
        direction = {
            "dx": round(dx / length, 6),
            "dy": round(dy / length, 6),
        }
    else:
        direction = {"dx": 0.0, "dy": 0.0}

    return {
        "person_id": person_idx,
        "wrist_side": wrist_side,
        "hand_index_in_crop": hand_idx,
        "crop": {
            "x": crop_x,
            "y": crop_y,
            "width": crop_w,
            "height": crop_h,
        },
        "landmarks": landmarks_json,
        "pointing_direction": {
            "start": {
                "x": round(sx, 3),
                "y": round(sy, 3),
                "x_norm": round(sx / image_w, 6),
                "y_norm": round(sy / image_h, 6),
            },
            "tip": {
                "x": round(tx, 3),
                "y": round(ty, 3),
                "x_norm": round(tx / image_w, 6),
                "y_norm": round(ty / image_h, 6),
            },
            "direction_2d": direction,
        },
    }


# --------------------------------------------------
# Process images
# --------------------------------------------------

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(f"Fant ikke bildemappen: {IMAGE_DIR}")

image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not image_paths:
    raise RuntimeError(f"Fant ingen bilder i {IMAGE_DIR}")

json_results = []
total_crops = 0
total_hands = 0
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

        pose_image, scale = resize_for_pose(original)

        mp_pose_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB),
        )

        pose_result = pose_detector.detect(mp_pose_image)

        image_crop_count = 0
        image_hand_count = 0

        if pose_result.pose_landmarks:
            for person_idx, pose_landmarks in enumerate(
                pose_result.pose_landmarks
            ):
                # Pose indices: left shoulder/elbow/wrist 11/13/15;
                # right shoulder/elbow/wrist 12/14/16.
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

                    wrist_x = int(round(
                        wrist.x * pose_image.shape[1] / scale
                    ))
                    wrist_y = int(round(
                        wrist.y * pose_image.shape[0] / scale
                    ))

                    if not (
                        0 <= wrist_x < original_w
                        and 0 <= wrist_y < original_h
                    ):
                        continue

                    crop_size = calculate_crop_size(
                        shoulder,
                        elbow,
                        pose_image.shape[1],
                        pose_image.shape[0],
                        scale_factor=scale,
                    )

                    crop, crop_x, crop_y = create_wrist_crop(
                        original, wrist_x, wrist_y, crop_size
                    )

                    if crop.size == 0:
                        continue

                    crop_h, crop_w = crop.shape[:2]
                    image_crop_count += 1
                    total_crops += 1

                    crop_name = (
                        f"{image_path.stem}_person{person_idx}_"
                        f"{side}_wrist.jpg"
                    )
                    cv2.imwrite(str(CROP_DIR / crop_name), crop)

                    mp_crop = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                    )

                    hand_result = hand_detector.detect(mp_crop)

                    # Draw crop rectangle and wrist point.
                    cv2.rectangle(
                        visualization,
                        (crop_x, crop_y),
                        (crop_x + crop_w - 1, crop_y + crop_h - 1),
                        (255, 0, 255),
                        2,
                    )
                    cv2.circle(
                        visualization, (wrist_x, wrist_y), 6,
                        (0, 0, 255), -1
                    )
                    cv2.putText(
                        visualization,
                        f"{side} wrist p{person_idx} crop={crop_size}",
                        (crop_x, max(20, crop_y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        2,
                    )

                    if not hand_result.hand_landmarks:
                        print(
                            f"{image_path.name} | person {person_idx} "
                            f"| {side}: ingen hånd funnet "
                            f"(crop={crop_size})"
                        )
                        continue

                    for hand_idx, hand_landmarks in enumerate(
                        hand_result.hand_landmarks
                    ):
                        points = draw_hand_landmarks_on_original(
                            visualization,
                            hand_landmarks,
                            crop_x,
                            crop_y,
                            crop_w,
                            crop_h,
                        )

                        pointing_start, pointing_tip = (
                            draw_pointing_direction(
                                visualization,
                                hand_landmarks,
                                crop_x,
                                crop_y,
                                crop_w,
                                crop_h,
                            )
                        )

                        hand_json = make_hand_json(
                            hand_landmarks=hand_landmarks,
                            image_w=original_w,
                            image_h=original_h,
                            crop_x=crop_x,
                            crop_y=crop_y,
                            crop_w=crop_w,
                            crop_h=crop_h,
                            person_idx=person_idx,
                            wrist_side=side,
                            hand_idx=hand_idx,
                            pointing_start=pointing_start,
                            pointing_tip=pointing_tip,
                        )

                        image_json["hands"].append(hand_json)

                        image_hand_count += 1
                        total_hands += 1

                        print(
                            f"{image_path.name} | person {person_idx} "
                            f"| {side} | hand {hand_idx}: "
                            f"{len(points)} landmarks, crop={crop_size}"
                        )
        else:
            print(f"{image_path.name}: ingen personer funnet")

        json_results.append(image_json)

        print(
            f"{image_path.name}: {image_crop_count} crops, "
            f"{image_hand_count} hender funnet"
        )

        cv2.imwrite(
            str(VIS_DIR / image_path.name),
            visualization,
        )

finally:
    pose_detector.close()
    hand_detector.close()

# Save JSON after all images have been processed.
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(json_results, f, indent=2, ensure_ascii=False)

print("\nFerdig.")
print(f"Bilder behandlet: {images_processed}")
print(f"Crops lagret: {total_crops}")
print(f"Hender funnet: {total_hands}")
print(f"JSON: {JSON_PATH}")
print(f"Crops: {CROP_DIR}")
print(f"Visualiseringer: {VIS_DIR}")
