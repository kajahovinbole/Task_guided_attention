
from pathlib import Path

import cv2
import mediapipe as mp


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent

IMAGE_DIR = (
    ROOT
    / "data/3_exported_from_fiftyone/1_2_3_andrea_merged/data"
)

MODEL_PATH = ROOT / "models/mediapipe/holistic_landmarker.task"

OUTPUT_DIR = ROOT / "results/holistic_landmarker/tom_scene28_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Check paths
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Fant ikke Holistic-modellen: {MODEL_PATH}"
    )

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Fant ikke bildemappen: {IMAGE_DIR}"
    )


# --------------------------------------------------
# MediaPipe setup
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
Vision = mp.tasks.vision

options = Vision.HolisticLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=Vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
    min_pose_landmarks_confidence=0.5,
    min_hand_landmarks_confidence=0.5,
)


# --------------------------------------------------
# Drawing functions
# --------------------------------------------------

def draw_pose_landmarks(image, landmarks):
    """Draw pose landmarks and the main body connections."""

    if not landmarks:
        return

    height, width = image.shape[:2]
    points = []

    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)

        points.append((x, y))

        # Only draw landmarks inside the image.
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(
                image,
                (x, y),
                3,
                (255, 0, 255),
                -1,
            )

    # Draw pose connections when available.
    connections = getattr(
        Vision,
        "PoseLandmarksConnections",
        None,
    )

    if connections is not None:
        for connection in connections.POSE_LANDMARKS:
            start = connection.start
            end = connection.end

            if start >= len(points) or end >= len(points):
                continue

            x1, y1 = points[start]
            x2, y2 = points[end]

            if (
                0 <= x1 < width
                and 0 <= y1 < height
                and 0 <= x2 < width
                and 0 <= y2 < height
            ):
                cv2.line(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 255),
                    2,
                )


def draw_hand_landmarks(image, landmarks, color):
    """Draw the 21 hand landmarks and hand connections."""

    if not landmarks:
        return

    height, width = image.shape[:2]
    points = []

    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        points.append((x, y))

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(
                image,
                (x, y),
                3,
                color,
                -1,
            )

    connections = Vision.HandLandmarksConnections.HAND_CONNECTIONS

    for connection in connections:
        start = connection.start
        end = connection.end

        if start >= len(points) or end >= len(points):
            continue

        x1, y1 = points[start]
        x2, y2 = points[end]

        if (
            0 <= x1 < width
            and 0 <= y1 < height
            and 0 <= x2 < width
            and 0 <= y2 < height
        ):
            cv2.line(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

def resize_for_mediapipe(image, target_size=(1280, 1280)):
    """Resize images to a consistent size while preserving aspect ratio."""

    target_w, target_h = target_size
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Pad the image to the same dimensions without stretching it.
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

# --------------------------------------------------
# Find images
# --------------------------------------------------

image_paths = sorted(
    p
    for p in IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not image_paths:
    raise FileNotFoundError(
        f"Fant ingen JPG-, JPEG- eller PNG-bilder i {IMAGE_DIR}"
    )


# --------------------------------------------------
# Run Holistic Landmarker
# --------------------------------------------------

with Vision.HolisticLandmarker.create_from_options(options) as model:

    for path in image_paths:
        image = cv2.imread(str(path))

        if image is None:
            print(f"Kunne ikke lese bildet: {path.name}")
            continue

        # Resize and pad every image to the same dimensions.
        image = resize_for_mediapipe(image)

        # MediaPipe expects RGB.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = model.detect(mp_image)

        # ------------------------------------------
        # Pose
        # ------------------------------------------

        pose_landmarks = result.pose_landmarks

        # In this API, pose_landmarks is a flat list
        # of landmarks for the detected pose.
        if pose_landmarks:
            draw_pose_landmarks(
                image,
                pose_landmarks,
            )

        # ------------------------------------------
        # Left hand
        # ------------------------------------------

        left_hand_landmarks = result.left_hand_landmarks

        if left_hand_landmarks:
            draw_hand_landmarks(
                image,
                left_hand_landmarks,
                (0, 255, 0),
            )

        # ------------------------------------------
        # Right hand
        # ------------------------------------------

        right_hand_landmarks = result.right_hand_landmarks

        if right_hand_landmarks:
            draw_hand_landmarks(
                image,
                right_hand_landmarks,
                (0, 255, 255),
            )

        # ------------------------------------------
        # Save visualization
        # ------------------------------------------

        output_path = OUTPUT_DIR / path.name

        success = cv2.imwrite(
            str(output_path),
            image,
        )

        if not success:
            print(f"Kunne ikke lagre: {output_path}")
            continue

        left_found = bool(left_hand_landmarks)
        right_found = bool(right_hand_landmarks)
        pose_found = bool(pose_landmarks)

        print(
            f"{path.name}: "
            f"pose={pose_found}, "
            f"left_hand={left_found}, "
            f"right_hand={right_found}"
        )


print(f"\nFerdig. Antall bilder: {len(image_paths)}")
print(f"Resultatene er lagret i: {OUTPUT_DIR}")
