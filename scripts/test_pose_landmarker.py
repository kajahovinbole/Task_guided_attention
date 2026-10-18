
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "mediapipe" / "pose_landmarker_lite.task"
IMAGE_DIR = ROOT / "data" / "exported" / "data"
OUTPUT_DIR = ROOT / "data" / "results" / "pose_landmarker_test"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Fant ikke modellen: {MODEL_PATH}")

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Fant ikke bildemappen: {IMAGE_DIR}")


# Find images
image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not image_paths:
    raise RuntimeError(f"Fant ingen bilder i {IMAGE_DIR}")


# MediaPipe Pose Landmarker
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_poses=5,
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3,
    min_tracking_confidence=0.5,
)

# Landmark indices:
# Left shoulder=11, elbow=13, wrist=15
# Right shoulder=12, elbow=14, wrist=16
ARMS = {
    "left": (11, 13, 15),
    "right": (12, 14, 16),
}

total_people = 0
total_arms = 0

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Kunne ikke lese: {image_path.name}")
            continue

        height, width = image.shape[:2]

        # Resize for faster processing while preserving aspect ratio.
        max_width = 1280
        if width > max_width:
            scale = max_width / width
            resized = cv2.resize(
                image,
                (round(width * scale), round(height * scale)),
            )
        else:
            resized = image.copy()

        vis_image = resized.copy()
        vis_h, vis_w = vis_image.shape[:2]

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = landmarker.detect(mp_image)

        people = len(result.pose_landmarks)
        arms_found = 0

        for person_idx, landmarks in enumerate(result.pose_landmarks):
            # Draw shoulders, elbows and wrists.
            for side, (shoulder_idx, elbow_idx, wrist_idx) in ARMS.items():
                shoulder = landmarks[shoulder_idx]
                elbow = landmarks[elbow_idx]
                wrist = landmarks[wrist_idx]

                # Ignore points with very low visibility.
                visibility_values = [
                    getattr(point, "visibility", 1.0)
                    for point in (shoulder, elbow, wrist)
                ]

                points = (shoulder, elbow, wrist)
                pixel_points = [
                    (round(p.x * vis_w), round(p.y * vis_h))
                    for p in points
                ]

                # Draw all available keypoints.
                for point in points:
                    px = round(point.x * vis_w)
                    py = round(point.y * vis_h)
                    cv2.circle(vis_image, (px, py), 6, (0, 255, 0), -1)

                # Shoulder -> elbow -> wrist
                cv2.line(
                    vis_image, pixel_points[0], pixel_points[1],
                    (255, 180, 0), 3
                )
                cv2.line(
                    vis_image, pixel_points[1], pixel_points[2],
                    (255, 0, 0), 3
                )

                if min(visibility_values) < 0.3:
                    continue

                ex, ey = pixel_points[1]
                wx, wy = pixel_points[2]

                dx = wx - ex
                dy = wy - ey
                length = (dx**2 + dy**2) ** 0.5

                if length < 1:
                    continue

                # Extend the forearm direction beyond the wrist.
                ux = dx / length
                uy = dy / length
                arrow_length = 180

                end = (
                    round(wx + ux * arrow_length),
                    round(wy + uy * arrow_length),
                )

                cv2.arrowedLine(
                    vis_image,
                    (wx, wy),
                    end,
                    (0, 0, 255),
                    4,
                    tipLength=0.12,
                )

                arms_found += 1

                print(
                    f"{image_path.name} | person {person_idx} | "
                    f"{side} arm | "
                    f"elbow=({ex}, {ey}), wrist=({wx}, {wy})"
                )

        output_path = OUTPUT_DIR / f"{image_path.stem}_pose.jpg"
        cv2.imwrite(str(output_path), vis_image)

        total_people += people
        total_arms += arms_found

        print(
            f"{image_path.name}: {people} person(er), "
            f"{arms_found} armretninger tegnet"
        )

print("\nFerdig!")
print(f"Bilder behandlet: {len(image_paths)}")
print(f"Persondeteksjoner totalt: {total_people}")
print(f"Armretninger tegnet: {total_arms}")
print(f"Visualiseringer lagret i: {OUTPUT_DIR}")
