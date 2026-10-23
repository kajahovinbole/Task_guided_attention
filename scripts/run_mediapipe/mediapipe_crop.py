from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "mediapipe" / "hand_landmarker.task"
IMAGE_DIR = ROOT / "data" / "exported" / "data"
OUTPUT_DIR = ROOT / "data" / "results" / "mediapipe_crop_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Fant ikke modellen: {MODEL_PATH}")

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Fant ikke bildemappen: {IMAGE_DIR}")

image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not image_paths:
    raise RuntimeError(f"Fant ingen bilder i {IMAGE_DIR}")

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
)

total_hands = 0
processed = 0

with vision.HandLandmarker.create_from_options(options) as landmarker:
    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Hopper over bilde som ikke kunne leses: {image_path.name}")
            continue

        height, width = image.shape[:2]

        # Midlertidig testutsnitt – samme relative koordinater for alle bilder.
        x1, y1 = int(width * 0.48), int(height * 0.35)
        x2, y2 = int(width * 0.67), int(height * 0.68)
        crop = image[y1:y2, x1:x2]

        stem = image_path.stem
        crop_path = OUTPUT_DIR / f"{stem}_crop.jpg"
        result_path = OUTPUT_DIR / f"{stem}_result.jpg"

        cv2.imwrite(str(crop_path), crop)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = landmarker.detect(mp_image)
        hand_count = len(result.hand_landmarks)

        print(f"{image_path.name}: {hand_count} hånd(er) funnet")

        # Tegn landemerkene i utsnittet.
        for landmarks in result.hand_landmarks:
            for point in landmarks:
                px = round(point.x * crop.shape[1])
                py = round(point.y * crop.shape[0])
                cv2.circle(crop, (px, py), 3, (0, 255, 0), -1)

        cv2.imwrite(str(result_path), crop)

        total_hands += hand_count
        processed += 1

print(f"\nFerdig! Bilder behandlet: {processed}")
print(f"Hender funnet totalt: {total_hands}")
print(f"Resultater lagret i: {OUTPUT_DIR}")
