import json
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Paths
ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "exported" / "merged" / "data"
MODEL_PATH = ROOT / "models" / "mediapipe" / "hand_landmarker.task"
OUTPUT_JSON = ROOT / "data" / "mediapipe_results_merged1.json"
VIS_DIR = ROOT / "data" / "results_mediapipe" / "test_18okt" / "mediapipe_visualizations_merged1"


VIS_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Fant ikke modellen: {MODEL_PATH}\n"
        "Legg hand_landmarker.task i prosjektmappen models/."
    )

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Fant ikke bildemappen: {IMAGE_DIR}")


# Find images
extensions = {".jpg", ".jpeg", ".png"}
image_paths = sorted(
    p for p in IMAGE_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
)

if not image_paths:
    raise RuntimeError(f"Fant ingen bilder i {IMAGE_DIR}")


# Configure MediaPipe
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
    min_tracking_confidence=0.2,
)

records = []

with vision.HandLandmarker.create_from_options(options) as landmarker:
    for path in image_paths:
        image = cv2.imread(str(path))

        if image is None:
            print(f"Hopper over bilde som ikke kunne leses: {path.name}")
            continue

        height, width = image.shape[:2]

        # Resize large images to make processing faster.
        # Keep the aspect ratio so normalized coordinates remain valid.
        max_width = 2560
        if width > max_width:
            scale = max_width / width
            resized = cv2.resize(
                image,
                (round(width * scale), round(height * scale))
            )
        else:
            resized = image

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)
        visual = resized.copy()
        hands = []

        if result.hand_landmarks:
            for hand_idx, landmarks in enumerate(result.hand_landmarks):
                points = [
                    {
                        "x": float(p.x),
                        "y": float(p.y),
                        "z": float(p.z),
                    }
                    for p in landmarks
                ]

                # Landmark 5 = index finger MCP/base.
                # Landmark 8 = index fingertip.
                base = points[5]
                tip = points[8]

                dx = tip["x"] - base["x"]
                dy = tip["y"] - base["y"]
                length = (dx**2 + dy**2) ** 0.5

                direction = (
                    {"x": dx / length, "y": dy / length}
                    if length > 1e-8
                    else None
                )

                # Pixel coordinates in the original image.
                base_px = [
                    round(base["x"] * width),
                    round(base["y"] * height),
                ]
                tip_px = [
                    round(tip["x"] * width),
                    round(tip["y"] * height),
                ]

                # Draw landmarks on the resized visualization.
                vis_h, vis_w = visual.shape[:2]
                vis_base = (
                    round(base["x"] * vis_w),
                    round(base["y"] * vis_h),
                )
                vis_tip = (
                    round(tip["x"] * vis_w),
                    round(tip["y"] * vis_h),
                )

                for p in points:
                    px = round(p["x"] * vis_w)
                    py = round(p["y"] * vis_h)
                    cv2.circle(visual, (px, py), 4, (0, 255, 0), -1)

                cv2.line(
                    visual, vis_base, vis_tip, (255, 0, 0), 3
                )

                if direction is not None:
                    end = (
                        round(vis_tip[0] + direction["x"] * 250),
                        round(vis_tip[1] + direction["y"] * 250),
                    )
                    cv2.arrowedLine(
                        visual,
                        vis_tip,
                        end,
                        (0, 0, 255),
                        3,
                        tipLength=0.08,
                    )

                hands.append({
                    "hand_index": hand_idx,
                    "landmarks_normalized": points,
                    "index_base_px": base_px,
                    "index_tip_px": tip_px,
                    "direction_2d": direction,
                })

        records.append({
            "image": path.name,
            "image_path": str(path.relative_to(ROOT)),
            "width": width,
            "height": height,
            "hands_detected": len(hands),
            "hands": hands,
        })

        # Use the relative path to avoid overwriting images with duplicate names.
        relative_path = path.relative_to(IMAGE_DIR)
        output_path = VIS_DIR / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), visual)

        print(f"{path.name}: {len(hands)} hånd(er) funnet")


# Save all results
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"\nFerdig! Resultater lagret i: {OUTPUT_JSON}")
print(f"Visualiseringer lagret i: {VIS_DIR}")
print(f"Bilder behandlet: {len(records)}")
print(
    "Bilder med minst én hånd:",
    sum(record["hands_detected"] > 0 for record in records),
)