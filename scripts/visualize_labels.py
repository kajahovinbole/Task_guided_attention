
import json
from pathlib import Path

import cv2

# Paths
EXPORT_DIR = Path("data/exported/merged")
LABELS_PATH = EXPORT_DIR / "labels.json"
IMAGE_DIR = EXPORT_DIR / "data"
OUTPUT_DIR = Path("data/exported/verified_bbox_merged_1")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load COCO annotations
with LABELS_PATH.open("r", encoding="utf-8") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

annotations_by_image = {}
for ann in coco["annotations"]:
    annotations_by_image.setdefault(ann["image_id"], []).append(ann)

annotated_count = 0
missing_count = 0

for image_id, image_info in images.items():
    filename = image_info["file_name"]

    # Support both export layouts:
    # data/exported/data/image.jpg
    # data/exported/image.jpg
    image_path = IMAGE_DIR / filename
    if not image_path.exists():
        image_path = EXPORT_DIR / filename

    if not image_path.exists():
        print(f"[MISSING IMAGE] {filename}")
        missing_count += 1
        continue

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERROR] Could not read {image_path}")
        missing_count += 1
        continue

    annotations = annotations_by_image.get(image_id, [])

    for ann in annotations:
        x, y, w, h = ann["bbox"]

        # COCO bbox is [x, y, width, height] in pixels
        x1 = max(0, round(x))
        y1 = max(0, round(y))
        x2 = min(image.shape[1] - 1, round(x + w))
        y2 = min(image.shape[0] - 1, round(y + h))

        if x2 <= x1 or y2 <= y1:
            print(f"[INVALID BBOX] {filename}: {ann['bbox']}")
            continue

        class_name = categories.get(ann["category_id"], "unknown")

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 6)

        label = class_name
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 15, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            4,
            cv2.LINE_AA,
        )

    output_path = OUTPUT_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)

    annotated_count += len(annotations)
    print(f"[OK] {filename}: {len(annotations)} bbox(es)")

print(f"\nFinished. Images saved to: {OUTPUT_DIR.resolve()}")
print(f"Total annotations: {annotated_count}")
print(f"Missing/unreadable images: {missing_count}")
