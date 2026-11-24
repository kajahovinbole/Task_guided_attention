
import json
import shutil
from pathlib import Path
from collections import Counter

EXPORT_DIR = Path("data/exported")
LABELS_PATH = EXPORT_DIR / "labels.json"
IMAGE_DIR = EXPORT_DIR / "data"

# Set to True to delete images from disk
DELETE_IMAGE_FILES = True

# Back up JSON before making changes
BACKUP_PATH = EXPORT_DIR / "labels_backup.json"
shutil.copy2(LABELS_PATH, BACKUP_PATH)

with LABELS_PATH.open("r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

# Count annotations per image ID
counts = Counter(ann["image_id"] for ann in annotations)

# Find images without bounding boxes
unannotated = [
    image for image in images
    if counts[image["id"]] == 0
]

unannotated_ids = {image["id"] for image in unannotated}

print("Images to remove:")
for image in unannotated:
    print(f"  {image['file_name']}")

# Remove image entries and their associated annotations
coco["images"] = [
    image for image in images
    if image["id"] not in unannotated_ids
]

coco["annotations"] = [
    ann for ann in annotations
    if ann["image_id"] not in unannotated_ids
]

# Delete image files from the export folder
if DELETE_IMAGE_FILES:
    for image in unannotated:
        image_path = IMAGE_DIR / image["file_name"]
        if image_path.exists():
            image_path.unlink()
            print(f"  Deleted file: {image_path}")

# Save the updated COCO file
with LABELS_PATH.open("w", encoding="utf-8") as f:
    json.dump(coco, f, indent=2)

print(f"\nDone. Removed {len(unannotated)} images.")
print(f"JSON backup: {BACKUP_PATH}")
print(f"Images remaining: {len(coco['images'])}")
print(f"Annotations remaining: {len(coco['annotations'])}")
