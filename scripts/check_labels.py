
import json
from pathlib import Path
from collections import Counter

LABELS_PATH = Path("data/exported/labels.json")

with LABELS_PATH.open("r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

image_ids = {image["id"] for image in images}
annotation_image_ids = {ann["image_id"] for ann in annotations}

# Tell antall annoteringer per bilde
counts = Counter(ann["image_id"] for ann in annotations)

# Bilder uten annotering
without_labels = [
    image["file_name"]
    for image in images
    if counts[image["id"]] == 0
]

# Annoteringer som peker på bilder som ikke finnes i JSON
orphan_annotations = [
    ann for ann in annotations
    if ann["image_id"] not in image_ids
]

print(f"Antall bilder i JSON: {len(images)}")
print(f"Antall annoteringer: {len(annotations)}")
print(f"Bilder uten annotering: {len(without_labels)}")
print(f"Annoteringer uten tilhørende bilde: {len(orphan_annotations)}")

if without_labels:
    print("\nBilder uten annotering:")
    for filename in without_labels:
        print(f"  - {filename}")

if orphan_annotations:
    print("\nFEIL: Annoteringer peker på bilder som ikke finnes:")
    for ann in orphan_annotations:
        print(f"  - Annotation ID {ann['id']}, image_id {ann['image_id']}")
else:
    print("\nOK: Alle annoteringer peker på et bilde som finnes i JSON.")
