
import json
from pathlib import Path

LABELS_PATH = Path("data/exported/merged_exp1_exp2/labels.json")
OUTPUT_PATH = Path("data/funker_bra_20okt/labels.json")

IMG_DIR = Path("data/funker_bra_20okt/imgs")

# Enter the filenames of the images you want to keep
selected_filenames = {img.name for img in IMG_DIR.iterdir()}

with LABELS_PATH.open("r", encoding="utf-8") as f:
    coco = json.load(f)

# Velg bilder
selected_images = [
    image for image in coco["images"]
    if image["file_name"] in selected_filenames
]

# Find the IDs of the selected images
selected_image_ids = {
    image["id"] for image in selected_images
}

# Keep only annotations for the selected images
selected_annotations = [
    annotation for annotation in coco["annotations"]
    if annotation["image_id"] in selected_image_ids
]

# Create a copy of the dataset with filtered images and annotations
filtered_coco = coco.copy()
filtered_coco["images"] = selected_images
filtered_coco["annotations"] = selected_annotations

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(filtered_coco, f, indent=2, ensure_ascii=False)

print(f"Selected images: {len(selected_images)}")
print(f"Annotasjoner: {len(selected_annotations)}")
print(f"Saved to: {OUTPUT_PATH}")
