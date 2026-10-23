
import json
from pathlib import Path

LABELS_PATH = Path("data/exported/merged_exp1_exp2/labels.json")
OUTPUT_PATH = Path("data/funker_bra_20okt/labels.json")

IMG_DIR = Path("data/funker_bra_20okt/imgs")

# Skriv inn filnavnene på bildene du vil beholde
selected_filenames = {img.name for img in IMG_DIR.iterdir()}

with LABELS_PATH.open("r", encoding="utf-8") as f:
    coco = json.load(f)

# Velg bilder
selected_images = [
    image for image in coco["images"]
    if image["file_name"] in selected_filenames
]

# Finn ID-ene til de valgte bildene
selected_image_ids = {
    image["id"] for image in selected_images
}

# Behold bare annotasjonene til de valgte bildene
selected_annotations = [
    annotation for annotation in coco["annotations"]
    if annotation["image_id"] in selected_image_ids
]

# Lag en kopi av datasettet med filtrerte bilder og annotasjoner
filtered_coco = coco.copy()
filtered_coco["images"] = selected_images
filtered_coco["annotations"] = selected_annotations

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(filtered_coco, f, indent=2, ensure_ascii=False)

print(f"Valgte bilder: {len(selected_images)}")
print(f"Annotasjoner: {len(selected_annotations)}")
print(f"Lagret til: {OUTPUT_PATH}")
