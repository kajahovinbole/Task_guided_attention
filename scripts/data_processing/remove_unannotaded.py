
import json
import shutil
from pathlib import Path
from collections import Counter

EXPORT_DIR = Path("data/exported")
LABELS_PATH = EXPORT_DIR / "labels.json"
IMAGE_DIR = EXPORT_DIR / "data"

# Settes til True for å faktisk slette bilder fra disken
DELETE_IMAGE_FILES = True

# Backup av JSON før endring
BACKUP_PATH = EXPORT_DIR / "labels_backup.json"
shutil.copy2(LABELS_PATH, BACKUP_PATH)

with LABELS_PATH.open("r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

# Tell annoteringer per bilde-ID
counts = Counter(ann["image_id"] for ann in annotations)

# Finn bilder uten bounding boxes
unannotated = [
    image for image in images
    if counts[image["id"]] == 0
]

unannotated_ids = {image["id"] for image in unannotated}

print("Bilder som fjernes:")
for image in unannotated:
    print(f"  {image['file_name']}")

# Fjern bildeoppføringene og eventuelle tilhørende annoteringer
coco["images"] = [
    image for image in images
    if image["id"] not in unannotated_ids
]

coco["annotations"] = [
    ann for ann in annotations
    if ann["image_id"] not in unannotated_ids
]

# Slett bildefilene fra eksportmappen
if DELETE_IMAGE_FILES:
    for image in unannotated:
        image_path = IMAGE_DIR / image["file_name"]
        if image_path.exists():
            image_path.unlink()
            print(f"  Slettet fil: {image_path}")

# Lagre oppdatert COCO-fil
with LABELS_PATH.open("w", encoding="utf-8") as f:
    json.dump(coco, f, indent=2)

print(f"\nFerdig. Fjernet {len(unannotated)} bilder.")
print(f"JSON-sikkerhetskopi: {BACKUP_PATH}")
print(f"Bilder igjen: {len(coco['images'])}")
print(f"Annoteringer igjen: {len(coco['annotations'])}")
