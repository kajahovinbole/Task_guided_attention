
import json
import csv
from pathlib import Path

# Endre disse stiene dersom filene ligger et annet sted
LABELS_PATH = Path("data/exported/export3/labels.json")
MAPPING_PATH = Path("data/exported/export3/data_renamed_by_scene/filename_mapping.csv")
RENAMED_IMAGES_DIR = Path("data/exported/export3/data_renamed_by_scene/imgs")

OUTPUT_PATH = LABELS_PATH.with_name("labels_renamed.json")

# Les koblingen mellom gamle og nye filnavn
with MAPPING_PATH.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    mapping = {
        row["old_filename"]: row["new_filename"]
        for row in reader
    }

# Les COCO-annotasjonene
with LABELS_PATH.open("r", encoding="utf-8") as f:
    coco = json.load(f)

# Oppdater filnavnene uten å endre annotasjonene
missing_mapping = []
missing_images = []

for image in coco["images"]:
    old_name = image["file_name"]

    if old_name not in mapping:
        missing_mapping.append(old_name)
        continue

    new_name = mapping[old_name]

    if not (RENAMED_IMAGES_DIR / new_name).is_file():
        missing_images.append(new_name)
        continue

    image["file_name"] = new_name

if missing_mapping or missing_images:
    print("STOPP: Fant problemer.")
    print("Mangler i mapping:", missing_mapping)
    print("Mangler bildefiler:", missing_images)
    raise SystemExit("Ingen ny JSON-fil ble skrevet.")

# Lagre som en ny fil, slik at originalen er urørt
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(coco, f, indent=2, ensure_ascii=False)

print(f"Ferdig: {OUTPUT_PATH}")
print(f"Oppdaterte filnavn for {len(coco['images'])} bilder.")
print(f"Beholdt {len(coco['annotations'])} annotasjoner.")
