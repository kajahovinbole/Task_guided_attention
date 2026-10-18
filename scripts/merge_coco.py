
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Endre disse stiene hvis eksportene dine ligger et annet sted.
EXPORTS = [
    ROOT / "data/exported/export1",
    ROOT / "data/exported/export2",
]

OUTPUT = ROOT / "data/exported/merged"

def main():
    output_images = OUTPUT / "data"

    # Ikke overskriv en tidligere sammenslåing.
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} finnes allerede. "
            "Gi OUTPUT en ny mappesti før du kjører skriptet."
        )

    OUTPUT.mkdir(parents=True)
    output_images.mkdir()

    merged_images = []
    merged_annotations = []
    merged_categories = []

    next_image_id = 1
    next_annotation_id = 1
    category_ids = {}

    for export_dir in EXPORTS:
        json_path = export_dir / "labels.json"
        image_dir = export_dir / "data"

        if not json_path.exists():
            raise FileNotFoundError(f"Fant ikke {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # Behold én felles kategori per kategorinavn.
        category_map = {}

        for category in coco["categories"]:
            name = category["name"]

            if name not in category_ids:
                new_category_id = len(category_ids) + 1
                category_ids[name] = new_category_id

                new_category = category.copy()
                new_category["id"] = new_category_id
                merged_categories.append(new_category)

            category_map[category["id"]] = category_ids[name]

        # Lag nye bilde-ID-er og kopier bildene.
        image_id_map = {}

        for image in coco["images"]:
            filename = Path(image["file_name"]).name
            source = image_dir / filename
            destination = output_images / filename

            if not source.is_file():
                raise FileNotFoundError(f"Fant ikke bildefilen: {source}")

            if destination.exists():
                raise FileExistsError(
                    f"Filnavn finnes allerede i sammenslåingen: {filename}"
                )

            shutil.copy2(source, destination)

            new_image = image.copy()
            new_image["id"] = next_image_id
            new_image["file_name"] = filename

            image_id_map[image["id"]] = next_image_id
            merged_images.append(new_image)
            next_image_id += 1

        # Oppdater bilde-ID-er, kategori-ID-er og annoterings-ID-er.
        for annotation in coco["annotations"]:
            if annotation["image_id"] not in image_id_map:
                raise ValueError(
                    f"Annotering {annotation['id']} viser til "
                    f"ukjent bilde-ID {annotation['image_id']}"
                )

            new_annotation = annotation.copy()
            new_annotation["id"] = next_annotation_id
            new_annotation["image_id"] = image_id_map[
                annotation["image_id"]
            ]
            new_annotation["category_id"] = category_map[
                annotation["category_id"]
            ]

            merged_annotations.append(new_annotation)
            next_annotation_id += 1

    merged_coco = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": merged_categories,
    }

    # Behold øvrige metadata fra første eksport hvis de finnes.
    with open(EXPORTS[0] / "labels.json", "r", encoding="utf-8") as f:
        original = json.load(f)

    for key in ("info", "licenses"):
        if key in original:
            merged_coco[key] = original[key]

    with open(OUTPUT / "labels.json", "w", encoding="utf-8") as f:
        json.dump(merged_coco, f, indent=2, ensure_ascii=False)

    print("Sammenslåing fullført!")
    print(f"Bilder: {len(merged_images)}")
    print(f"Annoteringer: {len(merged_annotations)}")
    print(f"Kategorier: {len(merged_categories)}")
    print(f"Resultat: {OUTPUT}")

if __name__ == "__main__":
    main()
