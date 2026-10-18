
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

# Endre disse til de faktiske mappene dine.
EXPORTS = [
    ROOT / "data/exported/export1/",
    ROOT / "data/exported/export2/",
]

for export_dir in EXPORTS:
    json_path = export_dir / "labels.json"
    image_dir = export_dir / "data"

    if not json_path.exists():
        print(f"\nFant ikke: {json_path}")
        continue

    with json_path.open(encoding="utf-8") as f:
        coco = json.load(f)

    print(f"\n--- {export_dir.name} ---")
    print(f"Bilder i JSON: {len(coco['images'])}")
    print(f"Annoteringer: {len(coco['annotations'])}")

    image_by_id = {img["id"]: img for img in coco["images"]}
    missing_files = []
    dimension_mismatches = []

    for item in coco["images"]:
        filename = Path(item["file_name"]).name
        image_path = image_dir / filename

        if not image_path.exists():
            missing_files.append(filename)
            continue

        with Image.open(image_path) as image:
            actual_width, actual_height = image.size

        expected_width = item["width"]
        expected_height = item["height"]

        if (actual_width, actual_height) != (
            expected_width, expected_height
        ):
            dimension_mismatches.append(
                (
                    filename,
                    (expected_width, expected_height),
                    (actual_width, actual_height),
                )
            )

    invalid_annotations = [
        ann["id"]
        for ann in coco["annotations"]
        if ann["image_id"] not in image_by_id
    ]

    print(f"Manglende bildefiler: {len(missing_files)}")
    for filename in missing_files[:10]:
        print("  Mangler:", filename)

    print(f"Dimensjonsavvik: {len(dimension_mismatches)}")
    for filename, expected, actual in dimension_mismatches[:10]:
        print(f"  {filename}: JSON={expected}, fil={actual}")

    print(f"Annoteringer uten bilde: {len(invalid_annotations)}")
    if invalid_annotations:
        print("  Annotation-ID-er:", invalid_annotations[:10])
