from pathlib import Path
import fiftyone as fo
import fiftyone.types as fot

# Last inn datasettet fra FiftyOne
dataset = fo.load_dataset("pointing_poc_2")

# Mappe der eksporten skal lagres
export_dir = Path("data/exported/18okt")
export_dir.mkdir(parents=True, exist_ok=True)

# Eksporter bilder og bounding boxes
dataset.export(
    export_dir=str(export_dir),
    dataset_type=fot.COCODetectionDataset,
    label_field="target_bbox",
)

print(f"Eksportert datasett til: {export_dir.resolve()}")