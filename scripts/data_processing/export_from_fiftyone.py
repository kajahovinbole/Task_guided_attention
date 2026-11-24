from pathlib import Path
import fiftyone as fo
import fiftyone.types as fot

# Load the dataset from FiftyOne
dataset = fo.load_dataset("pointing_poc_5")

# Folder where the export should be saved
export_dir = Path("data/exported/export3")
export_dir.mkdir(parents=True, exist_ok=True)

# Export images and bounding boxes
dataset.export(
    export_dir=str(export_dir),
    dataset_type=fot.COCODetectionDataset,
    label_field="target_bbox",
)

print(f"Exported dataset to: {export_dir.resolve()}")