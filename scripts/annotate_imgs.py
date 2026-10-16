
from pathlib import Path
import fiftyone as fo

IMAGE_DIR = Path("data/images_grouped")

# Finn bilder rekursivt
image_paths = sorted(
    p for p in IMAGE_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not image_paths:
    raise ValueError(f"Fant ingen bilder i {IMAGE_DIR.resolve()}")

dataset = fo.Dataset("pointing_poc")

for path in image_paths:
    sample = fo.Sample(filepath=str(path.resolve()))

    # Lagre scene-ID fra filnavnet
    sample["scene_id"] = path.stem.split("_img_")[0]

    dataset.add_sample(sample)

session = fo.launch_app(dataset)
session.wait()
