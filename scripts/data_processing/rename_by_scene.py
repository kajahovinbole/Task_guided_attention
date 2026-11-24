
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor
from sklearn.cluster import KMeans

# ---- Innstillinger ----
INPUT_DIR = Path("data/exported/export3/data")
OUTPUT_DIR = Path("data/exported/export3/data_renamed_by_scene")
N_SCENES = 11  # Change if you have more scenes
START_SCENE_NUMBER = 18  # Startnummer for scene-mapper (scene_04, scene_05, ...)

MODEL_NAME = "openai/clip-vit-base-patch32"

# ---- Find images ----
files = sorted(
    p for p in INPUT_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not files:
    raise ValueError(f"Could not find images in {INPUT_DIR.resolve()}")

print(f"Found {len(files)} images.")

# ---- Last inn CLIP ----
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

embeddings = []

for i, path in enumerate(files, start=1):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
            )
        
        features = model.visual_projection(outputs.pooler_output)

    features = features / features.norm(dim=-1, keepdim=True)
    embeddings.append(features.cpu().numpy()[0])

    print(f"Analysert {i}/{len(files)}: {path.name}")

embeddings = np.stack(embeddings)

# ---- Group similar images ----
if N_SCENES > len(files):
    raise ValueError("N_SCENES cannot be greater than the number of images.")

clustering = KMeans(
    n_clusters=N_SCENES,
    random_state=42,
    n_init=10
)
labels = clustering.fit_predict(embeddings)

# Give groups stable numbers based on the original filename
clusters = sorted(
    set(labels),
    key=lambda label: min(
        files[i].name
        for i in range(len(files))
        if labels[i] == label
    )
)

# ---- Copy and rename files ----
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
records = []

for scene_number, cluster_id in enumerate(clusters, start=START_SCENE_NUMBER):
    members = [
        files[i]
        for i in range(len(files))
        if labels[i] == cluster_id
    ]

    members.sort(key=lambda p: p.name.lower())

    for image_number, source in enumerate(members, start=1):
        new_name = (
            f"scene_{scene_number:02d}_img_{image_number:03d}"
            f"{source.suffix.lower()}"
        )

        destination = OUTPUT_DIR / new_name

        # Copy the file; the original is unchanged
        shutil.copy2(source, destination)

        records.append({
            "old_filename": source.name,
            "new_filename": new_name,
            "old_filepath": str(source.resolve()),
            "new_filepath": str(destination.resolve()),
            "scene": scene_number,
        })

        print(f"{source.name} -> {new_name}")

pd.DataFrame(records).to_csv(
    OUTPUT_DIR / "filename_mapping.csv",
    index=False
)

print(f"\nDone! Images are in {OUTPUT_DIR.resolve()}")
print("Review the grouping manually before continuing.")
