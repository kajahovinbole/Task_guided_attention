
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor
from sklearn.cluster import KMeans

# ---- Innstillinger ----
INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/images_grouped")
N_SCENES = 2  # Endre hvis du har flere scener

MODEL_NAME = "openai/clip-vit-base-patch32"

# ---- Finn bilder ----
files = sorted(
    p for p in INPUT_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not files:
    raise ValueError(f"Fant ingen bilder i {INPUT_DIR.resolve()}")

print(f"Fant {len(files)} bilder.")

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

# ---- Grupper lignende bilder ----
if N_SCENES > len(files):
    raise ValueError("N_SCENES kan ikke være større enn antall bilder.")

clustering = KMeans(
    n_clusters=N_SCENES,
    random_state=42,
    n_init=10
)
labels = clustering.fit_predict(embeddings)

# Gi gruppene stabile nummer basert på originalt filnavn
clusters = sorted(
    set(labels),
    key=lambda label: min(
        files[i].name
        for i in range(len(files))
        if labels[i] == label
    )
)

# ---- Kopier og gi nye navn ----
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
records = []

for scene_number, cluster_id in enumerate(clusters, start=1):
    members = [
        files[i]
        for i in range(len(files))
        if labels[i] == cluster_id
    ]

    members.sort(key=lambda p: p.name.lower())

    for image_number, source in enumerate(members, start=1):
        new_name = (
            f"scene_{scene_number:02d}_img_{image_number:03d}.jpg"
        )
        destination = OUTPUT_DIR / new_name

        # Kopier filen; originalen endres ikke
        shutil.copy2(source, destination)

        records.append({
            "old_filename": source.name,
            "new_filename": new_name,
            "scene": scene_number,
        })

        print(f"{source.name} -> {new_name}")

pd.DataFrame(records).to_csv(
    OUTPUT_DIR / "filename_mapping.csv",
    index=False
)

print(f"\nFerdig! Bilder ligger i {OUTPUT_DIR.resolve()}")
print("Kontroller grupperingen manuelt før videre bruk.")
