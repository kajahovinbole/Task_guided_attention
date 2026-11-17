from pathlib import Path
import sys

# Finner prosjektroten uavhengig av hvilken mappe kommandoen kjøres fra.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import torch
from models.AbSViT.absvit import absvit_tiny_patch16_224

# 1. Sett enhet (MPS for Mac M-chip, ellers CPU)
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Kjører på device: {device}")

# 2. Initialiser arkitekturen
model = absvit_tiny_patch16_224(pretrained=False)

# 3. Last inn det lokale sjekkpunktet
checkpoint_path = project_root / "checkpoints" / "absvit_tiny.pth"
if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Fant ikke checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("model", checkpoint)
try:
    incompatible = model.load_state_dict(state_dict, strict=False)
except RuntimeError as error:
    print("Checkpointet inneholder vekter med former som ikke passer modellen:")
    print(error)
    raise

if incompatible.missing_keys:
    print("Vekter som mangler i checkpointet (beholder modellens initialisering):")
    for key in incompatible.missing_keys:
        print(f"  - {key}")
else:
    print("Ingen modellvekter mangler i checkpointet.")

if incompatible.unexpected_keys:
    print("Vekter i checkpointet som modellen ikke bruker:")
    for key in incompatible.unexpected_keys:
        print(f"  - {key}")
else:
    print("Ingen uventede vekter i checkpointet.")

model.to(device)
model.eval()

# 4. Kjør et dummy-bilde (Batch=1, Channels=3, H=224, W=224)
dummy_img = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    output = model(dummy_img)

print("Modellen kjørte feilfritt!")
print(f"Output-størrelse: {output.shape}")
