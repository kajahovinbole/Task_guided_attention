from pathlib import Path
import sys

# Finner rotmappen til prosjektet (Task_guided_attention/) og legger den til Python-søket
project_root = Path(__file__).resolve().parent.parent
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
checkpoint = torch.load("checkpoints/absvit_tiny.pth", map_location="cpu", weights_only=False)
state_dict = checkpoint.get("model", checkpoint)
model.load_state_dict(state_dict, strict=False)

model.to(device)
model.eval()

# 4. Kjør et dummy-bilde (Batch=1, Channels=3, H=224, W=224)
dummy_img = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    output = model(dummy_img)

print("Modellen kjørte feilfritt!")
print(f"Output-størrelse: {output.shape}")