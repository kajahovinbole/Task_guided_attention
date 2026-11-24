"""Save the pointing prior as standalone images for presentations.

Reads maps saved by absvit_prior.py in maps/ and creates two
images per hand in the same 224x224 format used by the model:
  <name>_cone.png  : photo with the cone overlaid (same as panel 1)
  <name>_grid.png  : cone divided into grid cells, as received by the model

Run from the repository root: uv run scripts/visualisering/export_prior_images.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.absvit_prior_experiment import (
    IMAGE_DIR,
    IMAGE_SIZE,
    MAPS_DIR,
    PRIOR_IMAGES_OUTPUT_DIR as OUTPUT_DIR,
)
CONE_ALPHA = 0.45


def save_image(rgb, overlay, path, grid=False):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rgb)
    ax.imshow(
        overlay, cmap="jet", alpha=CONE_ALPHA,
        extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0), interpolation="nearest",
    )
    if grid:
        # Draw the saved model-token grid.
        patch_size = IMAGE_SIZE // overlay.shape[0]
        for k in range(overlay.shape[0] + 1):
            ax.axhline(k * patch_size, color="white", linewidth=0.3, alpha=0.5)
            ax.axvline(k * patch_size, color="white", linewidth=0.3, alpha=0.5)
    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(IMAGE_SIZE, 0)
    ax.axis("off")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    map_paths = sorted(MAPS_DIR.glob("*.npz"))
    if not map_paths:
        print(f"No maps found in {MAPS_DIR}. Run scripts/absvit_prior.py first.")
        return

    for map_path in map_paths:
        with np.load(map_path) as data:
            image_name = str(data["image_name"])
            heatmap = data["heatmap_224"]
            patch_prior = data["patch_prior"]

        image = Image.open(IMAGE_DIR / image_name).convert("RGB")
        rgb = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC))

        save_image(rgb, heatmap, OUTPUT_DIR / f"{map_path.stem}_cone.png")
        save_image(rgb, patch_prior, OUTPUT_DIR / f"{map_path.stem}_grid.png", grid=True)

    print(f"Saved {len(map_paths) * 2} images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
