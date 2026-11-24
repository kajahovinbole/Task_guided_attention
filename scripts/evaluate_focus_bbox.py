"""Measure how much focus falls inside the ground-truth box.

Reads maps saved by absvit_prior.py and boxes in
data/funker_bra_20okt/labels.json (COCO: [x, y, width, height] in
original pixels).

For each image, calculate:
  - random    : fraction of the image covered by the box
  - cone      : fraction of the pointing cone inside the box (prior only)
  - bottom-up : fraction of token norm inside the box without feedback
  - pointing  : fraction of token norm inside the box with pointing feedback
  - hit       : whether the strongest-focus patch overlaps the box ("pointing game")

Run absvit_prior.py first, then: uv run scripts/evaluate_focus_bbox.py
"""
from pathlib import Path
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.absvit_prior_experiment import (
    FOCUS_BBOX_OUTPUT_DIR as OUTPUT_DIR,
    IMAGE_DIR,
    IMAGE_SIZE,
    LABELS_PATH,
    MAPS_DIR,
)


def load_boxes():
    """Return {image name without extension: [x, y, w, h]} in original pixels."""
    coco = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    images = {image["id"]: Path(image["file_name"]).stem for image in coco["images"]}
    return {images[ann["image_id"]]: ann["bbox"] for ann in coco["annotations"]}


def box_to_224(bbox, image_w, image_h):
    """Scale the box the same way as the image was resized to 224x224."""
    x, y, w, h = bbox
    sx, sy = IMAGE_SIZE / image_w, IMAGE_SIZE / image_h
    return x * sx, y * sy, w * sx, h * sy


def box_masks(box_224, grid_shape):
    """Return pixel mask and fractional box coverage for the saved token grid."""
    x, y, w, h = box_224
    xx = np.arange(IMAGE_SIZE) + 0.5
    inside_x = (xx >= x) & (xx <= x + w)
    inside_y = (xx >= y) & (xx <= y + h)
    pixel_mask = (inside_y[:, None] & inside_x[None, :]).astype(np.float32)

    coverage = F.adaptive_avg_pool2d(
        torch.from_numpy(pixel_mask)[None, None],
        output_size=grid_shape,
    )[0, 0]
    return pixel_mask, coverage.numpy()


def mass_in_box(token_map, coverage):
    """Fraction of minimum-adjusted token norm in the box. Partially covered patches count proportionally."""
    values = token_map - token_map.min()
    total = values.sum()
    if total <= 1e-12:
        return float("nan")
    return float((values * coverage).sum() / total)


def peak_in_box(token_map, coverage):
    """True if the patch with the highest token norm overlaps the box."""
    row, col = np.unravel_index(np.argmax(token_map), token_map.shape)
    return bool(coverage[row, col] > 0)


def draw_box(ax, box_224):
    x, y, w, h = box_224
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    boxes = load_boxes()

    map_paths = sorted(MAPS_DIR.glob("*.npz"))
    if not map_paths:
        print(f"No maps found in {MAPS_DIR}. Run scripts/absvit_prior.py first.")
        return

    rows = []
    print(
        f"{'image':<28} {'random':>9} {'cone':>7} {'bottom-up':>10} "
        f"{'pointing':>7}   hit (bu / pointing)"
    )

    for map_path in map_paths:
        stem = map_path.stem
        with np.load(map_path) as data:
            image_name = str(data["image_name"])
            image_w, image_h = int(data["image_w"]), int(data["image_h"])
            heatmap = data["heatmap_224"]
            baseline_map = data["baseline_map"]
            pointing_map = data["pointing_map"]
            patch_prior = data["patch_prior"]

        if baseline_map.shape != pointing_map.shape:
            raise ValueError(f"Inconsistent focus grids in {map_path}")
        if baseline_map.shape != patch_prior.shape:
            raise ValueError(
                f"Focus grid {baseline_map.shape} and prior grid "
                f"{patch_prior.shape} do not match in {map_path}"
            )

        image_stem = Path(image_name).stem
        if image_stem not in boxes:
            print(f"Missing box for {image_name}")
            continue

        box_224 = box_to_224(boxes[image_stem], image_w, image_h)
        pixel_mask, coverage = box_masks(box_224, baseline_map.shape)

        random_fraction = float(pixel_mask.mean())
        cone_fraction = float((heatmap * pixel_mask).sum() / max(heatmap.sum(), 1e-12))
        baseline_fraction = mass_in_box(baseline_map, coverage)
        pointing_fraction = mass_in_box(pointing_map, coverage)
        baseline_hit = peak_in_box(baseline_map, coverage)
        pointing_hit = peak_in_box(pointing_map, coverage)

        rows.append({
            "image": stem,
            "random": random_fraction,
            "cone": cone_fraction,
            "bottom_up": baseline_fraction,
            "pointing": pointing_fraction,
            "hit_bottom_up": baseline_hit,
            "hit_pointing": pointing_hit,
        })
        print(
            f"{stem:<28} {random_fraction:>9.2f} {cone_fraction:>7.2f} "
            f"{baseline_fraction:>10.2f} {pointing_fraction:>7.2f}   "
            f"{'yes' if baseline_hit else 'no':>3} / {'yes' if pointing_hit else 'no'}"
        )

        # --- Figur ---
        image = Image.open(IMAGE_DIR / image_name).convert("RGB")
        rgb = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC))

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(rgb)
        axes[0].imshow(heatmap, cmap="jet", alpha=0.4)
        axes[0].set_title(f"{stem}\nPointing cone, fraction in box: {cone_fraction:.2f}")

        shared_min = min(baseline_map.min(), pointing_map.min())
        shared_max = max(baseline_map.max(), pointing_map.max())
        panels = [
            ("Bottom-up (no feedback)", baseline_map, baseline_fraction, baseline_hit),
            ("Pointing top-down", pointing_map, pointing_fraction, pointing_hit),
        ]
        for ax, (name, token_map, fraction, hit) in zip(axes[1:], panels):
            # Bottom-up uses its own scale; pointing keeps the shared scale.
            vmin, vmax = (None, None) if name.startswith("Bottom-up") else (shared_min, shared_max)
            ax.imshow(rgb)
            ax.imshow(
                token_map, cmap="viridis", alpha=0.6,
                extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0),
                interpolation="nearest", vmin=vmin, vmax=vmax,
            )
            # Mark the patch with the strongest focus.
            row, col = np.unravel_index(np.argmax(token_map), token_map.shape)
            grid_height, grid_width = token_map.shape
            cell_width = IMAGE_SIZE / grid_width
            cell_height = IMAGE_SIZE / grid_height
            ax.add_patch(Rectangle(
                (col * cell_width, row * cell_height),
                cell_width, cell_height,
                fill=False, edgecolor="white", linewidth=2, linestyle="--",
            ))
            ax.set_title(
                f"{name}\nFraction in box: {fraction:.2f}  |  peak in box: "
                f"{'yes' if hit else 'no'}"
            )

        for ax in axes:
            draw_box(ax, box_224)
            ax.set_xlim(0, IMAGE_SIZE)
            ax.set_ylim(IMAGE_SIZE, 0)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    if not rows:
        print("No images were evaluated.")
        return

    # --- Summary ---
    def mean(key):
        return float(np.nanmean([row[key] for row in rows]))

    print(
        f"{'MEAN':<28} {mean('random'):>9.2f} {mean('cone'):>7.2f} "
        f"{mean('bottom_up'):>10.2f} {mean('pointing'):>7.2f}   "
        f"{sum(r['hit_bottom_up'] for r in rows)}/{len(rows)} / "
        f"{sum(r['hit_pointing'] for r in rows)}/{len(rows)}"
    )

    with open(OUTPUT_DIR / "bbox_eval.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Bar chart per image: bottom-up vs pointing, with random as reference.
    names = [row["image"].replace("_hand", "\nhand") for row in rows]
    positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.6), 5))
    ax.bar(positions - 0.2, [r["bottom_up"] for r in rows], width=0.4, label="Bottom-up")
    ax.bar(positions + 0.2, [r["pointing"] for r in rows], width=0.4, label="Pointing")
    ax.scatter(positions, [r["random"] for r in rows], color="black", marker="_",
               s=200, label="Random (box area)", zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("Fraction of token norm in box")
    ax.set_title("Focus inside the ground-truth box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "summary.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigurer, summary.png og bbox_eval.csv lagret i {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
