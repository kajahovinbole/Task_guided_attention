"""Mål hvor mye av fokuset som havner innenfor fasit-boksen.

Leser kartene som test_absvit_prior.py lagrer i absvit_prior_test/maps/
og boksene i data/labels.json (COCO: [x, y, bredde, høyde] i originalpiksler).

For hvert bilde regnes ut:
  - tilfeldig  : boksens andel av bildet (forventet verdi ved jevnt fokus)
  - kjegle     : andel av pekekjeglen som ligger i boksen (prioren alene)
  - bottom-up  : andel av token-norm i boksen uten feedback
  - peking     : andel av token-norm i boksen med peke-feedback
  - treff      : om patchen med sterkest fokus ligger i boksen ("pointing game")

Kjør test_absvit_prior.py først, deretter:  uv run scripts/evaluate_bbox.py
"""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent

LABELS_PATH = ROOT / "data" / "labels.json"
IMAGE_DIR = ROOT / "data" / "funker_bra_20okt" / "imgs"
MAPS_DIR = ROOT / "data" / "results" / "absvit_prior_test_28x28_lay9" / "maps"    #absvit_prior_test
OUTPUT_DIR = ROOT / "data" / "results" / "bbox_eval_28x28_lay9" #bbox_eval

IMAGE_SIZE = 224
PATCH_SIZE = 8 #16


def load_boxes():
    """Returnerer {bildenavn uten endelse: [x, y, w, h]} i originalpiksler."""
    coco = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    images = {image["id"]: Path(image["file_name"]).stem for image in coco["images"]}
    return {images[ann["image_id"]]: ann["bbox"] for ann in coco["annotations"]}


def box_to_224(bbox, image_w, image_h):
    """Skalerer boksen på samme måte som bildet ble presset til 224x224."""
    x, y, w, h = bbox
    sx, sy = IMAGE_SIZE / image_w, IMAGE_SIZE / image_h
    return x * sx, y * sy, w * sx, h * sy


def box_masks(box_224):
    """Pikselmaske 224x224 og andel av hver patch som dekkes av boksen (14x14)."""
    x, y, w, h = box_224
    xx = np.arange(IMAGE_SIZE) + 0.5
    inside_x = (xx >= x) & (xx <= x + w)
    inside_y = (xx >= y) & (xx <= y + h)
    pixel_mask = (inside_y[:, None] & inside_x[None, :]).astype(np.float32)

    coverage = F.avg_pool2d(torch.from_numpy(pixel_mask)[None, None], PATCH_SIZE)[0, 0]
    return pixel_mask, coverage.numpy()


def mass_in_box(token_map, coverage):
    """Andel minimumsjustert token-norm i boksen. Delvis dekkede patcher teller delvis."""
    values = token_map - token_map.min()
    total = values.sum()
    if total <= 1e-12:
        return float("nan")
    return float((values * coverage).sum() / total)


def peak_in_box(token_map, coverage):
    """True hvis patchen med høyest token-norm overlapper boksen."""
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
        print(f"Fant ingen kart i {MAPS_DIR}. Kjør test_absvit_prior.py først.")
        return

    rows = []
    print(
        f"{'bilde':<28} {'tilfeldig':>9} {'kjegle':>7} {'bottom-up':>10} "
        f"{'peking':>7}   treff (bu / peking)"
    )

    for map_path in map_paths:
        stem = map_path.stem
        with np.load(map_path) as data:
            image_name = str(data["image_name"])
            image_w, image_h = int(data["image_w"]), int(data["image_h"])
            heatmap = data["heatmap_224"]
            baseline_map = data["baseline_map"]
            pointing_map = data["pointing_map"]

        image_stem = Path(image_name).stem
        if image_stem not in boxes:
            print(f"Mangler boks for {image_name}")
            continue

        box_224 = box_to_224(boxes[image_stem], image_w, image_h)
        pixel_mask, coverage = box_masks(box_224)

        random_fraction = float(pixel_mask.mean())
        cone_fraction = float((heatmap * pixel_mask).sum() / max(heatmap.sum(), 1e-12))
        baseline_fraction = mass_in_box(baseline_map, coverage)
        pointing_fraction = mass_in_box(pointing_map, coverage)
        baseline_hit = peak_in_box(baseline_map, coverage)
        pointing_hit = peak_in_box(pointing_map, coverage)

        rows.append({
            "bilde": stem,
            "tilfeldig": random_fraction,
            "kjegle": cone_fraction,
            "bottom_up": baseline_fraction,
            "peking": pointing_fraction,
            "treff_bottom_up": baseline_hit,
            "treff_peking": pointing_hit,
        })
        print(
            f"{stem:<28} {random_fraction:>9.2f} {cone_fraction:>7.2f} "
            f"{baseline_fraction:>10.2f} {pointing_fraction:>7.2f}   "
            f"{'ja' if baseline_hit else 'nei':>3} / {'ja' if pointing_hit else 'nei'}"
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
            # Bottom-up får egen skala, peking beholder felles skala.
            vmin, vmax = (None, None) if name.startswith("Bottom-up") else (shared_min, shared_max)
            ax.imshow(rgb)
            ax.imshow(
                token_map, cmap="viridis", alpha=0.6,
                extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0),
                interpolation="nearest", vmin=vmin, vmax=vmax,
            )
            # Markér patchen med sterkest fokus.
            row, col = np.unravel_index(np.argmax(token_map), token_map.shape)
            ax.add_patch(Rectangle(
                (col * PATCH_SIZE, row * PATCH_SIZE), PATCH_SIZE, PATCH_SIZE,
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
        print("Ingen bilder ble evaluert.")
        return

    # --- Oppsummering ---
    def mean(key):
        return float(np.nanmean([row[key] for row in rows]))

    print(
        f"{'SNITT':<28} {mean('tilfeldig'):>9.2f} {mean('kjegle'):>7.2f} "
        f"{mean('bottom_up'):>10.2f} {mean('peking'):>7.2f}   "
        f"{sum(r['treff_bottom_up'] for r in rows)}/{len(rows)} / "
        f"{sum(r['treff_peking'] for r in rows)}/{len(rows)}"
    )

    with open(OUTPUT_DIR / "bbox_eval.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Søylediagram per bilde: bottom-up vs peking, med tilfeldig som referanse.
    names = [row["bilde"].replace("_hand", "\nhand") for row in rows]
    positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.6), 5))
    ax.bar(positions - 0.2, [r["bottom_up"] for r in rows], width=0.4, label="Bottom-up")
    ax.bar(positions + 0.2, [r["peking"] for r in rows], width=0.4, label="Pointing")
    ax.scatter(positions, [r["tilfeldig"] for r in rows], color="black", marker="_",
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
