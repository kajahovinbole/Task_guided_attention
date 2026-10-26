"""Sammenlign bottom-up ViT med AbSViT styrt av en spatial pekeprior.

Ingen prompt-prior brukes. Skriptet bruker prosjektets eksisterende
models.AbSViT og scripts.create_numeric_prior.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.AbSViT.absvit import absvit_tiny_patch8_224_gap
#from models.AbSViT.absvit import absvit_tiny_patch16_224
from scripts.create_numeric_prior import create_cone_heatmap


# --------------------------------------------------
# Innstillinger
# --------------------------------------------------
DATA_DIR = ROOT / "data" / "funker_bra_20okt"
IMAGE_DIR = DATA_DIR / "imgs"
PRIOR_DIR = DATA_DIR / "numeric_prior"
CHECKPOINT = ROOT / "checkpoints" / "absvit_tiny_patch8_224_gap.pth"#"absvit_tiny.pth"
OUTPUT_DIR = ROOT / "data" / "results" / "absvit_prior_test_28x28_lay9" #"absvit_prior_test"
# Kartene lagres her slik at evaluate_bbox.py kan bruke dem.
MAPS_DIR = OUTPUT_DIR / "maps"

IMAGE_SIZE = 224
PATCH_SIZE = 8

# Valgt fra tidligere sveip. Bør senere valideres mot objektlokalisering,
# ikke bare andelen token-norm inne i den geometriske kjeglen.
PRIOR_SCALE = 40.0
CLS_PRIOR = 0.0

# Antall lag fra slutten til laget hvor token-normen visualiseres.
ATTENTION_LAYER_FROM_END = 3

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_image(path):
    """Returnerer (normalisert [1,3,224,224], RGB-array, (W,H))."""
    image = Image.open(path).convert("RGB")
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    rgb = np.asarray(resized).copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return ((tensor - MEAN) / STD).unsqueeze(0), rgb, image.size


def make_patch_prior(tip, direction, image_w, image_h):
    """Lag pekekjegle med originalt sideforhold, resample til ViT-grid."""
    small_w = 448
    small_h = round(small_w * image_h / image_w)
    heatmap = create_cone_heatmap(tip, direction, small_w, small_h)
    heatmap = torch.as_tensor(heatmap, dtype=torch.float32)[None, None]
    heatmap = F.interpolate(
        heatmap, size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear", align_corners=False,
    )
    patch_prior = F.avg_pool2d(heatmap, PATCH_SIZE)[0, 0]
    return heatmap[0, 0], patch_prior


def prior_to_mask(patch_prior, num_prefix_tokens):
    """[14,14] -> [1,197,1] når modellen har ett CLS-token."""
    patches = patch_prior.flatten()
    max_value = patches.max().item()
    if max_value > 0:
        patches = patches / max_value * PRIOR_SCALE
    else:
        patches = torch.zeros_like(patches)
    prefix = torch.full(
        (num_prefix_tokens,), CLS_PRIOR, dtype=patches.dtype,
        device=patches.device,
    )
    return torch.cat([prefix, patches], dim=0)[None, :, None]


def get_token_norm_map(model, image_tensor, td=None):
    """Token-norm i et mellomliggende ViT-lag, ikke QK attention-vekter."""
    layer = len(model.blocks) - ATTENTION_LAYER_FROM_END
    if not (1 <= layer <= len(model.blocks)):
        raise ValueError("Ugyldig ATTENTION_LAYER_FROM_END")

    tokens = model._pos_embed(model.patch_embed(image_tensor))
    for i in range(layer):
        tokens = model.blocks[i](tokens, td[i] if td is not None else None)
    norms = tokens.norm(dim=-1)[0, model.num_prefix_tokens:]
    side = int(norms.numel() ** 0.5)
    if side * side != norms.numel():
        raise ValueError(f"Token-antall {norms.numel()} kan ikke formes som kvadrat")
    return norms.reshape(side, side)


def run_pointing(model, image_tensor, features, pointing_mask):
    """Lag top-down feedback fra pekeprioren og returner token-norm i andre pass."""
    td = model.feedback(features * pointing_mask * model.feedback_aug)
    return get_token_norm_map(model, image_tensor, td)


def mass_in_prior(token_norm_map, patch_prior):
    """Andel minimumsjustert token-norm i patches der pekeprioren > 0.05."""
    if patch_prior.max().item() <= 0:
        return float("nan")
    inside = (patch_prior / patch_prior.max()) > 0.05
    values = token_norm_map - token_norm_map.min()
    total = values.sum().item()
    if total <= 1e-12:
        return float("nan")
    return values[inside].sum().item() / total


def find_image(stem):
    return next(iter(sorted(IMAGE_DIR.glob(f"{stem}.*"))), None)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    #model = absvit_tiny_patch16_224(pretrained=False)
    model = absvit_tiny_patch8_224_gap(pretrained=False)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()

    print(f"{'bilde':<28} {'tilfeldig':>9} {'bottom-up':>10} {'peking':>9}")
    all_masses = []

    for prior_path in sorted(PRIOR_DIR.glob("*.npz")):
        stem = prior_path.stem
        image_stem = stem.rsplit("_hand_", 1)[0]
        image_path = find_image(image_stem)
        if image_path is None:
            print(f"Mangler bilde for {stem}")
            continue

        with np.load(prior_path) as data:
            tip = data["tip"]
            direction = data["direction"]

        x, rgb, (image_w, image_h) = load_image(image_path)
        heatmap, patch_prior = make_patch_prior(tip, direction, image_w, image_h)
        pointing_mask = prior_to_mask(patch_prior, model.num_prefix_tokens)

        with torch.no_grad():
            # Pass 1 trengs fordi feedbacken bygges fra disse featurene.
            features, _, _ = model.forward_features(x)
            baseline_map = get_token_norm_map(model, x)
            pointing_map = run_pointing(model, x, features, pointing_mask)

        random_fraction = (
            (patch_prior / patch_prior.max() > 0.05).float().mean().item()
            if patch_prior.max().item() > 0 else float("nan")
        )
        baseline_fraction = mass_in_prior(baseline_map, patch_prior)
        pointing_fraction = mass_in_prior(pointing_map, patch_prior)
        all_masses.append([random_fraction, baseline_fraction, pointing_fraction])

        print(
            f"{stem:<28} {random_fraction:>9.2f} {baseline_fraction:>10.2f} "
            f"{pointing_fraction:>9.2f}"
        )

        np.savez_compressed(
            MAPS_DIR / f"{stem}.npz",
            image_name=image_path.name,
            image_w=image_w,
            image_h=image_h,
            heatmap_224=heatmap.numpy(),
            patch_prior=patch_prior.numpy(),
            baseline_map=baseline_map.numpy(),
            pointing_map=pointing_map.numpy(),
        )

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(rgb)
        axes[0].imshow(heatmap.numpy(), cmap="jet", alpha=0.4)
        axes[0].set_title(f"{stem}\nInput + pointing prior")

        results = [
            ("Bottom-up (no feedback)", baseline_map, baseline_fraction),
            ("Pointing top-down", pointing_map, pointing_fraction),
        ]
        # Samme fargeskala for direkte visuell sammenligning av token-norm.
        shared_min = min(float(baseline_map.min()), float(pointing_map.min()))
        shared_max = max(float(baseline_map.max()), float(pointing_map.max()))
        for ax, (name, token_map, fraction) in zip(axes[1:], results):
            # Bottom-up får egen skala, peking beholder felles skala.
            vmin, vmax = (None, None) if name.startswith("Bottom-up") else (shared_min, shared_max)
            ax.imshow(rgb)
            ax.imshow(
                token_map.numpy(), cmap="viridis", alpha=0.6,
                extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0),
                interpolation="nearest", vmin=vmin, vmax=vmax,
            )
            ax.set_title(f"{name}\nFraction in cone: {fraction:.2f}")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        # bbox_inches="tight" tar med titler som ellers kuttes i toppen.
        fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    if not all_masses:
        print("Fant ingen bilde/prior-par å evaluere.")
        return

    mean = np.nanmean(np.asarray(all_masses), axis=0)
    print(f"{'SNITT':<28} {mean[0]:>9.2f} {mean[1]:>10.2f} {mean[2]:>9.2f}")
    print(f"\nFigurer lagret i {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
