import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


JSON_PATH = Path(
    "data/results/mediapipe_wrist_hand/hand_landmarks.json"
)

OUTPUT_DIR = Path("data/results/numeric_pointing_prior")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
PATCH_SIZE = 16

CONE_LENGTH = 0.60
CONE_WIDTH = 0.015
CONE_WIDENING = 0.12
CONE_DECAY = 0.50


def create_cone_heatmap(tip, direction, width, height):
    """Returnerer en numerisk heatmap med verdier i [0, 1]."""

    tip_x, tip_y = tip[0] * width, tip[1] * height
    dx, dy = direction

    diagonal = np.hypot(width, height)
    cone_length = CONE_LENGTH * diagonal
    sigma_start = CONE_WIDTH * diagonal

    yy, xx = np.mgrid[0:height, 0:width]
    vx, vy = xx - tip_x, yy - tip_y

    t = vx * dx + vy * dy
    r = np.abs(vx * dy - vy * dx)

    sigma = sigma_start + CONE_WIDENING * np.maximum(t, 0)
    lateral = np.exp(-0.5 * (r / sigma) ** 2)
    longitudinal = np.exp(-t / (CONE_DECAY * cone_length))

    valid = (t > 0) & (t < cone_length)
    heatmap = (lateral * longitudinal * valid).astype(np.float32)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def get_pointing_data(hand):
    pointing = hand.get("pointing_direction")
    if not pointing:
        return None

    tip_data = pointing.get("tip")
    direction = pointing.get("direction_2d")

    if not tip_data or not direction:
        return None

    tip = (tip_data.get("x_norm"), tip_data.get("y_norm"))
    dx, dy = direction.get("dx"), direction.get("dy")

    if None in (*tip, dx, dy):
        return None

    norm = np.hypot(dx, dy)
    if norm < 1e-6:
        return None

    return tip, (dx / norm, dy / norm)


def main():
    data = json.loads(JSON_PATH.read_text())
    patch_grid_size = IMAGE_SIZE // PATCH_SIZE

    for entry in data:
        image_name = Path(entry["image"]).stem

        for hand_index, hand in enumerate(entry.get("hands", [])):
            pointing = get_pointing_data(hand)
            if pointing is None:
                continue

            tip, direction = pointing

            # Heatmap beregnes direkte på modellens inputoppløsning.
            heatmap = create_cone_heatmap(
                tip, direction, IMAGE_SIZE, IMAGE_SIZE
            )

            # [1, 1, H, W] for PyTorch-resampling.
            tensor = torch.from_numpy(heatmap)[None, None]

            # Gjennomsnittlig prior per patch.
            patch_prior = F.avg_pool2d(
                tensor,
                kernel_size=PATCH_SIZE,
                stride=PATCH_SIZE,
            )[0, 0]

            assert patch_prior.shape == (
                patch_grid_size, patch_grid_size
            )

            output_name = f"{image_name}_hand_{hand_index}"

            np.savez_compressed(
                OUTPUT_DIR / f"{output_name}.npz",
                heatmap_224=heatmap,
                patch_prior=patch_prior.numpy(),
                tip=np.asarray(tip, dtype=np.float32),
                direction=np.asarray(direction, dtype=np.float32),
            )

    print(f"Numeriske priorer lagret i {OUTPUT_DIR}")


if __name__ == "__main__":
    main()