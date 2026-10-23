import json
from pathlib import Path

import numpy as np


JSON_PATH = Path(
    "data/results/mediapipe_wrist_hand/hand_landmarks.json"
)

OUTPUT_DIR = Path("data/results/numeric_pointing_prior")

CONE_LENGTH = 0.60
CONE_WIDTH = 0.015
CONE_WIDENING = 0.12

# Kjeglen er like sterk langs hele midtlinjen (ingen avtakende styrke
# utover). Nærmest fingertuppen trappes den opp lineært fra 0 til 1 over
# denne avstanden (andel av diagonalen), siden man sjelden peker på noe
# rett foran hånden. Sett til 0 for full styrke helt fra fingertuppen.
CONE_NEAR_RAMP = 0.05


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
    near_ramp = CONE_NEAR_RAMP * diagonal
    if near_ramp > 0:
        longitudinal = np.clip(t / near_ramp, 0, 1)
    else:
        longitudinal = np.ones_like(t, dtype=np.float64)

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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(JSON_PATH.read_text())

    for entry in data:
        image_name = Path(entry["image"]).stem

        for hand_index, hand in enumerate(entry.get("hands", [])):
            pointing = get_pointing_data(hand)
            if pointing is None:
                continue

            tip, direction = pointing
            output_name = f"{image_name}_hand_{hand_index}"

            # Bare pekingen lagres. Kjeglen lages i test_absvit_prior.py,
            # med riktig sideforhold og rutenett for modellen som brukes.
            np.savez_compressed(
                OUTPUT_DIR / f"{output_name}.npz",
                tip=np.asarray(tip, dtype=np.float32),
                direction=np.asarray(direction, dtype=np.float32),
            )

    print(f"Pekedata lagret i {OUTPUT_DIR}")


if __name__ == "__main__":
    main()