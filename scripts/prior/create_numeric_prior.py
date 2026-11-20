import json
import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "data" / "3_exported_from_fiftyone" / "1_2_3_andrea_merged" / "data"
RUN_NAME = IMAGE_DIR.parent.name
ENSEMBLE_RUN_DIR = ROOT / "results" / "combined_holistic_pose_hand" / RUN_NAME
JSON_PATH = ENSEMBLE_RUN_DIR / "ensemble_results.json"
OUTPUT_DIR = ENSEMBLE_RUN_DIR / "numeric_pointing_prior"

CONE_LENGTH = 0.60
CONE_WIDTH = 0.015
CONE_WIDENING = 0.12

# Kjeglen er like sterk langs hele midtlinjen (ingen avtakende styrke
# utover). Nærmest fingertuppen trappes den opp lineært fra 0 til 1 over
# denne avstanden (andel av diagonalen), siden man sjelden peker på noe
# rett foran hånden. Sett til 0 for full styrke helt fra fingertuppen.
CONE_NEAR_RAMP = 0.05


def create_cone_heatmap(
    tip,
    direction,
    width,
    height,
    cone_length_fraction=CONE_LENGTH,
    cone_width_fraction=CONE_WIDTH,
    cone_widening=CONE_WIDENING,
    near_ramp_fraction=CONE_NEAR_RAMP,
    longitudinal_decay_fraction=None,
):
    """Returnerer en numerisk heatmap med verdier i [0, 1]."""

    tip_x, tip_y = tip[0] * width, tip[1] * height
    dx, dy = direction

    diagonal = np.hypot(width, height)
    cone_length = cone_length_fraction * diagonal
    sigma_start = cone_width_fraction * diagonal

    yy, xx = np.mgrid[0:height, 0:width]
    vx, vy = xx - tip_x, yy - tip_y

    t = vx * dx + vy * dy
    r = np.abs(vx * dy - vy * dx)

    sigma = sigma_start + cone_widening * np.maximum(t, 0)
    lateral = np.exp(-0.5 * (r / sigma) ** 2)
    near_ramp = near_ramp_fraction * diagonal
    if near_ramp > 0:
        longitudinal = np.clip(t / near_ramp, 0, 1)
    else:
        longitudinal = np.ones_like(t, dtype=np.float64)

    if longitudinal_decay_fraction is not None:
        if longitudinal_decay_fraction <= 0:
            raise ValueError("longitudinal_decay_fraction must be positive")
        decay_distance = longitudinal_decay_fraction * cone_length
        longitudinal *= np.exp(-np.maximum(t, 0) / decay_distance)

    valid = (t > 0) & (t < cone_length)
    heatmap = (lateral * longitudinal * valid).astype(np.float32)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def get_pointing_data(hand, image_width=None, image_height=None):
    """Return normalized fingertip coordinates and direction for one hand.

    The preferred ensemble output stores pixel landmarks under
    ``final_landmarks``. The older ``pointing_direction`` format is still
    accepted for compatibility with existing landmark files.
    """
    landmarks = hand.get("final_landmarks")
    if isinstance(landmarks, list) and len(landmarks) == 21:
        if not image_width or not image_height:
            return None
        try:
            base = np.asarray(
                [landmarks[5]["x"], landmarks[5]["y"]], dtype=np.float64
            )
            tip_px = np.asarray(
                [landmarks[8]["x"], landmarks[8]["y"]], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError):
            return None

        direction = tip_px - base
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None

        tip = (tip_px[0] / image_width, tip_px[1] / image_height)
        direction = direction / norm
        return tip, (float(direction[0]), float(direction[1]))

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
    if not JSON_PATH.is_file():
        raise FileNotFoundError(f"Fant ikke landemerke-JSON: {JSON_PATH}")

    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    generated_count = 0

    for entry in data:
        image_name = Path(entry["image"]).stem
        image_width = entry.get("width", entry.get("image_width"))
        image_height = entry.get("height", entry.get("image_height"))
        hands = entry.get("hands", {})

        if isinstance(hands, dict):
            hand_entries = [
                (hand_index, hands.get(side))
                for hand_index, side in enumerate(("left", "right"))
            ]
        elif isinstance(hands, list):
            hand_entries = list(enumerate(hands))
        else:
            hand_entries = []

        for hand_index, hand in hand_entries:
            if not isinstance(hand, dict):
                continue

            pointing = get_pointing_data(hand, image_width, image_height)
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
            generated_count += 1

    manifest = {
        "source_run": RUN_NAME,
        "source_json": str(JSON_PATH.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(JSON_PATH.read_bytes()).hexdigest(),
        "image_count": len(data),
        "prior_count": generated_count,
        "direction_method": "index MCP landmark 5 to fingertip landmark 8",
        "cone": {
            "length": CONE_LENGTH,
            "width": CONE_WIDTH,
            "widening": CONE_WIDENING,
            "near_ramp": CONE_NEAR_RAMP,
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Pekedata lagret: {generated_count} priorer i {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
