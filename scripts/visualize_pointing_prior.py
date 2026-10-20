import json
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# Paths
# --------------------------------------------------

JSON_PATH = Path("data/results/mediapipe_wrist_hand/hand_landmarks.json")
IMAGE_DIR = Path("data/exported/merged/data")
OUTPUT_DIR = Path("data/results/pointing_prior_heatmap_visualization")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Cone parameters
# --------------------------------------------------

# Lengde på kjeglen som andel av bildediagonalen
CONE_LENGTH = 0.60

# Startbredde ved fingertuppen, andel av bildediagonalen
CONE_WIDTH = 0.015

# Hvor mye kjeglen utvider seg per andel av bildediagonalen
CONE_WIDENING = 0.12

# Hvor raskt prioren svekkes langs pekretningen
CONE_DECAY = 0.50

# Styrke på heatmap-overlay
OVERLAY_ALPHA = 0.45


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def get_pointing_data(hand):
    """
    Henter fingertupp og retning fra én hånd.
    Returnerer koordinater i normalisert bildeformat.
    """

    pointing = hand.get("pointing_direction")

    if pointing is None:
        return None

    direction = pointing.get("direction_2d")

    if direction is None:
        return None

    tip = pointing.get("tip")

    if tip is None:
        return None

    tip_x = tip.get("x_norm")
    tip_y = tip.get("y_norm")

    if tip_x is None or tip_y is None:
        return None

    dx = direction.get("dx")
    dy = direction.get("dy")

    if dx is None or dy is None:
        return None

    norm = np.sqrt(dx**2 + dy**2)

    if norm < 1e-6:
        return None

    dx /= norm
    dy /= norm

    return (tip_x, tip_y), (dx, dy)


def create_cone_heatmap(
    image_width,
    image_height,
    tip,
    direction,
):
    """
    Lager en kjegleformet heatmap med verdier mellom 0 og 1.

    tip: (x, y) normalisert til [0, 1]
    direction: (dx, dy), normalisert retning
    """

    # Pikselkoordinater
    tip_x = tip[0] * image_width
    tip_y = tip[1] * image_height

    dx, dy = direction

    # Bruk bildediagonalen slik at parametrene
    # skaleres med bildestørrelsen
    diagonal = np.sqrt(image_width**2 + image_height**2)

    cone_length = CONE_LENGTH * diagonal
    sigma_start = CONE_WIDTH * diagonal
    widening = CONE_WIDENING

    # Pikselkoordinater for hele bildet
    yy, xx = np.mgrid[0:image_height, 0:image_width]

    # Avstand fra fingertupp
    vx = xx - tip_x
    vy = yy - tip_y

    # Avstand langs pekretningen
    t = vx * dx + vy * dy

    # Avstand vinkelrett på pekretningen
    r = np.abs(vx * dy - vy * dx)

    # Kjeglen blir bredere jo lenger unna fingertuppen
    sigma = sigma_start + widening * np.maximum(t, 0)

    # Gaussisk falloff på tvers av kjeglen
    lateral = np.exp(-0.5 * (r / sigma) ** 2)

    # Avtakende styrke langs kjeglen
    longitudinal = np.exp(-t / (CONE_DECAY * cone_length))

    # Bare fremover, og bare innenfor kjeglelengden
    valid = (t > 0) & (t < cone_length)

    heatmap = lateral * longitudinal * valid

    # Normaliser til [0, 1]
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap.astype(np.float32)


def overlay_heatmap(image, heatmap):
    """
    Tegner heatmap oppå originalbildet.
    """

    # OpenCV bruker BGR
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    colored = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    # Ikke fargelegg områder med tilnærmet null prior
    mask = heatmap > 0.02

    result = image.copy()

    blended = cv2.addWeighted(
        image,
        1 - OVERLAY_ALPHA,
        colored,
        OVERLAY_ALPHA,
        0,
    )

    result[mask] = blended[mask]

    return result


# --------------------------------------------------
# Main
# --------------------------------------------------

with open(JSON_PATH, "r") as f:
    data = json.load(f)

# Støtt både JSON som liste og JSON med "images"-felt
images = data if isinstance(data, list) else data.get("images", [])

processed = 0

for image_entry in images:

    image_name = image_entry.get("image")

    if not image_name:
        continue

    image_path = IMAGE_DIR / Path(image_name).name

    if not image_path.exists():
        print(f"Mangler bilde: {image_path}")
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Kunne ikke lese: {image_path}")
        continue

    height, width = image.shape[:2]

    hands = image_entry.get("hands", [])

    # Ett resultat per hånd
    for hand_index, hand in enumerate(hands):

        pointing_data = get_pointing_data(hand)

        if pointing_data is None:
            continue

        tip, direction = pointing_data

        heatmap = create_cone_heatmap(
            width,
            height,
            tip,
            direction,
        )

        result = overlay_heatmap(image, heatmap)

        # Tegn fingertupp
        tip_x = int(tip[0] * width)
        tip_y = int(tip[1] * height)

        cv2.circle(
            result,
            (tip_x, tip_y),
            7,
            (255, 255, 255),
            -1,
        )

        # Tegn pil i pekretningen
        arrow_length = int(CONE_LENGTH * np.sqrt(width**2 + height**2))

        end_x = int(tip_x + direction[0] * arrow_length)
        end_y = int(tip_y + direction[1] * arrow_length)

        cv2.arrowedLine(
            result,
            (tip_x, tip_y),
            (end_x, end_y),
            (255, 255, 255),
            3,
            tipLength=0.04,
        )

        output_name = (
            f"{image_path.stem}_hand_{hand_index}_prior.jpg"
        )

        output_path = OUTPUT_DIR / output_name

        cv2.imwrite(str(output_path), result)

        processed += 1

print(f"Ferdig. Lagret {processed} heatmaps i {OUTPUT_DIR}")