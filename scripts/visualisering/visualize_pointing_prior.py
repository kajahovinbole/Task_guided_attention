import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Kjeglen og pekedataene hentes fra create_numeric_prior.py, så
# visualiseringen alltid viser samme kjegle som AbSViT får.
from scripts.prior.create_numeric_prior import (
    CONE_LENGTH,
    create_cone_heatmap,
    get_pointing_data,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

IMAGE_DIR = ROOT / "data" / "3_exported_from_fiftyone" / "1_2_3_andrea_merged" / "data"
RUN_NAME = IMAGE_DIR.parent.name
ENSEMBLE_RUN_DIR = ROOT / "results" / "combined_holistic_pose_hand" / RUN_NAME
JSON_PATH = ENSEMBLE_RUN_DIR / "ensemble_results.json"
OUTPUT_DIR = ENSEMBLE_RUN_DIR / "pointing_prior_visualizations"


# Styrke på heatmap-overlay
OVERLAY_ALPHA = 0.45


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

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

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not JSON_PATH.is_file():
        raise FileNotFoundError(f"Fant ikke landemerke-JSON: {JSON_PATH}")

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Fant ikke bildemappen: {IMAGE_DIR}")

    with JSON_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    # Støtt både JSON som liste og JSON med "images"-felt.
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

        hands = image_entry.get("hands", {})

        if isinstance(hands, dict):
            hand_entries = [
                (hand_index, hands.get(side))
                for hand_index, side in enumerate(("left", "right"))
            ]
        elif isinstance(hands, list):
            hand_entries = list(enumerate(hands))
        else:
            hand_entries = []

        # Ett resultat per hånd
        for hand_index, hand in hand_entries:
            if not isinstance(hand, dict):
                continue

            pointing_data = get_pointing_data(
                hand,
                image_entry.get("width", image_entry.get("image_width")),
                image_entry.get("height", image_entry.get("image_height")),
            )

            if pointing_data is None:
                continue

            tip, direction = pointing_data

            heatmap = create_cone_heatmap(tip, direction, width, height)

            result = overlay_heatmap(image, heatmap)

            # Tegn fingertupp
            tip_x = int(tip[0] * width)
            tip_y = int(tip[1] * height)

            cv2.circle(result, (tip_x, tip_y), 7, (255, 255, 255), -1)

            # Tegn pil i pekretningen
            arrow_length = int(CONE_LENGTH * np.sqrt(width**2 + height**2))

            end_x = int(tip_x + direction[0] * arrow_length)
            end_y = int(tip_y + direction[1] * arrow_length)

            cv2.arrowedLine(result, (tip_x, tip_y), (end_x, end_y),
                            (255, 255, 255), 3, tipLength=0.04)

            output_name = f"{image_path.stem}_hand_{hand_index}_prior.jpg"

            output_path = OUTPUT_DIR / output_name

            if not cv2.imwrite(str(output_path), result):
                print(f"Kunne ikke lagre: {output_path}")
                continue

            processed += 1

    print(f"Ferdig. Lagret {processed} heatmaps i {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
