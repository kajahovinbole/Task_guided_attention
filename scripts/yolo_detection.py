from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "funker_bra_20okt" / "imgs"
OUTPUT_DIR = ROOT / "data" / "results" / "yolo_detection"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Fant ikke bildemappen: {IMAGE_DIR}")

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"Fant ingen bilder i: {IMAGE_DIR}")

    model = YOLO("yolo26n.pt")
    results = model.predict(
        source=str(IMAGE_DIR),
        save=True,
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True,
    )

    for result in results:
        detection_count = len(result.boxes) if result.boxes is not None else 0
        print(f"{Path(result.path).name}: {detection_count} deteksjoner")

    print(f"Resultatbilder lagret i: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()