from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "funker_bra_20okt" / "imgs"
OUTPUT_DIR = ROOT / "data" / "results" / "yolo_detection"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_yolo_model(model_name="yolo26n.pt"):
    """Create the shared Ultralytics model used by image and video runs."""
    return YOLO(model_name)


def detect_objects(model, frame_bgr, confidence=0.05):
    """Return frame detections in the format consumed by full_pipeline.py."""
    result = model.predict(
        source=frame_bgr,
        conf=confidence,
        verbose=False,
    )[0]
    detections = []
    if result.boxes is None:
        return detections

    boxes = result.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    for index, (box, score, class_id) in enumerate(
        zip(xyxy, confidences, class_ids)
    ):
        detections.append({
            "index": index,
            "class_id": int(class_id),
            "class_name": str(result.names[int(class_id)]),
            "confidence": float(score),
            "bbox_xyxy": [float(value) for value in box],
        })
    return detections


def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Fant ikke bildemappen: {IMAGE_DIR}")

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"Fant ingen bilder i: {IMAGE_DIR}")

    model = load_yolo_model()
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
