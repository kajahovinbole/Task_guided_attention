
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "exported" / "data"

MODEL_DIR = ROOT / "models" / "hand_detector" / "yolo_v4_tiny_cross_hands"
CFG_PATH = MODEL_DIR / "cross-hands-yolov4-tiny.cfg"
WEIGHTS_PATH = MODEL_DIR / "cross-hands-yolov4-tiny.weights"

OUTPUT_DIR = ROOT / "data" / "yolo_detector" / "hand_detector_test_16okt" / "test_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not CFG_PATH.exists() or not WEIGHTS_PATH.exists():
    raise FileNotFoundError("Fant ikke YOLO-konfigurasjonen eller vektene.")

# Load the pretrained hand detector
net = cv2.dnn.readNetFromDarknet(
    str(CFG_PATH),
    str(WEIGHTS_PATH),
)

# Use CPU for compatibility
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

output_layer_names = net.getUnconnectedOutLayersNames()

# Test images: change these filenames if they differ in your folder
test_images = [
    "scene_02_img_009.jpg",
    "scene_03_img_004.jpeg",
    "scene_01_img_010.jpg",
]

for filename in test_images:
    image_path = IMAGE_DIR / filename

    if not image_path.exists():
        print(f"Fant ikke bildet: {filename}")
        continue

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Kunne ikke lese: {filename}")
        continue

    height, width = image.shape[:2]

    # Model was trained with 416x416 input.
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1 / 255.0,
        size=(416, 416),
        swapRB=True,
        crop=False,
    )

    net.setInput(blob)
    outputs = net.forward(output_layer_names)

    boxes = []
    confidences = []

    for output in outputs:
        for detection in output:
            # YOLO output: center_x, center_y, width, height,
            # objectness, followed by class scores.
            objectness = float(detection[4])
            class_scores = detection[5:]
            class_score = float(class_scores.max())
            confidence = objectness * class_score

            if confidence < 0.25:
                continue

            center_x = float(detection[0]) * width
            center_y = float(detection[1]) * height
            box_width = float(detection[2]) * width
            box_height = float(detection[3]) * height

            x = int(center_x - box_width / 2)
            y = int(center_y - box_height / 2)

            boxes.append([x, y, int(box_width), int(box_height)])
            confidences.append(confidence)

    # Remove duplicate overlapping boxes.
    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        score_threshold=0.25,
        nms_threshold=0.4,
    )

    if len(indices) > 0:
        indices = indices.flatten()
    else:
        indices = []

    print(f"\n{filename}: {len(indices)} hånd(er) funnet")

    for idx in indices:
        x, y, w, h = boxes[idx]
        confidence = confidences[idx]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width - 1, x + w)
        y2 = min(height - 1, y + h)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(
            image,
            f"hand {confidence:.2f}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        print(
            f"  confidence={confidence:.2f}, "
            f"bbox=({x1}, {y1}, {x2}, {y2})"
        )

    cv2.imwrite(str(OUTPUT_DIR / filename), image)

print(f"\nVisualiseringer lagret i: {OUTPUT_DIR}")
