"""Run video hand detection, prior-conditioned AbSViT, and YOLO together."""

from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from scripts.absvit_prior import (
    load_abs_vit_model,
    make_patch_prior,
    preprocess_rgb,
    prior_to_mask,
    run_pointing,
)
from scripts.prior.create_numeric_prior import get_pointing_data
from scripts.prior.score_pointing_hand import (
    HAND_CONNECTIONS,
    THRESHOLD,
    score_pointing_hand,
)
from scripts.yolo_detection import detect_objects, load_yolo_model
from src.video_hand_detector import VideoHandDetector


ROOT = Path(__file__).resolve().parents[1]


def choose_device(requested):
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA ble valgt, men er ikke tilgjengelig.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS ble valgt, men er ikke tilgjengelig.")
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_frame_tensor(frame_bgr, device):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return preprocess_rgb(rgb).to(device)


def normalize_saliency(token_map):
    values = token_map.detach().float().cpu().numpy()
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


def create_saliency_map(model, frame_bgr, pointing_hand, device):
    height, width = frame_bgr.shape[:2]
    hand_data = {"final_landmarks": pointing_hand["landmarks"]}
    pointing = get_pointing_data(hand_data, width, height)
    if pointing is None:
        return None

    tip, direction = pointing
    _, patch_prior = make_patch_prior(tip, direction, width, height)
    pointing_mask = prior_to_mask(
        patch_prior, model.num_prefix_tokens
    ).to(device)
    image_tensor = prepare_frame_tensor(frame_bgr, device)

    with torch.no_grad():
        features, _, _ = model.forward_features(image_tensor)
        token_map = run_pointing(
            model, image_tensor, features, pointing_mask
        )

    return normalize_saliency(token_map)


def resize_saliency(saliency, frame_shape):
    if saliency is None:
        return None
    height, width = frame_shape[:2]
    resized = cv2.resize(saliency, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(resized, 0.0, 1.0)


def score_box_with_saliency(saliency, box):
    if saliency is None:
        return None
    height, width = saliency.shape[:2]
    x1, y1, x2, y2 = box
    x1 = int(np.clip(np.floor(x1), 0, width))
    y1 = int(np.clip(np.floor(y1), 0, height))
    x2 = int(np.clip(np.ceil(x2), 0, width))
    y2 = int(np.clip(np.ceil(y2), 0, height))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(saliency[y1:y2, x1:x2].mean())


def choose_detection(detections, min_confidence, saliency):
    eligible = []
    for detection in detections:
        detection["saliency_score"] = score_box_with_saliency(
            saliency, detection["bbox_xyxy"]
        )
        detection["eligible"] = (
            detection["confidence"] >= min_confidence
            and detection["saliency_score"] is not None
        )
        detection["selected"] = False
        if detection["eligible"]:
            eligible.append(detection)

    if not eligible:
        return None

    # Saliency ranks candidates; YOLO confidence breaks exact ties.
    selected = max(
        eligible,
        key=lambda item: (item["saliency_score"], item["confidence"]),
    )
    selected["selected"] = True
    return selected


def draw_hand_landmarks(image, hands, selected_side):
    height, width = image.shape[:2]
    for hand in hands:
        points = hand["landmarks"]
        color = (0, 255, 0) if hand["side"] == selected_side else (255, 180, 0)
        for start, end in HAND_CONNECTIONS:
            p1 = (int(points[start]["x"]), int(points[start]["y"]))
            p2 = (int(points[end]["x"]), int(points[end]["y"]))
            if all(0 <= x < width and 0 <= y < height for x, y in (p1, p2)):
                cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
        for point in points:
            x, y = int(point["x"]), int(point["y"])
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(image, (x, y), 3, color, -1)


def draw_saliency_overlay(image, saliency):
    if saliency is None:
        return image
    colored = cv2.applyColorMap(
        np.uint8(np.clip(saliency, 0.0, 1.0) * 255),
        cv2.COLORMAP_JET,
    )
    return cv2.addWeighted(image, 0.55, colored, 0.45, 0)


def draw_selected_view(frame, selected):
    output = frame.copy()
    if selected is None:
        return output
    x1, y1, x2, y2 = map(int, selected["bbox_xyxy"])
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 3)
    label = f"{selected['class_name']} {selected['confidence']:.2f}"
    cv2.putText(
        output, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 220, 0), 2, cv2.LINE_AA,
    )
    return output


def draw_verification_view(frame, detections, hands, selected_hand, saliency):
    output = draw_saliency_overlay(frame.copy(), saliency)
    draw_hand_landmarks(output, hands, selected_hand)
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox_xyxy"])
        color = (0, 220, 0) if detection["selected"] else (0, 0, 255)
        focus = detection["saliency_score"]
        focus_text = "n/a" if focus is None else f"{focus:.2f}"
        label = (
            f"{detection['class_name']} conf={detection['confidence']:.2f} "
            f"focus={focus_text}"
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2, cv2.LINE_AA,
        )

    if selected_hand is None:
        cv2.putText(
            output, "NO POINTING HAND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 255), 2, cv2.LINE_AA,
        )
    return output


def _pointing_hands(hands, min_pointing_score):
    scored = []
    for hand in hands:
        score = score_pointing_hand(hand["landmarks"])
        hand["pointing_score"] = score["score"]
        hand["pointing_components"] = score["components"]
        if score["score"] >= min_pointing_score:
            scored.append(hand)
    if not scored:
        return None
    return max(scored, key=lambda hand: hand["pointing_score"])


def _source_label(source):
    if isinstance(source, int):
        return f"camera_{source}"
    return Path(source).stem


def run_full_pipeline(
    source,
    output_dir=None,
    checkpoint_path=None,
    yolo_model_name="yolo26n.pt",
    min_pointing_score=THRESHOLD,
    min_yolo_confidence=0.25,
    yolo_inference_confidence=0.05,
    device_name="auto",
    show=False,
    max_frames=None,
):
    device = choose_device(device_name)
    if checkpoint_path is None:
        checkpoint_path = ROOT / "checkpoints" / "absvit_tiny_patch8_224_gap.pth"
    if output_dir is None:
        output_dir = ROOT / "results" / "video_pipeline" / _source_label(source)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    abs_vit = load_abs_vit_model(checkpoint_path, device)
    yolo_model = load_yolo_model(yolo_model_name)
    hand_detector = VideoHandDetector(ROOT / "models" / "mediapipe")

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        hand_detector.close()
        raise RuntimeError(f"Kunne ikke åpne videokilde: {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_path = output_dir / "selected_object.mp4"
    verification_path = output_dir / "verification.mp4"
    records_path = output_dir / "frames.jsonl"

    selected_writer = None
    verification_writer = None
    frame_index = 0
    started = time.time()
    last_timestamp_ms = -1

    try:
        with records_path.open("w", encoding="utf-8") as records_file:
            records_file.write(json.dumps({
                "type": "metadata",
                "source": str(source),
                "fps": fps,
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "yolo_model": yolo_model_name,
                "min_pointing_score": min_pointing_score,
                "min_yolo_confidence": min_yolo_confidence,
                "selection_rule": "highest mean AbSViT saliency inside eligible YOLO boxes; YOLO confidence breaks ties",
            }, ensure_ascii=False) + "\n")

            while max_frames is None or frame_index < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                if selected_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    selected_writer = cv2.VideoWriter(
                        str(selected_path), fourcc, fps, (width, height)
                    )
                    verification_writer = cv2.VideoWriter(
                        str(verification_path), fourcc, fps, (width, height)
                    )
                    if not selected_writer.isOpened() or not verification_writer.isOpened():
                        raise RuntimeError("Kunne ikke opprette output-videoene.")

                timestamp_ms = max(
                    last_timestamp_ms + 1,
                    int(round(frame_index * 1000.0 / fps)),
                )
                last_timestamp_ms = timestamp_ms

                hands = hand_detector.detect(frame, timestamp_ms)
                pointing_hand = _pointing_hands(hands, min_pointing_score)
                saliency_small = (
                    create_saliency_map(abs_vit, frame, pointing_hand, device)
                    if pointing_hand is not None else None
                )
                saliency = resize_saliency(saliency_small, frame.shape)

                detections = detect_objects(
                    yolo_model, frame, yolo_inference_confidence
                )
                selected = choose_detection(
                    detections, min_yolo_confidence, saliency
                )

                selected_view = draw_selected_view(frame, selected)
                verification_view = draw_verification_view(
                    frame,
                    detections,
                    hands,
                    pointing_hand["side"] if pointing_hand else None,
                    saliency,
                )
                selected_writer.write(selected_view)
                verification_writer.write(verification_view)

                record = {
                    "type": "frame",
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "selected_hand": (
                        {
                            "side": pointing_hand["side"],
                            "score": pointing_hand["pointing_score"],
                            "decision": pointing_hand["decision"],
                        }
                        if pointing_hand else None
                    ),
                    "hands": [
                        {
                            "side": hand["side"],
                            "decision": hand["decision"],
                            "pointing_score": hand.get("pointing_score"),
                            "landmarks": hand["landmarks"],
                        }
                        for hand in hands
                    ],
                    "detections": detections,
                    "selected_detection_index": (
                        selected["index"] if selected else None
                    ),
                }
                records_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                if show:
                    cv2.imshow("Selected object", selected_view)
                    cv2.imshow("Pipeline verification", verification_view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_index += 1
                if frame_index % 30 == 0:
                    total = f"/{frame_count}" if frame_count > 0 else ""
                    print(f"Behandlet {frame_index}{total} frames")
    finally:
        capture.release()
        if selected_writer is not None:
            selected_writer.release()
        if verification_writer is not None:
            verification_writer.release()
        hand_detector.close()
        if show:
            cv2.destroyAllWindows()

    if frame_index == 0:
        raise RuntimeError("Videokilden returnerte ingen frames.")

    elapsed = time.time() - started
    print(f"Ferdig: {frame_index} frames på {elapsed:.1f} sekunder")
    print(f"Valgt objekt-video: {selected_path}")
    print(f"Verifiseringsvideo: {verification_path}")
    print(f"Frame-data: {records_path}")

