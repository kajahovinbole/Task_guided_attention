"""Run video hand detection, prior-conditioned AbSViT, and YOLO together."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from scripts.absvit_prior import (
    get_token_norm_map,
    load_abs_vit_model,
    make_patch_prior,
    preprocess_rgb,
    prior_to_mask,
    run_pointing,
)
from scripts.prior.create_numeric_prior import get_pointing_data
from scripts.prior.score_pointing_hand import (
    HAND_CONNECTIONS,
    score_pointing_hand,
)
from scripts.yolo_detection import detect_objects, load_yolo_model
from configs.full_pipeline import DEFAULT_CONFIG, PipelineConfig
from src.reach_estimator import (
    ReachScoreSmoother,
    adaptive_heat_extent,
    make_reach_score,
    measure_arm,
    measure_torso_fraction,
    person_scale_width_factor,
)
from src.temporal_object_selector import TemporalObjectSelector
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


def create_focus_map(
    model, frame_bgr, pointing_hand, device, config, heat_extent=None,
    cone_width=None,
):
    """Create prior-conditioned focus, or bottom-up focus if no hand points."""
    height, width = frame_bgr.shape[:2]
    image_tensor = prepare_frame_tensor(frame_bgr, device)
    with torch.no_grad():
        if pointing_hand is None:
            token_map = get_token_norm_map(
                model,
                image_tensor,
                attention_layer_from_end=config.attention_layer_from_end,
            )
            mode = "bottom_up"
        else:
            hand_data = {"final_landmarks": pointing_hand["landmarks"]}
            pointing = get_pointing_data(hand_data, width, height)
            if pointing is None:
                token_map = get_token_norm_map(
                    model,
                    image_tensor,
                    attention_layer_from_end=config.attention_layer_from_end,
                )
                mode = "bottom_up"
            else:
                tip, direction = pointing
                _, patch_prior = make_patch_prior(
                    tip,
                    direction,
                    width,
                    height,
                    cone_length=config.cone_length,
                    cone_width=(
                        config.cone_width if cone_width is None else cone_width
                    ),
                    cone_widening=config.cone_widening,
                    cone_near_ramp=config.cone_near_ramp,
                    longitudinal_decay_fraction=heat_extent,
                )
                pointing_mask = prior_to_mask(
                    patch_prior,
                    model.num_prefix_tokens,
                    prior_scale=config.prior_scale,
                    cls_prior=config.cls_prior,
                ).to(device)
                features, _, _ = model.forward_features(image_tensor)
                token_map = run_pointing(
                    model,
                    image_tensor,
                    features,
                    pointing_mask,
                    attention_layer_from_end=config.attention_layer_from_end,
                )
                mode = "pointing_prior"

    return normalize_saliency(token_map), mode


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


def score_detections(
    detections,
    min_confidence,
    saliency,
    pointing_available,
    saliency_weight,
    confidence_weight,
):
    weight_total = saliency_weight + confidence_weight
    for detection in detections:
        detection["saliency_score"] = score_box_with_saliency(
            saliency, detection["bbox_xyxy"]
        )
        detection["eligible"] = (
            detection["confidence"] >= min_confidence
            and detection["saliency_score"] is not None
            and pointing_available
        )
        detection["selected"] = False
        detection["selection_score"] = (
            saliency_weight * detection["saliency_score"]
            + confidence_weight * detection["confidence"]
        ) / weight_total if detection["saliency_score"] is not None else None


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


def draw_saliency_overlay(image, saliency, opacity):
    if saliency is None:
        return image
    colored = cv2.applyColorMap(
        np.uint8(np.clip(saliency, 0.0, 1.0) * 255),
        cv2.COLORMAP_JET,
    )
    return cv2.addWeighted(image, 1.0 - opacity, colored, opacity, 0)


def draw_selected_view(frame, selected):
    output = frame.copy()
    if selected is None:
        return output
    box = selected.get("smoothed_bbox_xyxy", selected["bbox_xyxy"])
    x1, y1, x2, y2 = map(int, box)
    color = (0, 220, 255) if selected.get("tracked_missing") else (0, 220, 0)
    cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
    suffix = " held" if selected.get("tracked_missing") else ""
    label = f"{selected['class_name']} {selected['confidence']:.2f}{suffix}"
    cv2.putText(
        output, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, color, 2, cv2.LINE_AA,
    )
    return output


def draw_verification_view(
    frame,
    detections,
    hands,
    selected_hand,
    selected_detection,
    focus_map,
    focus_mode,
    config,
    reach_score=None,
    heat_extent=None,
    cone_width_factor=None,
):
    output = draw_saliency_overlay(
        frame.copy(), focus_map, config.heatmap_opacity
    )
    draw_hand_landmarks(output, hands, selected_hand)
    for detection in detections:
        box = (
            detection.get("smoothed_bbox_xyxy", detection["bbox_xyxy"])
            if detection["selected"] else detection["bbox_xyxy"]
        )
        x1, y1, x2, y2 = map(int, box)
        color = (0, 220, 0) if detection["selected"] else (0, 0, 255)
        focus = detection["saliency_score"]
        focus_text = "n/a" if focus is None else f"{focus:.2f}"
        label = (
            f"{detection['class_name']} conf={detection['confidence']:.2f} "
            f"focus={focus_text} "
            f"score={detection['selection_score']:.2f}"
            if detection["selection_score"] is not None else
            f"{detection['class_name']} conf={detection['confidence']:.2f} "
            f"focus={focus_text} score=n/a"
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
    if selected_detection is not None and selected_detection.get("tracked_missing"):
        x1, y1, x2, y2 = map(
            int, selected_detection.get(
                "smoothed_bbox_xyxy", selected_detection["bbox_xyxy"]
            )
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 255), 3)
        cv2.putText(
            output,
            f"{selected_detection['class_name']} held",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        output,
        f"FOCUS MAP: {focus_mode.replace('_', ' ').upper()}",
        (20, output.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if config.use_reach_adaptive_heat:
        reach_text = "n/a" if reach_score is None else f"{reach_score:.2f}"
        extent_text = "n/a" if heat_extent is None else f"{heat_extent:.2f}x"
        cv2.putText(
            output,
            f"REACH: {reach_text} | HEAT EXTENT: {extent_text}",
            (20, output.shape[0] - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if cone_width_factor is not None:
        cv2.putText(
            output,
            f"CONE WIDTH SCALE: {cone_width_factor:.2f}x",
            (20, output.shape[0] - (75 if config.use_reach_adaptive_heat else 45)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
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
    config: PipelineConfig = DEFAULT_CONFIG,
    output_dir=None,
    show=None,
    max_frames=None,
):
    config = config or DEFAULT_CONFIG
    device = choose_device(config.device)
    if output_dir is None:
        output_dir = config.output_root / _source_label(source)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    show = config.show_windows if show is None else show
    max_frames = config.max_frames if max_frames is None else max_frames
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames må være større enn null.")

    abs_vit = load_abs_vit_model(config.checkpoint_path, device)
    yolo_model = load_yolo_model(config.yolo_model)
    hand_detector = VideoHandDetector(
        ROOT / "models" / "mediapipe",
        crop_size=config.hand_crop_size,
        agreement_threshold=config.hand_agreement_threshold,
        min_detection_confidence=config.hand_detection_confidence,
    )
    object_selector = TemporalObjectSelector(
        score_alpha=config.score_smoothing_alpha,
        bbox_alpha=config.bbox_smoothing_alpha,
        switch_margin=config.track_switch_margin,
        switch_confirmation_frames=config.switch_confirmation_frames,
        match_iou_threshold=config.track_match_iou_threshold,
        max_missing_frames=config.track_max_missing_frames,
        selected_missing_grace_frames=config.selected_missing_grace_frames,
    )
    reach_smoother = ReachScoreSmoother(
        alpha=config.reach_score_alpha,
        missing_hold_frames=config.reach_score_hold_missing_frames,
    )

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
                "checkpoint": str(Path(config.checkpoint_path).resolve()),
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in asdict(config).items()
                },
                "selection_rule": "weighted mean of normalized AbSViT saliency and YOLO confidence; pointing hand and YOLO confidence threshold are required",
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
                pointing_hand = _pointing_hands(
                    hands, config.min_pointing_score
                )
                arm_measurements = None
                reach_score_raw = None
                reach_score_smoothed = None
                heat_extent = None
                torso_fraction = None
                cone_width_factor = None
                cone_width = config.cone_width
                if config.use_reach_adaptive_heat:
                    arm_measurements = measure_arm(
                        hand_detector.last_pose_landmarks,
                        pointing_hand["side"] if pointing_hand else None,
                        hand_detector.last_pose_visibility,
                    )
                    reach_score_raw = make_reach_score(
                        arm_measurements,
                        config.reach_elbow_weight,
                        config.reach_shoulder_weight,
                    )
                    reach_score_smoothed = reach_smoother.update(reach_score_raw)
                    heat_extent = adaptive_heat_extent(
                        reach_score_smoothed,
                        config.reach_near_heat_extent,
                        config.reach_far_heat_extent,
                    )
                if config.use_person_scale_adaptive_cone:
                    torso_fraction = measure_torso_fraction(
                        hand_detector.last_pose_landmarks,
                        frame.shape[0],
                        hand_detector.last_pose_visibility,
                    )
                    cone_width_factor = person_scale_width_factor(
                        torso_fraction,
                        config.person_scale_reference_fraction,
                        config.person_scale_min_width_factor,
                        config.person_scale_max_width_factor,
                    )
                    if cone_width_factor is not None:
                        cone_width *= cone_width_factor

                focus_small, focus_mode = create_focus_map(
                    abs_vit,
                    frame,
                    pointing_hand,
                    device,
                    config,
                    heat_extent=heat_extent,
                    cone_width=cone_width,
                )
                focus_map = resize_saliency(focus_small, frame.shape)

                detections = detect_objects(
                    yolo_model,
                    frame,
                    config.yolo_inference_confidence,
                    image_size=config.yolo_image_size,
                )
                score_detections(
                    detections,
                    config.min_yolo_confidence,
                    focus_map,
                    pointing_available=focus_mode == "pointing_prior",
                    saliency_weight=config.saliency_weight,
                    confidence_weight=config.yolo_confidence_weight,
                )
                selected = object_selector.select(
                    detections,
                    pointing_available=focus_mode == "pointing_prior",
                )

                selected_view = draw_selected_view(frame, selected)
                verification_view = draw_verification_view(
                    frame,
                    detections,
                    hands,
                    pointing_hand["side"] if pointing_hand else None,
                    selected,
                    focus_map,
                    focus_mode,
                    config,
                    reach_score=reach_score_smoothed,
                    heat_extent=heat_extent,
                    cone_width_factor=cone_width_factor,
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
                    "focus_mode": focus_mode,
                    "reach_adaptive_heat_enabled": config.use_reach_adaptive_heat,
                    "arm_measurements": arm_measurements,
                    "reach_score_raw": reach_score_raw,
                    "reach_score_smoothed": reach_score_smoothed,
                    "cone_length": config.cone_length,
                    "heat_extent_fraction_of_cone": heat_extent,
                    "person_scale_adaptive_cone_enabled": (
                        config.use_person_scale_adaptive_cone
                    ),
                    "torso_fraction_of_frame_height": torso_fraction,
                    "cone_width_factor": cone_width_factor,
                    "cone_width": cone_width,
                    "selected_track_id": (
                        selected["track_id"] if selected else None
                    ),
                    "selected_tracked_missing": bool(
                        selected and selected.get("tracked_missing")
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
                        selected["index"]
                        if selected and not selected.get("tracked_missing") else None
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

