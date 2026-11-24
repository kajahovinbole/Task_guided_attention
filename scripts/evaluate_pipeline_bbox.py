"""Evaluate full-pipeline object selection on COCO-style FiftyOne images.

Named scene frames are evaluated in order with video-mode hand detection and a
persistent temporal selector. Other still images are treated independently.
Per-candidate scores and IoUs are written for debugging.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import re
from dataclasses import asdict
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_LABELS = (
    ROOT / "data" / "3_exported_from_fiftyone" / "1_2_3_andrea_merged"
    / "labels.json"
)
DEFAULT_IMAGE_DIR = DEFAULT_LABELS.parent / "data"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "pipeline_bbox_eval" / "1_2_3_andrea_merged"
HAND_DETECTOR_CANVAS_SIZE = 1280


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the pipeline's selected object box against COCO-style "
            "FiftyOne annotations."
        )
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-images", type=int,
        help="Optional image limit for a short run; uses label-file order.",
    )
    parser.add_argument(
        "--save-overlays", action="store_true",
        help=(
            "Save overlays: ground truth green, sequence choice blue, "
            "framewise choice yellow."
        ),
    )
    return parser.parse_args()


def coco_xywh_to_xyxy(box):
    x, y, width, height = map(float, box)
    return [x, y, x + width, y + height]


def prepare_hand_detector_frame(frame, canvas_size=HAND_DETECTOR_CANVAS_SIZE):
    """Letterbox an image to a fixed size and return its inverse coordinate map."""
    height, width = frame.shape[:2]
    scale = min(canvas_size / width, canvas_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    offset_x = (canvas_size - resized_width) // 2
    offset_y = (canvas_size - resized_height) // 2
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=frame.dtype)
    canvas[
        offset_y:offset_y + resized_height,
        offset_x:offset_x + resized_width,
    ] = resized
    # Use the actual rounded resize dimensions so the inverse is exact.
    transform = {
        "scale_x": resized_width / width,
        "scale_y": resized_height / height,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }
    return canvas, transform


def map_landmarks_to_original(landmarks, transform):
    """Map pixel landmarks from the detector canvas to the source image."""
    if landmarks is None:
        return None
    scale_x = transform["scale_x"]
    scale_y = transform["scale_y"]
    offset_x = transform["offset_x"]
    offset_y = transform["offset_y"]
    return [
        {
            **point,
            "x": int(round((point["x"] - offset_x) / scale_x)),
            "y": int(round((point["y"] - offset_y) / scale_y)),
        }
        for point in landmarks
    ]


def detect_hands_on_fixed_canvas(hand_detector, frame, timestamp_ms=None):
    """Run MediaPipe at one fixed frame size, then restore source coordinates."""
    detector_frame, transform = prepare_hand_detector_frame(frame)
    hands = hand_detector.detect(detector_frame, timestamp_ms or 0)
    for hand in hands:
        hand["landmarks"] = map_landmarks_to_original(hand["landmarks"], transform)
    hand_detector.last_pose_landmarks = map_landmarks_to_original(
        hand_detector.last_pose_landmarks, transform
    )
    return hands


def group_image_records(image_records):
    """Group named scene frames as sequences; treat other images independently."""
    groups = {}
    pattern = re.compile(r"^(scene_\d+)_img_(\d+)", re.IGNORECASE)
    for position, record in enumerate(image_records):
        stem = Path(record["file_name"]).stem
        match = pattern.match(stem)
        if match:
            sequence_id = match.group(1).lower()
            frame_index = int(match.group(2))
            key = ("scene", sequence_id)
            running_mode = "video"
        else:
            sequence_id = f"still_{record['id']}"
            frame_index = None
            key = ("still", "independent_stills")
            running_mode = "image"

        if key not in groups:
            groups[key] = {
                "sequence_id": sequence_id if running_mode == "video" else "independent_stills",
                "running_mode": running_mode,
                "first_position": position,
                "records": [],
            }
        groups[key]["records"].append((frame_index, record))

    ordered_groups = sorted(groups.values(), key=lambda group: group["first_position"])
    for group in ordered_groups:
        if group["running_mode"] == "video":
            group["records"].sort(key=lambda pair: pair[0])
        group["records"] = [record for _, record in group["records"]]
    return ordered_groups


def make_object_selector(config):
    from src.temporal_object_selector import TemporalObjectSelector

    return TemporalObjectSelector(
        score_alpha=config.score_smoothing_alpha,
        bbox_alpha=config.bbox_smoothing_alpha,
        switch_margin=config.track_switch_margin,
        switch_confirmation_frames=config.switch_confirmation_frames,
        match_iou_threshold=config.track_match_iou_threshold,
        max_missing_frames=config.track_max_missing_frames,
        selected_missing_grace_frames=config.selected_missing_grace_frames,
    )


def mean_in_box(grid, box, image_width, image_height):
    """Mean value from a lower-resolution map in source-image bbox coords."""
    if grid is None:
        return None
    grid_height, grid_width = grid.shape[:2]
    x1 = int(np.clip(np.floor(box[0] * grid_width / image_width), 0, grid_width))
    y1 = int(np.clip(np.floor(box[1] * grid_height / image_height), 0, grid_height))
    x2 = int(np.clip(np.ceil(box[2] * grid_width / image_width), 0, grid_width))
    y2 = int(np.clip(np.ceil(box[3] * grid_height / image_height), 0, grid_height))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(grid[y1:y2, x1:x2].mean())


def make_cone_diagnostic(pointing_hand, width, height, config, cone_width, heat_extent):
    """Build the exact low-resolution geometric cone for candidate diagnostics."""
    if pointing_hand is None:
        return None
    from scripts.prior.create_numeric_prior import create_cone_heatmap, get_pointing_data

    pointing = get_pointing_data(
        {"final_landmarks": pointing_hand["landmarks"]}, width, height
    )
    if pointing is None:
        return None
    tip, direction = pointing
    small_width = 448
    small_height = round(small_width * height / width)
    return create_cone_heatmap(
        tip,
        direction,
        small_width,
        small_height,
        cone_length_fraction=config.cone_length,
        cone_width_fraction=cone_width,
        cone_widening=config.cone_widening,
        near_ramp_fraction=config.cone_near_ramp,
        longitudinal_decay_fraction=heat_extent,
    )


def summarize_selection(rows, bbox_key, score_key, iou_key):
    metric_rows = [
        {
            "image_id": row["image_id"],
            "ground_truth_boxes_xyxy": row["ground_truth_boxes_xyxy"],
            "selected_bbox_xyxy": row[bbox_key],
            "selection_score": row[score_key],
        }
        for row in rows
    ]
    thresholds = [round(value, 2) for value in np.arange(0.50, 0.951, 0.05)]
    threshold_metrics = [
        evaluate_at_iou(metric_rows, threshold) for threshold in thresholds
    ]
    by_threshold = {metric["iou_threshold"]: metric for metric in threshold_metrics}
    overlaps = [row["iou"] for row in rows if row[bbox_key] is not None]
    count = len(rows)
    selected_count = len(overlaps)
    return {
        "images_evaluated": count,
        "images_with_selected_box": selected_count,
        "no_selection_rate": (count - selected_count) / count if count else 0.0,
        "mean_iou_all_images_misses_as_zero": float(
            np.mean([row[iou_key] for row in rows])
        ) if count else 0.0,
        "mean_iou_selected_images": float(np.mean(overlaps)) if overlaps else 0.0,
        "hit_rate_iou_50": float(np.mean([
            row[iou_key] >= 0.50 if row[bbox_key] is not None else False
            for row in rows
        ])) if count else 0.0,
        "hit_rate_iou_75": float(np.mean([
            row[iou_key] >= 0.75 if row[bbox_key] is not None else False
            for row in rows
        ])) if count else 0.0,
        "AP50": by_threshold[0.5]["average_precision"],
        "AP75": by_threshold[0.75]["average_precision"],
        "mAP50_95": float(np.mean([
            metric["average_precision"] for metric in threshold_metrics
        ])),
        "metrics_by_iou_threshold": threshold_metrics,
    }


def iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def load_dataset(labels_path, image_dir):
    coco = json.loads(labels_path.read_text(encoding="utf-8"))
    annotations_by_image = {}
    for annotation in coco.get("annotations", []):
        if annotation.get("iscrowd", 0):
            continue
        annotations_by_image.setdefault(annotation["image_id"], []).append(
            annotation
        )

    images = [
        image for image in coco.get("images", [])
        if annotations_by_image.get(image["id"])
    ]
    if not images:
        raise ValueError("Fant ingen bilder med ikke-crowd fasitbokser i labels.")

    category_ids = {
        annotation["category_id"]
        for annotations in annotations_by_image.values()
        for annotation in annotations
    }
    if len(category_ids) != 1:
        raise ValueError(
            "Denne evalueringen forventer én generell fasitklasse for objektet. "
            f"Fant category_id-verdier: {sorted(category_ids)}"
        )
    return images, annotations_by_image, next(iter(category_ids))


def draw_overlay(
    frame, ground_truth_boxes, predicted_box, image_name, score, overlap,
    framewise_box=None,
):
    output = frame.copy()
    for box in ground_truth_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 4)
    if framewise_box is not None:
        x1, y1, x2, y2 = map(int, framewise_box)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 3)
    if predicted_box is not None:
        x1, y1, x2, y2 = map(int, predicted_box)
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 4)
    text = f"{image_name} | IoU {overlap:.3f} | score {score:.3f}" if score is not None else (
        f"{image_name} | no selected box | IoU {overlap:.3f}"
    )
    cv2.putText(
        output, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.rectangle(output, (20, 58), (42, 80), (0, 220, 0), -1)
    cv2.putText(
        output, "Ground truth", (52, 77), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.rectangle(output, (220, 58), (242, 80), (255, 0, 0), -1)
    cv2.putText(
        output, "Pipeline selection", (252, 77), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.rectangle(output, (500, 58), (522, 80), (0, 255, 255), -1)
    cv2.putText(
        output, "Framewise choice", (532, 77), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2, cv2.LINE_AA,
    )
    max_width = 1800
    if output.shape[1] > max_width:
        scale = max_width / output.shape[1]
        output = cv2.resize(
            output, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    return output


def evaluate_at_iou(rows, threshold):
    """Compute image-level P/R and 101-point interpolated AP for one class."""
    predictions = [row for row in rows if row["selected_bbox_xyxy"] is not None]
    predictions.sort(key=lambda row: row["selection_score"], reverse=True)

    ground_truth_count = sum(len(row["ground_truth_boxes_xyxy"]) for row in rows)
    matched_by_image = {}
    true_positives = []
    false_positives = []
    tp_total = 0

    for prediction in predictions:
        image_id = prediction["image_id"]
        gt_boxes = prediction["ground_truth_boxes_xyxy"]
        used = matched_by_image.setdefault(image_id, set())
        overlaps = [
            (iou(prediction["selected_bbox_xyxy"], gt_box), gt_index)
            for gt_index, gt_box in enumerate(gt_boxes)
            if gt_index not in used
        ]
        best_overlap, best_index = max(overlaps, default=(0.0, None))
        if best_overlap >= threshold:
            used.add(best_index)
            true_positives.append(1)
            false_positives.append(0)
            tp_total += 1
        else:
            true_positives.append(0)
            false_positives.append(1)

    precision = tp_total / len(predictions) if predictions else 0.0
    recall = tp_total / ground_truth_count if ground_truth_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    if not predictions or ground_truth_count == 0:
        average_precision = 0.0
    else:
        tp_cumulative = np.cumsum(true_positives)
        fp_cumulative = np.cumsum(false_positives)
        recalls = tp_cumulative / ground_truth_count
        precisions = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1)
        # Monotonic precision envelope, then COCO-style 101-point interpolation.
        for index in range(len(precisions) - 2, -1, -1):
            precisions[index] = max(precisions[index], precisions[index + 1])
        recall_samples = np.linspace(0.0, 1.0, 101)
        interpolated = [
            float(precisions[recalls >= recall].max())
            if np.any(recalls >= recall) else 0.0
            for recall in recall_samples
        ]
        average_precision = float(np.mean(interpolated))

    return {
        "iou_threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp_total,
        "false_positives": len(predictions) - tp_total,
        "false_negatives": ground_truth_count - tp_total,
        "average_precision": average_precision,
    }


def run_evaluation(args):
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images må være større enn null.")
    if not args.labels.is_file():
        raise FileNotFoundError(f"Fant ikke COCO-labels: {args.labels}")
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Fant ikke bildemappen: {args.image_dir}")

    from configs.full_pipeline import DEFAULT_CONFIG
    from scripts.full_pipeline import (
        ROOT as PIPELINE_ROOT,
        _pointing_hands,
        choose_device,
        create_focus_map,
        resize_saliency,
        score_box_with_saliency,
        score_detections,
    )
    from scripts.absvit_prior import load_abs_vit_model
    from scripts.yolo_detection import detect_objects, load_yolo_model
    from src.reach_estimator import (
        ReachScoreSmoother,
        adaptive_heat_extent,
        make_reach_score,
        measure_arm,
        measure_torso_fraction,
        person_scale_width_factor,
    )
    from src.video_hand_detector import VideoHandDetector

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlays"
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    image_records, annotations_by_image, category_id = load_dataset(
        args.labels, args.image_dir
    )
    if args.max_images is not None:
        image_records = image_records[:args.max_images]
    groups = group_image_records(image_records)

    config = DEFAULT_CONFIG
    device = choose_device(config.device)
    model = load_abs_vit_model(config.checkpoint_path, device)
    yolo_model = load_yolo_model(config.yolo_model)
    rows = []
    rows = []
    framewise_predictions = []
    sequence_predictions = []
    processed_count = 0
    scene_group_count = sum(group["running_mode"] == "video" for group in groups)
    still_count = sum(
        len(group["records"])
        for group in groups if group["running_mode"] == "image"
    )

    def selected_info(selected):
        if selected is None:
            return {
                "bbox": None, "score": None, "confidence": None,
                "saliency": None, "class_name": None, "detection_index": None,
                "tracked_missing": False,
            }
        bbox = selected.get("smoothed_bbox_xyxy", selected.get("bbox_xyxy"))
        return {
            "bbox": list(bbox) if bbox is not None else None,
            "score": (
                float(selected["smoothed_selection_score"])
                if selected.get("smoothed_selection_score") is not None else
                float(selected["selection_score"])
                if selected.get("selection_score") is not None else None
            ),
            "confidence": float(selected["confidence"]),
            "saliency": float(selected["saliency_score"]),
            "class_name": selected["class_name"],
            "detection_index": selected.get("index"),
            "tracked_missing": bool(selected.get("tracked_missing", False)),
        }

    def append_prediction(target, image_id, category, info):
        if info["bbox"] is None:
            return
        x1, y1, x2, y2 = info["bbox"]
        target.append({
            "image_id": image_id,
            "category_id": category,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": info["score"],
        })

    for group in groups:
        hand_detector = VideoHandDetector(
            PIPELINE_ROOT / "models" / "mediapipe",
            crop_size=config.hand_crop_size,
            agreement_threshold=config.hand_agreement_threshold,
            min_detection_confidence=config.hand_detection_confidence,
            running_mode=group["running_mode"],
        )
        sequence_selector = (
            make_object_selector(config)
            if group["running_mode"] == "video" else None
        )
        sequence_reach_smoother = (
            ReachScoreSmoother(
                alpha=config.reach_score_alpha,
                missing_hold_frames=config.reach_score_hold_missing_frames,
            )
            if group["running_mode"] == "video" else None
        )
        last_timestamp_ms = -1
        try:
            for image_record in group["records"]:
                processed_count += 1
                image_path = args.image_dir / image_record["file_name"]
                frame = cv2.imread(str(image_path))
                if frame is None:
                    raise FileNotFoundError(f"Kunne ikke lese bilde: {image_path}")
                height, width = frame.shape[:2]
                annotations = annotations_by_image[image_record["id"]]
                gt_boxes = [
                    coco_xywh_to_xyxy(annotation["bbox"])
                    for annotation in annotations
                ]

                match = re.match(
                    r"^(scene_\d+)_img_(\d+)",
                    Path(image_record["file_name"]).stem,
                    re.IGNORECASE,
                )
                frame_number = int(match.group(2)) if match else None
                timestamp_ms = None
                if group["running_mode"] == "video":
                    timestamp_ms = max(
                        last_timestamp_ms + 1,
                        int(frame_number * 1000 / 30),
                    )
                    last_timestamp_ms = timestamp_ms

                hands = detect_hands_on_fixed_canvas(
                    hand_detector, frame, timestamp_ms
                )
                pointing_hand = _pointing_hands(hands, config.min_pointing_score)

                heat_extent = None
                if config.use_reach_adaptive_heat:
                    arm = measure_arm(
                        hand_detector.last_pose_landmarks,
                        pointing_hand["side"] if pointing_hand else None,
                        hand_detector.last_pose_visibility,
                    )
                    reach_score = make_reach_score(
                        arm, config.reach_elbow_weight, config.reach_shoulder_weight
                    )
                    reach_smoother = (
                        sequence_reach_smoother
                        if sequence_reach_smoother is not None else
                        ReachScoreSmoother(
                            alpha=config.reach_score_alpha,
                            missing_hold_frames=config.reach_score_hold_missing_frames,
                        )
                    )
                    heat_extent = adaptive_heat_extent(
                        reach_smoother.update(reach_score),
                        config.reach_near_heat_extent,
                        config.reach_far_heat_extent,
                    )

                cone_width = config.cone_width
                if config.use_person_scale_adaptive_cone:
                    torso_fraction = measure_torso_fraction(
                        hand_detector.last_pose_landmarks,
                        height,
                        hand_detector.last_pose_visibility,
                    )
                    width_factor = person_scale_width_factor(
                        torso_fraction,
                        config.person_scale_reference_fraction,
                        config.person_scale_min_width_factor,
                        config.person_scale_max_width_factor,
                    )
                    if width_factor is not None:
                        cone_width *= width_factor

                focus_small, focus_mode = create_focus_map(
                    model,
                    frame,
                    pointing_hand,
                    device,
                    config,
                    heat_extent=heat_extent,
                    cone_width=cone_width,
                )
                focus_map = resize_saliency(focus_small, frame.shape)
                cone_map = make_cone_diagnostic(
                    pointing_hand, width, height, config, cone_width, heat_extent
                )

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

                candidates = []
                for detection in detections:
                    box = detection["bbox_xyxy"]
                    cone_mean = mean_in_box(cone_map, box, width, height)
                    cone_coverage = (
                        mean_in_box((cone_map > 0.1).astype(np.float32), box, width, height)
                        if cone_map is not None else None
                    )
                    candidates.append({
                        "detection_index": detection["index"],
                        "class_name": detection["class_name"],
                        "confidence": detection["confidence"],
                        "saliency_score": detection["saliency_score"],
                        "selection_score": detection["selection_score"],
                        "eligible": detection["eligible"],
                        "bbox_xyxy": box,
                        "iou_to_ground_truth": max(
                            (iou(box, gt_box) for gt_box in gt_boxes), default=0.0
                        ),
                        "cone_mean_heat": cone_mean,
                        "cone_coverage_fraction": cone_coverage,
                    })
                candidates.sort(
                    key=lambda item: (
                        item["selection_score"] is not None,
                        item["selection_score"] or 0.0,
                    ),
                    reverse=True,
                )

                pointing_available = focus_mode == "pointing_prior"
                framewise_selected = make_object_selector(config).select(
                    copy.deepcopy(detections), pointing_available
                )
                if sequence_selector is not None:
                    sequence_selected = sequence_selector.select(
                        copy.deepcopy(detections), pointing_available
                    )
                else:
                    sequence_selected = framewise_selected

                framewise = selected_info(framewise_selected)
                sequence = selected_info(sequence_selected)
                for candidate in candidates:
                    candidate["selected_framewise"] = (
                        not framewise["tracked_missing"]
                        and candidate["detection_index"] == framewise["detection_index"]
                    )
                    candidate["selected_sequence"] = (
                        not sequence["tracked_missing"]
                        and candidate["detection_index"] == sequence["detection_index"]
                    )
                framewise_iou = max(
                    (iou(framewise["bbox"], box) for box in gt_boxes), default=0.0
                )
                sequence_iou = max(
                    (iou(sequence["bbox"], box) for box in gt_boxes), default=0.0
                )
                row = {
                    "image_id": image_record["id"],
                    "image_name": image_record["file_name"],
                    "sequence_id": group["sequence_id"],
                    "sequence_frame_index": frame_number,
                    "ground_truth_count": len(gt_boxes),
                    "ground_truth_boxes_xyxy": gt_boxes,
                    "candidates": candidates,
                    "framewise_bbox_xyxy": framewise["bbox"],
                    "framewise_selection_score": framewise["score"],
                    "framewise_yolo_class": framewise["class_name"],
                    "framewise_iou": framewise_iou,
                    "sequence_bbox_xyxy": sequence["bbox"],
                    "sequence_selection_score": sequence["score"],
                    "sequence_yolo_class": sequence["class_name"],
                    "sequence_tracked_missing": sequence["tracked_missing"],
                    "sequence_iou": sequence_iou,
                    "focus_mode": focus_mode,
                }
                # Keep the original output column names as aliases for the
                # sequence-aware result shown by the actual video pipeline.
                row.update({
                    "selected_bbox_xyxy": sequence["bbox"],
                    "selection_score": sequence["score"],
                    "selected_yolo_class": sequence["class_name"],
                    "iou": sequence_iou,
                    "hit_iou_50": sequence_iou >= 0.50,
                    "hit_iou_75": sequence_iou >= 0.75,
                })
                rows.append(row)

                append_prediction(
                    framewise_predictions, image_record["id"], category_id, framewise
                )
                append_prediction(
                    sequence_predictions, image_record["id"], category_id, sequence
                )

                if args.save_overlays:
                    overlay = draw_overlay(
                        frame,
                        gt_boxes,
                        sequence["bbox"],
                        image_record["file_name"],
                        sequence["score"],
                        sequence_iou,
                        framewise_box=framewise["bbox"],
                    )
                    overlay_name = (
                        f"{image_record['id']}_{Path(image_record['file_name']).name}"
                    )
                    cv2.imwrite(str(overlay_dir / overlay_name), overlay)

                if processed_count % 10 == 0 or processed_count == len(image_records):
                    print(f"Evaluert {processed_count}/{len(image_records)} bilder")
        finally:
            active_exception = sys.exc_info()[0] is not None
            try:
                hand_detector.close()
            except Exception:
                if not active_exception:
                    raise

    framewise_metrics = summarize_selection(
        rows,
        "framewise_bbox_xyxy",
        "framewise_selection_score",
        "framewise_iou",
    )
    sequence_metrics = summarize_selection(
        rows,
        "sequence_bbox_xyxy",
        "sequence_selection_score",
        "sequence_iou",
    )
    metrics = {
        "labels_path": str(args.labels.resolve()),
        "image_dir": str(args.image_dir.resolve()),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "images_evaluated": len(rows),
        "scene_sequences_evaluated": scene_group_count,
        "independent_still_images": still_count,
        "framewise_selection": framewise_metrics,
        "sequence_selection": sequence_metrics,
        # Keep the top-level metrics as the sequence-aware result for older
        # tooling that reads these keys directly.
        **sequence_metrics,
        "note": (
            "scene_XX_img_YYY groups use video-mode MediaPipe and preserve "
            "reach and object-selector state within each scene. Other images "
            "are independent stills. Framewise selection is also reported. "
            "AP uses one generic object class, selection_score, and 101-point "
            "interpolation; each image contributes at most one prediction."
        ),
    }

    csv_path = args.output_dir / "per_image.csv"
    fieldnames = [
        "image_id", "image_name", "sequence_id", "sequence_frame_index",
        "selected_yolo_class", "selection_score", "iou", "hit_iou_50",
        "hit_iou_75", "framewise_yolo_class", "framewise_selection_score",
        "framewise_iou", "sequence_yolo_class", "sequence_selection_score",
        "sequence_iou", "sequence_tracked_missing", "focus_mode",
        "candidates", "selected_bbox_xyxy", "framewise_bbox_xyxy",
        "sequence_bbox_xyxy", "ground_truth_boxes_xyxy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row[key])
                if isinstance(row[key], (list, dict)) else row[key]
                for key in fieldnames
            })

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "predictions_coco.json").write_text(
        json.dumps(sequence_predictions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "predictions_framewise_coco.json").write_text(
        json.dumps(framewise_predictions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "framewise_selection": {
            key: framewise_metrics[key]
            for key in ("hit_rate_iou_50", "AP50", "AP75", "mAP50_95")
        },
        "sequence_selection": {
            key: sequence_metrics[key]
            for key in ("hit_rate_iou_50", "AP50", "AP75", "mAP50_95")
        },
    }, indent=2))
    print(f"Per-image kandidater og metrics lagret i {args.output_dir}")


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
