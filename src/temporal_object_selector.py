"""Track YOLO detections and stabilize object choice across video frames."""

from copy import deepcopy

import numpy as np


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_w * intersection_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class TemporalObjectSelector:
    """Associate boxes frame-to-frame, smooth scores/boxes, and resist flicker."""

    def __init__(
        self,
        score_alpha=0.35,
        bbox_alpha=0.45,
        switch_margin=0.08,
        switch_confirmation_frames=4,
        match_iou_threshold=0.20,
        max_missing_frames=5,
        selected_missing_grace_frames=3,
    ):
        self.score_alpha = score_alpha
        self.bbox_alpha = bbox_alpha
        self.switch_margin = switch_margin
        self.switch_confirmation_frames = switch_confirmation_frames
        self.match_iou_threshold = match_iou_threshold
        self.max_missing_frames = max_missing_frames
        self.selected_missing_grace_frames = selected_missing_grace_frames
        self.tracks = {}
        self.next_track_id = 1
        self.selected_track_id = None
        self.pending_track_id = None
        self.pending_frames = 0

    def _associate(self, detections):
        available_tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if track["missing_frames"] <= self.max_missing_frames
        }
        pairs = []
        for detection_index, detection in enumerate(detections):
            for track_id, track in available_tracks.items():
                if detection["class_id"] != track["class_id"]:
                    continue
                overlap = _iou(detection["bbox_xyxy"], track["bbox_xyxy"])
                if overlap >= self.match_iou_threshold:
                    pairs.append((overlap, detection_index, track_id))

        pairs.sort(reverse=True)
        detection_to_track = {}
        used_tracks = set()
        for _, detection_index, track_id in pairs:
            if detection_index in detection_to_track or track_id in used_tracks:
                continue
            detection_to_track[detection_index] = track_id
            used_tracks.add(track_id)
        return detection_to_track, used_tracks

    def _update_tracks(self, detections):
        detection_to_track, used_tracks = self._associate(detections)
        for detection_index, detection in enumerate(detections):
            track_id = detection_to_track.get(detection_index)
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {
                    "class_id": detection["class_id"],
                    "bbox_xyxy": np.asarray(
                        detection["bbox_xyxy"], dtype=np.float32
                    ),
                    "smoothed_score": detection["selection_score"],
                    "missing_frames": 0,
                    "last_detection": None,
                }
            else:
                track = self.tracks[track_id]
                current_box = np.asarray(detection["bbox_xyxy"], dtype=np.float32)
                track["bbox_xyxy"] = (
                    self.bbox_alpha * current_box
                    + (1.0 - self.bbox_alpha) * track["bbox_xyxy"]
                )
                current_score = detection["selection_score"]
                if current_score is not None:
                    previous_score = track["smoothed_score"]
                    track["smoothed_score"] = (
                        current_score if previous_score is None else
                        self.score_alpha * current_score
                        + (1.0 - self.score_alpha) * previous_score
                    )
                track["missing_frames"] = 0

            track = self.tracks[track_id]
            detection["track_id"] = track_id
            detection["smoothed_bbox_xyxy"] = [
                float(value) for value in track["bbox_xyxy"]
            ]
            detection["smoothed_selection_score"] = track["smoothed_score"]
            detection["selected"] = False
            track["last_detection"] = deepcopy(detection)

        for track_id, track in list(self.tracks.items()):
            if track_id not in used_tracks and not any(
                detection.get("track_id") == track_id for detection in detections
            ):
                track["missing_frames"] += 1
            if track["missing_frames"] > self.max_missing_frames:
                del self.tracks[track_id]

    def _advance_selection(self, eligible):
        best = max(
            eligible,
            key=lambda detection: detection["smoothed_selection_score"],
            default=None,
        )
        if best is None:
            return

        selected_track = self.tracks.get(self.selected_track_id)
        if selected_track is None:
            self.selected_track_id = best["track_id"]
            self.pending_track_id = None
            self.pending_frames = 0
            return

        if best["track_id"] == self.selected_track_id:
            self.pending_track_id = None
            self.pending_frames = 0
            return

        selected_score = selected_track["smoothed_score"]
        best_score = best["smoothed_selection_score"]
        if selected_score is None or best_score < selected_score + self.switch_margin:
            self.pending_track_id = None
            self.pending_frames = 0
            return

        if self.pending_track_id == best["track_id"]:
            self.pending_frames += 1
        else:
            self.pending_track_id = best["track_id"]
            self.pending_frames = 1

        if self.pending_frames >= self.switch_confirmation_frames:
            self.selected_track_id = best["track_id"]
            self.pending_track_id = None
            self.pending_frames = 0

    def select(self, detections, pointing_available):
        """Mutate detections with track info and return the selected detection."""
        self._update_tracks(detections)
        eligible = [
            detection for detection in detections
            if detection["eligible"] and detection["smoothed_selection_score"] is not None
        ]

        if pointing_available:
            selected_current = next(
                (
                    detection for detection in eligible
                    if detection["track_id"] == self.selected_track_id
                ),
                None,
            )
            if selected_current is None and self.selected_track_id in self.tracks:
                selected_track = self.tracks[self.selected_track_id]
                if selected_track["missing_frames"] == 0:
                    # The selected object is visible but no longer eligible.
                    self.selected_track_id = None
                    self.pending_track_id = None
                    self.pending_frames = 0

            self._advance_selection(eligible)

            selected_current = next(
                (
                    detection for detection in eligible
                    if detection["track_id"] == self.selected_track_id
                ),
                None,
            )
            if selected_current is not None:
                selected_current["selected"] = True
                return selected_current

            selected_track = self.tracks.get(self.selected_track_id)
            if (
                selected_track is not None
                and 0 < selected_track["missing_frames"]
                <= self.selected_missing_grace_frames
                and selected_track["last_detection"] is not None
            ):
                held = deepcopy(selected_track["last_detection"])
                held["selected"] = True
                held["tracked_missing"] = True
                return held

        return None
