"""Frame-by-frame hand detection using the project's MediaPipe ensemble."""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


POSE_INDEX = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}


class VideoHandDetector:
    """Shared Holistic/Hand Landmarker ensemble for images and video frames.

    ``running_mode="image"`` is used by the dataset script;
    ``running_mode="video"`` enables MediaPipe's frame-to-frame tracking.
    Both paths use the same matching, crop fallback, and selection logic.
    """

    def __init__(
        self,
        model_dir,
        crop_size=500,
        agreement_threshold=0.08,
        min_detection_confidence=0.3,
        running_mode="video",
        crop_output_dir=None,
    ):
        model_dir = Path(model_dir)
        holistic_path = model_dir / "holistic_landmarker.task"
        hand_path = model_dir / "hand_landmarker.task"
        for path in (holistic_path, hand_path):
            if not path.is_file():
                raise FileNotFoundError(f"Could not find MediaPipe model: {path}")

        self.crop_size = crop_size
        self.agreement_threshold = agreement_threshold
        self.running_mode = running_mode.lower()
        self.crop_output_dir = Path(crop_output_dir) if crop_output_dir else None
        if self.crop_output_dir is not None:
            self.crop_output_dir.mkdir(parents=True, exist_ok=True)
        if self.running_mode not in {"image", "video"}:
            raise ValueError("running_mode must be 'image' or 'video'.")
        self.vision = mp.tasks.vision
        base_options = mp.tasks.BaseOptions
        mode = (
            self.vision.RunningMode.VIDEO
            if self.running_mode == "video"
            else self.vision.RunningMode.IMAGE
        )

        holistic_options = self.vision.HolisticLandmarkerOptions(
            base_options=base_options(model_asset_path=str(holistic_path)),
            running_mode=mode,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_landmarks_confidence=min_detection_confidence,
            min_hand_landmarks_confidence=min_detection_confidence,
        )
        video_hand_options = self.vision.HandLandmarkerOptions(
            base_options=base_options(model_asset_path=str(hand_path)),
            running_mode=mode,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        fallback_options = self.vision.HandLandmarkerOptions(
            base_options=base_options(model_asset_path=str(hand_path)),
            running_mode=self.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )

        self.holistic_model = (
            self.vision.HolisticLandmarker.create_from_options(holistic_options)
        )
        self.hand_model = self.vision.HandLandmarker.create_from_options(
            video_hand_options
        )
        self.fallback_model = self.vision.HandLandmarker.create_from_options(
            fallback_options
        )

    @staticmethod
    def _to_mp_image(rgb):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    @staticmethod
    def _to_pixels(landmarks, width, height, origin=(0, 0)):
        ox, oy = origin
        return [
            {"x": int(point.x * width) + ox, "y": int(point.y * height) + oy}
            for point in landmarks
        ]

    @staticmethod
    def _distance(points_a, points_b, width, height):
        if not points_a or not points_b:
            return float("inf")
        a = np.array([points_a[0]["x"], points_a[0]["y"]], dtype=float)
        b = np.array([points_b[0]["x"], points_b[0]["y"]], dtype=float)
        return float(np.linalg.norm(a - b) / np.hypot(width, height))

    @staticmethod
    def _handedness(result, index):
        try:
            return result.handedness[index][0].category_name.lower()
        except (IndexError, TypeError, AttributeError):
            return "unknown"

    @staticmethod
    def _pose_point(pose, name, width, height):
        if not pose:
            return None
        landmark = pose[POSE_INDEX[name]]
        visibility = getattr(landmark, "visibility", None)
        if visibility is not None and visibility < 0.1:
            return None
        return [int(landmark.x * width), int(landmark.y * height)]

    def _crop_centers(self, pose, side, width, height):
        shoulder = self._pose_point(pose, f"{side}_shoulder", width, height)
        elbow = self._pose_point(pose, f"{side}_elbow", width, height)
        wrist = self._pose_point(pose, f"{side}_wrist", width, height)
        centers = []

        if wrist is not None:
            centers.append(("wrist", wrist))
        if elbow is not None and wrist is not None:
            arm = np.asarray(wrist, dtype=float) - np.asarray(elbow, dtype=float)
            length = np.linalg.norm(arm)
            if length > 1:
                center = np.asarray(wrist, dtype=float) + arm / length * length
                centers.append(("beyond_wrist", center.astype(int).tolist()))
        if wrist is None and shoulder is not None and elbow is not None:
            arm = np.asarray(elbow, dtype=float) - np.asarray(shoulder, dtype=float)
            length = np.linalg.norm(arm)
            if length > 1:
                center = np.asarray(elbow, dtype=float) + arm / length * length
                centers.append(("beyond_elbow", center.astype(int).tolist()))

        unique = []
        for name, center in centers:
            if all(
                np.linalg.norm(np.asarray(center) - np.asarray(existing)) > 30
                for _, existing in unique
            ):
                unique.append((name, center))
        return unique

    def _crop(self, image, center):
        height, width = image.shape[:2]
        cx, cy = map(int, center)
        half = self.crop_size // 2
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(width, cx + half), min(height, cy + half)
        if x2 <= x1 or y2 <= y1:
            return None, None

        crop = image[y1:y2, x1:x2].copy()
        crop_h, crop_w = crop.shape[:2]
        top = (self.crop_size - crop_h) // 2
        bottom = self.crop_size - crop_h - top
        left = (self.crop_size - crop_w) // 2
        right = self.crop_size - crop_w - left
        crop = cv2.copyMakeBorder(
            crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return crop, (x1 - left, y1 - top)

    def _crop_fallback(self, frame_bgr, pose, side, frame_name=None):
        height, width = frame_bgr.shape[:2]
        for candidate_name, center in self._crop_centers(
            pose, side, width, height
        ):
            crop, origin = self._crop(frame_bgr, center)
            if crop is None:
                continue
            if self.crop_output_dir is not None and frame_name:
                cv2.imwrite(
                    str(self.crop_output_dir / f"{frame_name}_{side}_{candidate_name}.jpg"),
                    crop,
                )
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = self.fallback_model.detect(self._to_mp_image(crop_rgb))
            if not result.hand_landmarks:
                continue
            points = self._to_pixels(
                result.hand_landmarks[0], self.crop_size, self.crop_size, origin
            )
            return points, f"crop_fallback:{candidate_name}"
        return None, None

    def detect(self, frame_bgr, timestamp_ms=0, frame_name=None):
        """Return final pixel landmarks plus the ensemble's per-model evidence."""
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._to_mp_image(rgb)
        if self.running_mode == "video":
            holistic = self.holistic_model.detect_for_video(mp_image, timestamp_ms)
            direct_result = self.hand_model.detect_for_video(mp_image, timestamp_ms)
        else:
            holistic = self.holistic_model.detect(mp_image)
            direct_result = self.hand_model.detect(mp_image)
        pose = holistic.pose_landmarks

        holistic_hands = {}
        self.last_pose_found = bool(pose)
        self.last_pose_landmarks = (
            self._to_pixels(pose, width, height) if pose else None
        )
        self.last_pose_visibility = (
            [float(getattr(point, "visibility", 1.0)) for point in pose]
            if pose else None
        )
        for side in ("left", "right"):
            normalized = getattr(holistic, f"{side}_hand_landmarks", None)
            holistic_hands[side] = (
                self._to_pixels(normalized, width, height) if normalized else None
            )

        direct_hands = []
        for index, normalized in enumerate(direct_result.hand_landmarks):
            direct_hands.append({
                "side": self._handedness(direct_result, index),
                "landmarks": self._to_pixels(normalized, width, height),
            })

        used_direct = set()
        final_hands = []
        for side in ("left", "right"):
            holistic_points = holistic_hands[side]
            candidates = [
                (index, hand) for index, hand in enumerate(direct_hands)
                if index not in used_direct
                and hand["side"] in {side, "unknown"}
            ]
            nearest = None
            disagreement = None
            if holistic_points and candidates:
                nearest = min(
                    candidates,
                    key=lambda pair: self._distance(
                        holistic_points, pair[1]["landmarks"], width, height
                    ),
                )
                disagreement = self._distance(
                    holistic_points, nearest[1]["landmarks"], width, height
                )
                used_direct.add(nearest[0])
                final_points = holistic_points
                decision = (
                    "agreement"
                    if disagreement <= self.agreement_threshold
                    else "disagreement"
                )
            elif holistic_points:
                final_points = holistic_points
                decision = "holistic_only"
            elif candidates:
                direct_index, direct_hand = candidates[0]
                used_direct.add(direct_index)
                final_points = direct_hand["landmarks"]
                decision = "hand_landmarker_only"
            else:
                final_points, source = self._crop_fallback(
                    frame_bgr, pose, side, frame_name
                )
                decision = source or "not_found"

            if final_points:
                fallback_points = (
                    final_points
                    if decision.startswith("crop_fallback") else None
                )
                direct_points = (
                    nearest[1]["landmarks"]
                    if nearest is not None and holistic_points else
                    final_points if decision == "hand_landmarker_only" else None
                )
                final_hands.append({
                    "side": side,
                    "landmarks": final_points,
                    "decision": decision,
                    "disagreement_normalized": disagreement,
                    "holistic_landmarks": holistic_points,
                    "hand_landmarker_landmarks": direct_points,
                    "fallback_landmarks": fallback_points,
                    "confidence_label": (
                        "higher_evidence" if decision == "agreement"
                        else "uncertain" if decision == "disagreement"
                        else "single_model" if final_points else "none"
                    ),
                })

        # Retain direct detections without a corresponding Holistic hand.
        for index, hand in enumerate(direct_hands):
            if index in used_direct:
                continue
            final_hands.append({
                "side": hand["side"] if hand["side"] != "unknown" else f"unknown_{index}",
                "landmarks": hand["landmarks"],
                "decision": "hand_landmarker_only",
                "disagreement_normalized": None,
                "holistic_landmarks": None,
                "hand_landmarker_landmarks": hand["landmarks"],
                "fallback_landmarks": None,
                "confidence_label": "single_model",
            })

        return final_hands

    def close(self):
        self.holistic_model.close()
        self.hand_model.close()
        self.fallback_model.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
