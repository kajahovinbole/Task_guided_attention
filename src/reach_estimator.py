"""2D arm-reach features used by the reach-cone experiment and video pipeline."""

import numpy as np


POSE_IDS = {
    "left": {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23},
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24},
}


def _distance(point_a, point_b):
    return float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))


def _angle_degrees(point_a, vertex, point_b):
    """Return the 2D angle at vertex, from 0 to 180 degrees."""
    vector_a = np.asarray(point_a, dtype=float) - np.asarray(vertex, dtype=float)
    vector_b = np.asarray(point_b, dtype=float) - np.asarray(vertex, dtype=float)
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator < 1e-8:
        return None
    cosine = np.clip(np.dot(vector_a, vector_b) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def measure_arm(pose_landmarks, side, pose_visibility=None, min_visibility=0.1):
    """Measure elbow opening and shoulder/torso angle from pixel pose points."""
    if side not in POSE_IDS or pose_landmarks is None:
        return None
    ids = POSE_IDS[side]
    if len(pose_landmarks) <= max(ids.values()):
        return None
    if pose_visibility is not None and any(
        pose_visibility[index] < min_visibility for index in ids.values()
    ):
        return None

    points = {
        name: np.array(
            [pose_landmarks[index]["x"], pose_landmarks[index]["y"]],
            dtype=float,
        )
        for name, index in ids.items()
    }
    elbow_angle = _angle_degrees(
        points["shoulder"], points["elbow"], points["wrist"]
    )
    shoulder_torso_angle = _angle_degrees(
        points["hip"], points["shoulder"], points["elbow"]
    )
    torso_length = _distance(points["shoulder"], points["hip"])
    shoulder_wrist_ratio = (
        _distance(points["shoulder"], points["wrist"]) / torso_length
        if torso_length > 1e-6 else None
    )
    if elbow_angle is None or shoulder_torso_angle is None:
        return None

    # Experiment mappings: straight elbow (180 deg) -> 1; 90 deg -> 0.
    elbow_extension = float(np.clip((elbow_angle - 90.0) / 90.0, 0.0, 1.0))
    # Arm angle relative to torso: 15..90 deg -> [0,1].
    shoulder_extension = float(
        np.clip((shoulder_torso_angle - 15.0) / 75.0, 0.0, 1.0)
    )
    return {
        "elbow_angle_degrees": elbow_angle,
        "shoulder_torso_angle_degrees": shoulder_torso_angle,
        "shoulder_wrist_to_torso_ratio": shoulder_wrist_ratio,
        "elbow_extension_score": elbow_extension,
        "shoulder_extension_score": shoulder_extension,
    }


def make_reach_score(arm_measurements, elbow_weight=0.7, shoulder_weight=0.3):
    """Combine normalized elbow and shoulder extension into a [0,1] proxy."""
    if arm_measurements is None:
        return None
    weight_total = elbow_weight + shoulder_weight
    if elbow_weight < 0 or shoulder_weight < 0 or weight_total <= 0:
        raise ValueError("Reach-score weights must be nonnegative and not both zero.")
    return (
        elbow_weight * arm_measurements["elbow_extension_score"]
        + shoulder_weight * arm_measurements["shoulder_extension_score"]
    ) / weight_total


def adaptive_heat_extent(reach_score, min_extent, max_extent):
    """Map reach proxy to falloff distance within a cone of fixed geometry.

    Extents are multiples of cone length. A larger extent retains more heat
    toward the cone's far end; it never changes the cone's support.
    """
    if reach_score is None:
        return None
    score = float(np.clip(reach_score, 0.0, 1.0))
    return float(min_extent + score * (max_extent - min_extent))


def measure_torso_fraction(
    pose_landmarks, image_height, pose_visibility=None, min_visibility=0.2
):
    """Estimate apparent torso size as shoulder-to-hip distance / image height."""
    if pose_landmarks is None or len(pose_landmarks) <= 24 or image_height <= 0:
        return None
    required = (11, 12, 23, 24)
    if pose_visibility is not None and any(
        index >= len(pose_visibility)
        or pose_visibility[index] < min_visibility
        for index in required
    ):
        return None

    points = [
        np.asarray(
            [pose_landmarks[index]["x"], pose_landmarks[index]["y"]],
            dtype=float,
        )
        for index in required
    ]
    shoulder_midpoint = (points[0] + points[1]) / 2.0
    hip_midpoint = (points[2] + points[3]) / 2.0
    return _distance(shoulder_midpoint, hip_midpoint) / image_height


def person_scale_width_factor(
    torso_fraction, reference_fraction, min_factor=0.4, max_factor=1.5
):
    """Scale cone width from apparent torso size relative to a reference."""
    if torso_fraction is None:
        return None
    if reference_fraction <= 0 or min_factor <= 0 or max_factor < min_factor:
        raise ValueError("Person scale limits must satisfy 0 < min <= max.")
    return float(np.clip(torso_fraction / reference_fraction, min_factor, max_factor))


class ReachScoreSmoother:
    """EMA with a short missing-pose hold to avoid abrupt cone-length jumps."""

    def __init__(self, alpha=0.35, missing_hold_frames=3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if missing_hold_frames < 0:
            raise ValueError("missing_hold_frames cannot be negative.")
        self.alpha = alpha
        self.missing_hold_frames = missing_hold_frames
        self.value = None
        self.missing_frames = 0

    def update(self, score):
        if score is None:
            if self.value is None:
                return None
            self.missing_frames += 1
            if self.missing_frames > self.missing_hold_frames:
                self.value = None
                return None
            return self.value

        score = float(np.clip(score, 0.0, 1.0))
        self.value = (
            score if self.value is None else
            self.alpha * score + (1.0 - self.alpha) * self.value
        )
        self.missing_frames = 0
        return self.value
