"""Central settings for the video hand-guided object selection pipeline."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelineConfig:
    # Models and runtime
    checkpoint_path: Path = ROOT / "checkpoints" / "absvit_tiny_patch8_224_gap.pth"
    yolo_model: str = "yolo26n.pt"
    yolo_image_size: int = 640
    device: str = "auto"  # auto, cpu, cuda, or mps

    # Input/output runtime defaults
    output_root: Path = ROOT / "results" / "video_pipeline"
    max_frames: int | None = None #hele videoen
    show_windows: bool = False

    # Hand detection and pointing-hand selection
    min_pointing_score: float = 0.60
    hand_detection_confidence: float = 0.30
    hand_crop_size: int = 500
    hand_agreement_threshold: float = 0.08

    # YOLO detections and object-selection signal weights
    yolo_inference_confidence: float = 0.25
    min_yolo_confidence: float = 0.25
    saliency_weight: float = 1
    yolo_confidence_weight: float = 0
    score_smoothing_alpha: float = 0.35     #lavere verdi gir mer smoothing
    bbox_smoothing_alpha: float = 0.45
    track_switch_margin: float = 0.08
    switch_confirmation_frames: int = 4
    track_match_iou_threshold: float = 0.20
    track_max_missing_frames: int = 5
    selected_missing_grace_frames: int = 3

    # Cone prior geometry; distances are fractions of the frame diagonal
    cone_length: float = 0.60
    cone_width: float = 0.015
    cone_widening: float = 0.12
    cone_near_ramp: float = 0.05

    # Optional arm-reach proxy for heat falloff within the fixed cone.
    use_reach_adaptive_heat: bool = True
    reach_near_heat_extent: float = 0.35
    reach_far_heat_extent: float = 1.50
    reach_elbow_weight: float = 0.70
    reach_shoulder_weight: float = 0.30
    reach_score_alpha: float = 0.35
    reach_score_hold_missing_frames: int = 3

    # Optional cone-width proxy based on apparent shoulder-to-hip size
    use_person_scale_adaptive_cone: bool = True
    person_scale_reference_fraction: float = 0.18
    person_scale_min_width_factor: float = 0.40
    person_scale_max_width_factor: float = 1.50

    # AbSViT prior and focus-map settings
    prior_scale: float = 25.0
    cls_prior: float = 0.0
    attention_layer_from_end: int = 3

    # Verification-video rendering
    heatmap_opacity: float = 0.45

    def __post_init__(self):
        if not 0.0 <= self.min_pointing_score <= 1.0:
            raise ValueError("min_pointing_score må være mellom 0 og 1.")
        if not 0.0 <= self.yolo_inference_confidence <= 1.0:
            raise ValueError("yolo_inference_confidence må være mellom 0 og 1.")
        if not 0.0 <= self.min_yolo_confidence <= 1.0:
            raise ValueError("min_yolo_confidence må være mellom 0 og 1.")
        if self.saliency_weight < 0 or self.yolo_confidence_weight < 0:
            raise ValueError("Signalvektene kan ikke være negative.")
        if self.saliency_weight + self.yolo_confidence_weight <= 0:
            raise ValueError("Minst én signalvekt må være større enn null.")
        if not 0.0 < self.score_smoothing_alpha <= 1.0:
            raise ValueError("score_smoothing_alpha må være større enn 0 og maks 1.")
        if not 0.0 < self.bbox_smoothing_alpha <= 1.0:
            raise ValueError("bbox_smoothing_alpha må være større enn 0 og maks 1.")
        if not 0.0 <= self.track_switch_margin <= 1.0:
            raise ValueError("track_switch_margin må være mellom 0 og 1.")
        if not 0.0 <= self.track_match_iou_threshold <= 1.0:
            raise ValueError("track_match_iou_threshold må være mellom 0 og 1.")
        if self.switch_confirmation_frames <= 0:
            raise ValueError("switch_confirmation_frames må være positiv.")
        if self.track_max_missing_frames < 0:
            raise ValueError("track_max_missing_frames kan ikke være negativ.")
        if not 0 <= self.selected_missing_grace_frames <= self.track_max_missing_frames:
            raise ValueError(
                "selected_missing_grace_frames må være mellom 0 og "
                "track_max_missing_frames."
            )
        if not 0.0 <= self.heatmap_opacity <= 1.0:
            raise ValueError("heatmap_opacity må være mellom 0 og 1.")
        if self.yolo_image_size <= 0 or self.hand_crop_size <= 0:
            raise ValueError("Bildestørrelse og crop-størrelse må være positive.")
        if not 0 < self.reach_near_heat_extent <= self.reach_far_heat_extent:
            raise ValueError("Reach heat extents must satisfy 0 < near <= far.")
        if self.reach_elbow_weight < 0 or self.reach_shoulder_weight < 0:
            raise ValueError("Reach score weights cannot be negative.")
        if self.reach_elbow_weight + self.reach_shoulder_weight <= 0:
            raise ValueError("At least one reach score weight must be positive.")
        if not 0.0 < self.reach_score_alpha <= 1.0:
            raise ValueError("reach_score_alpha must be in (0, 1].")
        if self.reach_score_hold_missing_frames < 0:
            raise ValueError("reach_score_hold_missing_frames cannot be negative.")
        if self.person_scale_reference_fraction <= 0:
            raise ValueError("person_scale_reference_fraction must be positive.")
        if not 0 < self.person_scale_min_width_factor <= self.person_scale_max_width_factor:
            raise ValueError("Person scale factors must satisfy 0 < min <= max.")


DEFAULT_CONFIG = PipelineConfig()
