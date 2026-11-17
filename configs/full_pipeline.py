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
    max_frames: int | None = None
    show_windows: bool = False

    # Hand detection and pointing-hand selection
    min_pointing_score: float = 0.60
    hand_detection_confidence: float = 0.30
    hand_crop_size: int = 500
    hand_agreement_threshold: float = 0.08

    # YOLO detections and object-selection signal weights
    yolo_inference_confidence: float = 0.05
    min_yolo_confidence: float = 0.25
    saliency_weight: float = 0.80
    yolo_confidence_weight: float = 0.20

    # Cone prior geometry; distances are fractions of the frame diagonal
    cone_length: float = 0.60
    cone_width: float = 0.015
    cone_widening: float = 0.12
    cone_near_ramp: float = 0.05

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
        if not 0.0 <= self.heatmap_opacity <= 1.0:
            raise ValueError("heatmap_opacity må være mellom 0 og 1.")
        if self.yolo_image_size <= 0 or self.hand_crop_size <= 0:
            raise ValueError("Bildestørrelse og crop-størrelse må være positive.")


DEFAULT_CONFIG = PipelineConfig()
