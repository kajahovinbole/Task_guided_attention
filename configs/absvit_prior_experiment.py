"""Settings shared by the offline AbSViT prior and focus-map evaluation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Change RUN_NAME when you want to keep a separate experiment's outputs.
RUN_NAME = "default"

DATA_DIR = ROOT / "data" / "funker_bra_20okt"
IMAGE_DIR = DATA_DIR / "imgs"
LABELS_PATH = DATA_DIR / "labels.json"

ENSEMBLE_RUN_NAME = "1_2_3_andrea_merged"
PRIOR_DIR = (
    ROOT / "results" / "combined_holistic_pose_hand"
    / ENSEMBLE_RUN_NAME / "numeric_pointing_prior"
)
CHECKPOINT = ROOT / "checkpoints" / "absvit_tiny_patch8_224_gap.pth"

IMAGE_SIZE = 224
PRIOR_SCALE = 25.0
CLS_PRIOR = 0.0
ATTENTION_LAYER_FROM_END = 3 #9th layer

RESULTS_DIR = ROOT / "results"
ABSVIT_OUTPUT_DIR = RESULTS_DIR / "absvit_prior" / RUN_NAME
MAPS_DIR = ABSVIT_OUTPUT_DIR / "maps"
FOCUS_BBOX_OUTPUT_DIR = RESULTS_DIR / "focus_bbox_eval" / RUN_NAME
PRIOR_IMAGES_OUTPUT_DIR = RESULTS_DIR / "prior_images" / RUN_NAME
