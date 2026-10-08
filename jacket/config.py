"""All tunable values. Nothing here is read from the environment."""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DATA_DIR = PROJECT_ROOT / "data"

# Camera
CAMERA_INDEX = 2  # C920; index 0 is the laptop webcam
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_WARMUP_FRAMES = 5  # first frames are often dark

# Person detector
YOLO_WEIGHTS_PATH = WEIGHTS_DIR / "yolo11n-pose.pt"  # pose model: boxes plus keypoints
YOLO_PERSON_CLASS_ID = 0
KEYPOINT_LEFT_SHOULDER = 5  # COCO keypoint order
KEYPOINT_RIGHT_SHOULDER = 6
KEYPOINT_NOSE = 0
KEYPOINT_LEFT_EYE = 1
KEYPOINT_RIGHT_EYE = 2
KEYPOINT_MIN_CONFIDENCE = 0.5
CHIN_BELOW_NOSE_EYE_DISTANCES = 1.8  # includes neck; eye gap ignores head tilt


@dataclass(frozen=True)
class ClipModelSpec:
    model_name: str
    pretrained: str | None  # None for hf-hub models


CLIP_MODELS: dict[str, ClipModelSpec] = {
    "clip": ClipModelSpec("ViT-B-32", "laion2b_s34b_b79k"),
    "fashion": ClipModelSpec("hf-hub:Marqo/marqo-fashionSigLIP", None),
}

MIN_PYTHON_VERSION = (3, 10)

# Person detector tuning
YOLO_MIN_CONFIDENCE = 0.35
YOLO_IMAGE_SIZE = 640
DEVICE = "auto"  # "auto" picks cuda when available, else cpu

# Debug window
DEBUG_WINDOW_NAME = "smart-vent debug"
DEBUG_QUIT_KEY = "q"
DEBUG_LABEL_COLORS_BGR = {
    "warm": (30, 30, 150),  # dark red
    "light": (170, 80, 40),  # medium-dark blue
    "unknown": (128, 128, 128),  # grey
}
DEBUG_BOX_THICKNESS = 2
DEBUG_FONT_SCALE = 0.6

# Crop rules
MIN_CLASSIFY_CONFIDENCE = 0.5  # below this the box is too doubtful to classify
MIN_CROP_SIDE_PX = 48
PERSON_CROP_HEIGHT_WIDTHS = 0.6  # crop height, in box widths
HEAD_CROP_GAP_HEADS = 0.1  # gap under the chin, in head heights
HEAD_CROP_HEIGHT_HEADS = 2.0
HEAD_CROP_WIDTH_HEADS = 2.0

# Crop strip in debug view
DEBUG_CROP_KEY = "c"
DEBUG_CROP_WINDOW_NAME = "crops (what CLIP sees)"
DEBUG_CROP_STRIP_HEIGHT_PX = 160

# Classifier
CLASSIFIER_PROMPT_TEMPLATE = "a photo of a person wearing {}"
WARM_ITEMS = ["a hoodie", "a jacket", "a sweater", "a sweatshirt", "a coat"]
LIGHT_ITEMS = ["a t-shirt", "a short-sleeved shirt", "a tank top"]
LOGIT_SCALE = 100.0  # CLIP's usual softmax sharpness
WARM_THRESHOLD = 0.55  # warm_prob above this is warm
LIGHT_THRESHOLD = 0.45  # warm_prob below this is light

# Pipeline
CLASSIFY_INTERVAL_SECONDS = 1.0  # CLIP runs per person at most this often
SMOOTHING_ALPHA = 0.4  # weight of the newest score
TRACK_MIN_IOU = 0.3  # boxes overlapping less are different people
TRACK_MAX_MISSING_SECONDS = 2.0  # forget a person after this long unseen

# Eval and crop saving
EVAL_DIR = DATA_DIR / "eval"  # holds warm/ and light/ subfolders
RAW_CROPS_DIR = DATA_DIR / "raw"  # --save-crops writes here, sort by hand
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
SAVE_CROP_INTERVAL_SECONDS = 2.0  # keeps saved crops varied, not near-duplicates
