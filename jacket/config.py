"""All tunable values live here."""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DATA_DIR = PROJECT_ROOT / "data"

# Camera
CAMERA_NAME_HINT = "C920"  # found by name because index numbers shift
CAMERA_DEFAULT_INDEX = 0  # Windows and macOS cannot look up names
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_WARMUP_FRAMES = 5  # first frames are often dark
CAMERA_FOURCC = "MJPG"  # raw 720p caps near 10 fps
CAMERA_FPS = 30

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

# Human-parsing segmenters; warm/light comes from how much bare arm shows
SEGFORMER_MODELS: dict[str, str] = {"segformer": "mattmdjaga/segformer_b2_clothes"}
CLOTHING_MODELS = (*CLIP_MODELS, *SEGFORMER_MODELS)  # every clothing model key

CLOTHING_NAMES = {"warm": "jacket", "light": "shirt", "unknown": "unsure"}  # shown in the UI
CLIP_MODEL_NAMES = {"clip": "CLIP ViT-B/32", "fashion": "Marqo-FashionSigLIP", "segformer": "SegFormer-B2 clothes"}  # shown in the UI
DEFAULT_CLIP_MODEL = "fashion"  # beat clip on every test so far
MIN_PYTHON_VERSION = (3, 10)

# Person detector tuning
YOLO_MIN_CONFIDENCE = 0.35
YOLO_IMAGE_SIZE = 640
DEVICE = "auto"  # cuda, then Apple mps, then cpu

# Head detector: YOLOv8n trained on SCUT-HEAD (github.com/Abcfsa/YOLOv8_head_detector)
HEAD_WEIGHTS_PATH = WEIGHTS_DIR / "yolov8n-head.pt"
HEAD_WEIGHTS_URL = (
    "https://github.com/Abcfsa/YOLOv8_head_detector/raw/69559cef0a02adc4977f7bb4af6d11278819f332/nano.pt"
)
HEAD_WEIGHTS_SHA256 = "dc4f4a2b2e37a65a2524a305dc60de172c23cef6d05551b7a7c6905fd2f14f4b"
HEAD_MIN_CONFIDENCE = 0.35
HEAD_IMAGE_SIZE = 640

# Which detector finds people; "both" fuses body and head boxes
DETECTOR_MODES = ("body", "head", "both")
DEFAULT_DETECTOR = "body"
DETECTOR_NAMES = {"body": "YOLO11n pose", "head": "YOLOv8n head", "both": "YOLO11n-pose + YOLOv8n-head"}  # shown in the UI
# One-click model groups on the live toolbar: name -> (detector, clothing model); "none" switches a stage off
MODEL_PRESETS: dict[str, tuple[str, str]] = {
    "fast": ("body", "clip"),
    "accurate": ("both", "fashion"),
    "segment": ("body", "segformer"),
    "people only": ("body", "none"),
}
FUSE_HEAD_TOP_FRACTION = 0.45  # a head belongs to a body if its centre is in this top share
FUSE_HEAD_SIDE_MARGIN = 0.1  # in body widths; heads may poke out a little

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

# SegFormer classifier; label names come from the model's config
SEGFORMER_CLOTHING_LABELS = ("Upper-clothes", "Dress")
SEGFORMER_SKIN_LABELS = ("Left-arm", "Right-arm", "Left-leg", "Right-leg")  # torso crops show forearms as legs
SEGFORMER_WARM_LABELS = ("Scarf",)  # any real amount means warm
SEGFORMER_ARM_SHARE_WARM = 0.02  # arm share at or below this is fully warm (hands only)
SEGFORMER_ARM_SHARE_LIGHT = 0.10  # arm share at or above this is fully light; tuned on two crops only
SEGFORMER_MIN_UPPER_FRACTION = 0.05  # less clothing+arm than this share of the crop is unsure
SEGFORMER_MIN_SCARF_FRACTION = 0.02  # scarf share of the crop that counts

# Pipeline
CLASSIFY_INTERVAL_SECONDS = 1.0  # CLIP runs per person at most this often
SMOOTHING_ALPHA = 0.4  # weight of the newest score
TRACK_MIN_IOU = 0.3  # boxes overlapping less are different people
TRACK_MAX_MISSING_SECONDS = 2.0  # forget a person after this long unseen

# Eval and crop saving
EVAL_DIR = DATA_DIR / "eval"  # holds warm/ and light/ subfolders
RAW_CROPS_DIR = DATA_DIR / "raw"  # --save-crops writes here, sort by hand
DISCARDED_CROPS_DIR = DATA_DIR / "discarded"  # crops sorted out on the data page
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
SAVE_CROP_INTERVAL_SECONDS = 2.0  # keeps saved crops varied, not near-duplicates

# Overlay drawing (shared by debug view and ren.ui)
OVERLAY_NEUTRAL_COLOR_BGR = (200, 200, 200)  # detected, not classified
OVERLAY_LABEL_TEXT_BGR = (255, 255, 255)
OVERLAY_LABEL_PADDING_PX = 4

# Frame sources other than the camera
STILL_IMAGE_FPS = 15  # a still image is re-sent at this rate

# Demo window (ren.ui): soft, pastel, Google-style
UI_APP_NAME = "Ren.ui"
UI_WINDOW_TITLE = "Ren.ui"
UI_START_SIZE = (1360, 820)
UI_MIN_SIZE = (1100, 700)
UI_SHUTDOWN_WAIT_MS = 10000  # a model load cannot be interrupted
UI_SIGNAL_TICK_MS = 200  # lets Python handle Ctrl+C while Qt runs
UI_FONT_FILE = PROJECT_ROOT / "ren" / "assets" / "fonts" / "GoogleSans-VariableFont.ttf"
UI_TEXT_FONT = "Google Sans"
UI_MONO_FONTS = ["Google Sans Mono", "Roboto Mono", "Ubuntu Sans Mono", "DejaVu Sans Mono", "monospace"]
UI_WEIGHT = {"regular": 400, "medium": 500, "bold": 700}
UI_COLORS = {
    "page": "#f6f8fc",  # behind the cards
    "background": "#ffffff",  # card surface
    "text": "#1f1f1f",
    "label": "#444746",
    "muted": "#747775",
    "hairline": "#e3e7ed",
    "hover": "#eef2f8",
    "chip": "#edf1f7",  # dropdowns and buttons
    "accent": "#0b57d0",
    "accent_soft": "#d3e3fd",  # active nav pill, switch track
    "switch": "#c9ced6",
    "error": "#b3261e",
    "viewport": "#f1f3f6",  # behind the camera picture
    "air": "#81c995",  # airflow and live dots
    "shadow": "#1f3b6b",  # tint for soft shadows
}
UI_BRAND_DOTS = ("#8ab4f8", "#f28b82", "#fdd663", "#81c995")  # pastel Google four
UI_RADIUS = {"card": 20, "viewport": 14, "box": 0, "tag": 2}  # boxes stay square
UI_FONT_PX = {"body": 17, "small": 14, "mono": 14, "title": 22, "brand": 28, "headline": 21, "tag": 12}
UI_PAGE_PADDING_PX = 28
UI_CARD_MARGIN_PX = 10  # room for the card shadow
UI_PANEL_HEADER_PX = 52
UI_ROW_HEIGHT_PX = 56
UI_BOX_LINE_PX = 2.0
UI_BOX_HALO_PX = 4.5  # white halo, visible on dark footage
UI_TAG_PADDING_PX = (8, 3)  # horizontal, vertical
UI_LIVE_TIMEOUT_S = 1.5  # a panel's dot greys out after this

# Seating plan, back row at the top
PLAN_ROWS = 3
PLAN_COLS = 6
# Corners as frame fractions, back-left clockwise
PLAN_CORNERS = ((0.10, 0.22), (0.90, 0.22), (0.97, 0.92), (0.03, 0.92))
PLAN_SEAT_MATCH_RADIUS = 1.0  # in seat widths; farther people are off-plan
PLAN_ANCHOR_FALLBACK = 0.30  # shoulder row guess, as fraction of box height
PLAN_ZONE_ROWS = 2
PLAN_ZONE_COLS = 3

# Seat colours; hot wants cooling
SEAT_COLORS = {"hot": "#f28b82", "cold": "#8ab4f8", "unsure": "#fdd663"}  # pastel Google

# Decision: how much each seat state wants air
DECISION_SMOOTHING_S = 1.5  # seat demand averaging time
DECISION_MIN_SHARE = 0.10  # zones below this share are skipped
DECISION_MIN_DWELL_S = 6.0  # vent stays on a zone at least this long
DECISION_CLOSE_BELOW = 0.5  # fewer seated people than this shuts the vent
DECISION_AIM_MODE = "sweep"  # sweep shares time; focus holds top zone
DECISION_FOCUS_MARGIN = 0.15  # focus mode: switch only if clearly better

# Vent geometry (ceiling mounted); angles follow the MAEverick design
VENT_U = 0.5  # plan position, 0 left to 1 right
VENT_V = 0.5  # plan position, 0 front to 1 back
VENT_DROP_M = 2.0  # ceiling to head height
VENT_TILT_MAX_DEG = 45.0  # flap tilt range is -45 to +45
VENT_CLOSED_TILT_DEG = 90.0  # flaps shut
VENT_ROTATION_OFFSET_DEG = 0.0  # zero is toward the back row
VENT_CLOCKWISE = True  # positive rotation turns clockwise from above
VENT_SLEW_DEG_PER_S = 60.0
VENT_TURN_START_DEG = 20.0  # bigger turns shut the flaps first
VENT_TURN_DONE_DEG = 3.0  # turn counts as finished within this
VENT_SETTLED_TILT_DEG = 2.0  # tilt within this counts as aimed
SEAT_WIDTH_M = 0.55
ROW_DEPTH_M = 0.80

# Saved app settings (gitignored with data/)
SETTINGS_FILE = DATA_DIR / "settings.json"
