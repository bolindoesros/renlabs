"""All tunable values. Nothing here is read from the environment."""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DATA_DIR = PROJECT_ROOT / "data"

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_WARMUP_FRAMES = 5  # first frames are often dark

# Person detector
YOLO_WEIGHTS_PATH = WEIGHTS_DIR / "yolo11n.pt"
YOLO_PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class ClipModelSpec:
    model_name: str
    pretrained: str | None  # None for hf-hub models


CLIP_MODELS: dict[str, ClipModelSpec] = {
    "clip": ClipModelSpec("ViT-B-32", "laion2b_s34b_b79k"),
    "fashion": ClipModelSpec("hf-hub:Marqo/marqo-fashionSigLIP", None),
}
ACTIVE_CLIP_MODEL = "clip"

MIN_PYTHON_VERSION = (3, 10)
