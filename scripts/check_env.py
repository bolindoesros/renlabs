"""Run with: python -m scripts.check_env"""
import importlib
import logging
import sys
from dataclasses import dataclass
from typing import Callable

import cv2
import open_clip
import torch

from jacket import config

logger = logging.getLogger("check_env")

REQUIRED_PACKAGES = ["ultralytics", "open_clip", "torch", "cv2", "PIL", "numpy", "pytest"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_python_version() -> str:
    found = sys.version_info[:2]
    if found < config.MIN_PYTHON_VERSION:
        raise RuntimeError(f"need >= {config.MIN_PYTHON_VERSION}, found {found}")
    return sys.version.split()[0]


def check_package_imports() -> str:
    versions = []
    for package_name in REQUIRED_PACKAGES:
        module = importlib.import_module(package_name)
        versions.append(f"{package_name} {getattr(module, '__version__', '?')}")
    return ", ".join(versions)


def check_torch_device() -> str:
    if not torch.cuda.is_available():
        return "cpu (CUDA not available)"
    return f"cuda: {torch.cuda.get_device_name(0)}"


def check_webcam() -> str:
    capture = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open camera index {config.CAMERA_INDEX}")
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        frame = None
        for _ in range(config.CAMERA_WARMUP_FRAMES):
            was_read, frame = capture.read()
            if not was_read:
                raise RuntimeError("camera opened but returned no frame")
    finally:
        capture.release()
    height, width = frame.shape[:2]
    return f"index {config.CAMERA_INDEX}, frame {width}x{height}"


def check_yolo_weights() -> str:
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    config.WEIGHTS_DIR.mkdir(exist_ok=True)
    weights_path = attempt_download_asset(str(config.YOLO_WEIGHTS_PATH))
    model = YOLO(weights_path)
    return f"{weights_path}, person class = {model.names[config.YOLO_PERSON_CLASS_ID]}"


def make_clip_check(model_key: str) -> Callable[[], str]:
    def check_clip_weights() -> str:
        spec = config.CLIP_MODELS[model_key]
        model, _, _ = open_clip.create_model_and_transforms(
            spec.model_name, pretrained=spec.pretrained
        )
        parameter_count = sum(p.numel() for p in model.parameters())
        return f"{spec.model_name}, {parameter_count / 1e6:.0f}M params"

    return check_clip_weights


def run_check(name: str, check: Callable[[], str]) -> CheckResult:
    # Broad catch is deliberate: report every failure, then exit non-zero.
    try:
        detail = check()
    except Exception as error:
        logger.debug("check %s failed", name, exc_info=True)
        return CheckResult(name, False, f"{type(error).__name__}: {error}")
    return CheckResult(name, True, detail)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
    logger.setLevel(logging.INFO)  # keep library logs quiet
    checks: list[tuple[str, Callable[[], str]]] = [
        ("python version", check_python_version),
        ("package imports", check_package_imports),
        ("torch device", check_torch_device),
        ("webcam", check_webcam),
        ("yolo weights", check_yolo_weights),
    ]
    checks += [(f"clip weights [{key}]", make_clip_check(key)) for key in config.CLIP_MODELS]

    results = [run_check(name, check) for name, check in checks]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        logger.info("[%s] %s: %s", status, result.name, result.detail)

    failed_count = sum(not result.passed for result in results)
    logger.info("%d/%d checks passed", len(results) - failed_count, len(results))
    return 1 if failed_count else 0


if __name__ == "__main__":
    sys.exit(main())
