"""Crops on disk: collect from the camera, sort into labelled folders."""
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2

from jacket import config
from jacket.types import CropResult

logger = logging.getLogger("ren.dataset")

BINS = ("warm", "light", "discard")  # warm and light feed python -m jacket.eval


def bin_dir(name: str) -> Path:
    if name not in BINS:
        raise KeyError(f"unknown bin {name!r}")
    return config.DISCARDED_CROPS_DIR if name == "discard" else config.EVAL_DIR / name


def images_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in config.IMAGE_EXTENSIONS)


def save_snapshot(crops: Sequence[CropResult], folder: Path | None = None) -> list[Path]:
    """Write every usable crop to the to-sort folder; returns the new files."""
    folder = folder or config.RAW_CROPS_DIR
    usable = [crop.crop for crop in crops if crop.crop is not None]
    if not usable:
        return []
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    paths = []
    for index, crop in enumerate(usable):
        path = folder / f"{stamp}_{index}.jpg"
        if not cv2.imwrite(str(path), crop):
            raise OSError(f"could not write crop to {path}")
        paths.append(path)
    logger.info("saved %d crop(s) to %s", len(paths), folder)
    return paths


def _free_path(folder: Path, name: str) -> Path:
    path = folder / name
    stem, suffix, number = path.stem, path.suffix, 1
    while path.exists():
        path = folder / f"{stem}_{number}{suffix}"
        number += 1
    return path


def file_into(paths: Sequence[Path], name: str) -> list[Path]:
    """Move collected crops into a bin; files from elsewhere are copied, never moved."""
    raw_dir = config.RAW_CROPS_DIR
    folder = bin_dir(name)
    folder.mkdir(parents=True, exist_ok=True)
    filed = []
    for path in paths:
        if path.suffix.lower() not in config.IMAGE_EXTENSIONS or not path.is_file():
            logger.warning("skipping %s: not an image file", path)
            continue
        target = _free_path(folder, path.name)
        if path.resolve().parent == raw_dir.resolve():
            shutil.move(path, target)
        else:
            shutil.copy2(path, target)
        filed.append(target)
    return filed
