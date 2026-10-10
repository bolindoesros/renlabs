import logging

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

from jacket import config
from jacket.classifier import result_from_warm_probability
from jacket.person_detector import resolve_device
from jacket.types import ClothingResult

logger = logging.getLogger(__name__)


def arm_share_to_warm_prob(arm_share: float, warm_share: float, light_share: float) -> float:
    """Linear ramp: covered arms are warm, bare arms are light."""
    ramp = (arm_share - warm_share) / (light_share - warm_share)
    return float(min(max(1.0 - ramp, 0.0), 1.0))


def result_from_mask(
    mask: np.ndarray, clothing_ids: list[int], skin_ids: list[int], warm_ids: list[int]
) -> ClothingResult:
    """Score a per-pixel label mask of one torso crop."""
    clothing = int(np.isin(mask, clothing_ids).sum())
    skin = int(np.isin(mask, skin_ids).sum())
    if np.isin(mask, warm_ids).sum() >= config.SEGFORMER_MIN_SCARF_FRACTION * mask.size:
        return ClothingResult("warm", 1.0, "")
    if clothing + skin < config.SEGFORMER_MIN_UPPER_FRACTION * mask.size:
        return ClothingResult("unknown", None, f"upper body only {(clothing + skin) / mask.size:.0%} of crop")
    warm_prob = arm_share_to_warm_prob(
        skin / (clothing + skin), config.SEGFORMER_ARM_SHARE_WARM, config.SEGFORMER_ARM_SHARE_LIGHT
    )
    return result_from_warm_probability(warm_prob, config.WARM_THRESHOLD, config.LIGHT_THRESHOLD)


class SegformerClassifier:
    """Warm/light from a clothes segmentation of one torso crop."""

    def __init__(self, model_key: str, device: str) -> None:
        repo = config.SEGFORMER_MODELS[model_key]
        self._model_key = model_key
        self._device = resolve_device(device)
        self._processor = AutoImageProcessor.from_pretrained(repo)
        self._model = AutoModelForSemanticSegmentation.from_pretrained(repo).to(self._device).eval()
        label_ids = self._model.config.label2id
        self._clothing_ids = [label_ids[name] for name in config.SEGFORMER_CLOTHING_LABELS]
        self._skin_ids = [label_ids[name] for name in config.SEGFORMER_SKIN_LABELS]
        self._warm_ids = [label_ids[name] for name in config.SEGFORMER_WARM_LABELS]
        logger.info("classifier: %s on %s", model_key, self._device)

    def classify(self, crop_bgr: np.ndarray) -> ClothingResult:
        image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            logits = self._model(**inputs).logits[0]  # quarter resolution is enough for shares
        mask = logits.argmax(dim=0).cpu().numpy()
        result = result_from_mask(mask, self._clothing_ids, self._skin_ids, self._warm_ids)
        logger.info("%s: warm_prob=%s -> %s", self._model_key, result.warm_prob, result.label)
        return result
