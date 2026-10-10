import logging

import cv2
import numpy as np
import open_clip
import torch
import torch.nn.functional as functional
from PIL import Image

from jacket import config
from jacket.person_detector import resolve_device
from jacket.types import ClothingResult

logger = logging.getLogger(__name__)


def prompt_probabilities(
    image_embedding: torch.Tensor, text_embeddings: torch.Tensor, logit_scale: float
) -> torch.Tensor:
    """Softmax over all prompts. Shapes: (dim,) and (prompts, dim)."""
    image_unit = functional.normalize(image_embedding, dim=-1)
    text_unit = functional.normalize(text_embeddings, dim=-1)
    return (logit_scale * text_unit @ image_unit).softmax(dim=0)


def warm_probability(probabilities: torch.Tensor, warm_count: int) -> float:
    """Warm prompts come first; sum the leading slice."""
    return float(probabilities[:warm_count].sum())


def result_from_warm_probability(
    warm_prob: float, warm_threshold: float, light_threshold: float
) -> ClothingResult:
    if warm_prob > warm_threshold:
        return ClothingResult("warm", warm_prob, "")
    if warm_prob < light_threshold:
        return ClothingResult("light", warm_prob, "")
    reason = f"warm_prob {warm_prob:.2f} between {light_threshold} and {warm_threshold}"
    return ClothingResult("unknown", warm_prob, reason)


class ClothingClassifier:
    """Zero-shot warm/light scoring of one torso crop."""

    def __init__(self, model_key: str, device: str) -> None:
        spec = config.CLIP_MODELS[model_key]
        self._model_key = model_key
        self._device = resolve_device(device)
        model, _, self._preprocess = open_clip.create_model_and_transforms(
            spec.model_name, pretrained=spec.pretrained
        )
        self._model = model.to(self._device).eval()

        items = config.WARM_ITEMS + config.LIGHT_ITEMS
        self._prompts = [config.CLASSIFIER_PROMPT_TEMPLATE.format(item) for item in items]
        tokens = open_clip.get_tokenizer(spec.model_name)(self._prompts).to(self._device)
        with torch.inference_mode():
            self._text_embeddings = self._model.encode_text(tokens)
        logger.info("classifier: %s on %s, %d prompts", model_key, self._device, len(self._prompts))

    def classify(self, crop_bgr: np.ndarray) -> ClothingResult:
        image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        batch = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            image_embedding = self._model.encode_image(batch)[0]
        probabilities = prompt_probabilities(
            image_embedding, self._text_embeddings, config.LOGIT_SCALE
        )
        warm_prob = warm_probability(probabilities, len(config.WARM_ITEMS))
        result = result_from_warm_probability(
            warm_prob, config.WARM_THRESHOLD, config.LIGHT_THRESHOLD
        )
        best = int(probabilities.argmax())
        logger.info(
            "%s: warm_prob=%.2f -> %s (top: '%s' %.2f)",
            self._model_key, warm_prob, result.label, self._prompts[best], probabilities[best],
        )
        return result


def build_classifier(model_key: str, device: str):
    """Any clothing model by key; segformer imports lazily."""
    if model_key in config.SEGFORMER_MODELS:
        from jacket.segformer_classifier import SegformerClassifier

        return SegformerClassifier(model_key, device)
    return ClothingClassifier(model_key, device)
