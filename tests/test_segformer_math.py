import numpy as np
import pytest

from jacket import config
from jacket.segformer_classifier import arm_share_to_warm_prob, result_from_mask

BACKGROUND, CLOTHES, ARM, SCARF = 0, 4, 14, 17


def mask_of(counts: dict[int, int], size: int = 1000) -> np.ndarray:
    labels = [label for label, count in counts.items() for _ in range(count)]
    return np.array(labels + [BACKGROUND] * (size - len(labels)))


def score(counts: dict[int, int]):
    return result_from_mask(mask_of(counts), [CLOTHES], [ARM], [SCARF])


def test_ramp_ends_and_clamps():
    warm, light = config.SEGFORMER_ARM_SHARE_WARM, config.SEGFORMER_ARM_SHARE_LIGHT
    assert arm_share_to_warm_prob(0.0, warm, light) == 1.0
    assert arm_share_to_warm_prob(warm, warm, light) == 1.0
    assert arm_share_to_warm_prob(light, warm, light) == 0.0
    assert arm_share_to_warm_prob(0.9, warm, light) == 0.0
    assert arm_share_to_warm_prob((warm + light) / 2, warm, light) == pytest.approx(0.5)


def test_covered_arms_are_warm():
    assert score({CLOTHES: 500, ARM: 10}).label == "warm"


def test_bare_arms_are_light():
    assert score({CLOTHES: 400, ARM: 200}).label == "light"


def test_scarf_is_warm_even_with_bare_arms():
    assert score({CLOTHES: 300, ARM: 200, SCARF: 50}).label == "warm"


def test_too_little_upper_body_is_unknown():
    result = score({CLOTHES: 10})
    assert result.label == "unknown" and result.warm_prob is None and result.reason
