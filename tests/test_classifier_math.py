import torch

from jacket import config
from jacket.classifier import (
    prompt_probabilities,
    result_from_warm_probability,
    warm_probability,
)

DIM, WARM_COUNT, LIGHT_COUNT = 16, 3, 2


def fixed_text_embeddings() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(WARM_COUNT + LIGHT_COUNT, DIM, generator=generator)


def test_probabilities_sum_to_one():
    image = torch.randn(DIM, generator=torch.Generator().manual_seed(1))
    probabilities = prompt_probabilities(image, fixed_text_embeddings(), config.LOGIT_SCALE)
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))


def test_warm_probability_is_sum_of_warm_group():
    probabilities = torch.tensor([0.1, 0.2, 0.3, 0.25, 0.15])
    assert abs(warm_probability(probabilities, WARM_COUNT) - 0.6) < 1e-6


def test_image_matching_a_warm_prompt_scores_warm():
    texts = fixed_text_embeddings()
    probabilities = prompt_probabilities(texts[0], texts, config.LOGIT_SCALE)
    assert warm_probability(probabilities, WARM_COUNT) > 0.9


def test_image_matching_a_light_prompt_scores_light():
    texts = fixed_text_embeddings()
    probabilities = prompt_probabilities(texts[-1], texts, config.LOGIT_SCALE)
    assert warm_probability(probabilities, WARM_COUNT) < 0.1


def test_embedding_scale_does_not_change_result():
    texts = fixed_text_embeddings()
    image = torch.randn(DIM, generator=torch.Generator().manual_seed(2))
    base = prompt_probabilities(image, texts, config.LOGIT_SCALE)
    scaled = prompt_probabilities(image * 7, texts * 3, config.LOGIT_SCALE)
    assert torch.allclose(base, scaled, atol=1e-5)


def test_thresholds_split_warm_light_unknown():
    assert result_from_warm_probability(0.61, 0.6, 0.4).label == "warm"
    assert result_from_warm_probability(0.39, 0.6, 0.4).label == "light"
    assert result_from_warm_probability(0.5, 0.6, 0.4).label == "unknown"


def test_threshold_edges_are_unknown():
    assert result_from_warm_probability(0.6, 0.6, 0.4).label == "unknown"
    assert result_from_warm_probability(0.4, 0.6, 0.4).label == "unknown"


def test_unknown_has_reason_and_known_does_not():
    assert result_from_warm_probability(0.5, 0.6, 0.4).reason
    assert result_from_warm_probability(0.9, 0.6, 0.4).reason == ""
