"""Score labelled crops: python -m jacket.eval"""
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from jacket import config
from jacket.classifier import ClothingClassifier

logger = logging.getLogger("jacket.eval")  # explicit: __name__ is __main__ under -m

TRUE_LABELS = ("warm", "light")
PREDICTED_LABELS = ("warm", "light", "unknown")


class EvalDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class Sample:
    path: Path
    true_label: str


@dataclass(frozen=True)
class EvalSummary:
    confusion: dict[str, dict[str, int]]  # confusion[true][predicted]
    total: int
    accuracy_on_known: float | None  # None when every sample was unknown
    unknown_rate: float
    correct_overall: float  # unknown counts as wrong


def load_samples(eval_dir: Path) -> list[Sample]:
    """Read the label folders; fail if one is empty."""
    samples: list[Sample] = []
    for label in TRUE_LABELS:
        folder = eval_dir / label
        if not folder.is_dir():
            raise EvalDataError(f"missing folder {folder}")
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in config.IMAGE_EXTENSIONS)
        if not paths:
            raise EvalDataError(f"no images in {folder}")
        samples += [Sample(path, label) for path in paths]
    return samples


def summarize(pairs: list[tuple[str, str]]) -> EvalSummary:
    """Pairs are (true label, predicted label)."""
    if not pairs:
        raise ValueError("no samples to summarize")
    confusion = {t: {p: 0 for p in PREDICTED_LABELS} for t in TRUE_LABELS}
    for true_label, predicted in pairs:
        confusion[true_label][predicted] += 1
    correct = sum(confusion[label][label] for label in TRUE_LABELS)
    unknown = sum(confusion[label]["unknown"] for label in TRUE_LABELS)
    known = len(pairs) - unknown
    return EvalSummary(
        confusion=confusion,
        total=len(pairs),
        accuracy_on_known=correct / known if known else None,
        unknown_rate=unknown / len(pairs),
        correct_overall=correct / len(pairs),
    )


def format_summary(model_key: str, summary: EvalSummary) -> str:
    header = "".join(f"{'pred ' + label:>14}" for label in PREDICTED_LABELS)
    rows = [
        f"{'true ' + true_label:<12}" + "".join(f"{summary.confusion[true_label][p]:>14}" for p in PREDICTED_LABELS)
        for true_label in TRUE_LABELS
    ]
    known_text = "n/a" if summary.accuracy_on_known is None else f"{summary.accuracy_on_known:.2f}"
    footer = (
        f"accuracy (known only): {known_text} | unknown rate: {summary.unknown_rate:.2f}"
        f" | correct overall: {summary.correct_overall:.2f}"
    )
    return "\n".join([f"model: {model_key} (n={summary.total})", " " * 12 + header, *rows, footer])


def evaluate_model(model_key: str, samples: list[Sample]) -> EvalSummary:
    classifier = ClothingClassifier(model_key, config.DEVICE)
    pairs: list[tuple[str, str]] = []
    for sample in samples:
        image = cv2.imread(str(sample.path))
        if image is None:
            raise EvalDataError(f"cannot read image {sample.path}")
        result = classifier.classify(image)
        pairs.append((sample.true_label, result.label))
        if result.label != sample.true_label:
            logger.info(
                "%s: %s is %s, predicted %s (warm_prob %.2f)",
                model_key, sample.path.name, sample.true_label, result.label, result.warm_prob,
            )
    return summarize(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(config.CLIP_MODELS), default=list(config.CLIP_MODELS))
    parser.add_argument("--eval-dir", type=Path, default=config.EVAL_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("jacket").setLevel(logging.INFO)
    logging.getLogger("jacket.classifier").setLevel(logging.WARNING)  # one line per crop is too noisy

    samples = load_samples(args.eval_dir)
    logger.info("loaded %d crops from %s", len(samples), args.eval_dir)
    for model_key in args.models:
        logger.info("\n%s", format_summary(model_key, evaluate_model(model_key, samples)))


if __name__ == "__main__":
    main()
