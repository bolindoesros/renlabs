"""Score data/bench/web against labels.csv. Run: python -m scripts.bench_web"""
import csv
import logging
from dataclasses import dataclass

import cv2

from jacket import config
from jacket.classifier import ClothingClassifier
from jacket.debug_view import draw_box
from jacket.person_detector import PersonDetector
from jacket.pipeline import ClothingPipeline

logger = logging.getLogger("jacket.bench_web")

WEB_DIR = config.DATA_DIR / "bench" / "web"
EXPECTED_LABEL = {"yes": "warm", "no": "light"}  # hoodie -> warm, t-shirt -> light


@dataclass(frozen=True)
class ImageOutcome:
    file: str
    pose: str
    people: str
    expected: str
    boxes_found: int
    primary_label: str  # label of the largest person
    primary_reason: str
    matching_people: int  # people whose label equals expected
    known_people: int  # people with a non-unknown label


def score_image(row: dict[str, str], detector: PersonDetector, classifier: ClothingClassifier, annotated_dir) -> ImageOutcome:
    frame = cv2.imread(str(WEB_DIR / row["file"]))
    if frame is None:
        raise FileNotFoundError(f"cannot read {row['file']}")
    boxes = detector.detect(frame)
    results = ClothingPipeline(classifier.classify).process(frame, boxes)
    expected = EXPECTED_LABEL[row["hoodie"]]
    for box, result in zip(boxes, results):
        draw_box(frame, box, result)
    cv2.imwrite(str(annotated_dir / row["file"].replace("/", "_")), frame)

    if boxes:
        largest = max(range(len(boxes)), key=lambda i: (boxes[i].x2 - boxes[i].x1) * (boxes[i].y2 - boxes[i].y1))
        primary_label, primary_reason = results[largest].label, results[largest].reason
    else:
        primary_label, primary_reason = "unknown", "no person detected"
    known = [r for r in results if r.label != "unknown"]
    return ImageOutcome(row["file"], row["pose"], row["people"], expected, len(boxes), primary_label,
                        primary_reason, sum(r.label == expected for r in known), len(known))


def summarize_single(outcomes: list[ImageOutcome]) -> str:
    total = len(outcomes)
    correct = sum(o.primary_label == o.expected for o in outcomes)
    unknown = sum(o.primary_label == "unknown" for o in outcomes)
    known = total - unknown
    known_text = f"{correct / known:.2f}" if known else "n/a"
    return f"n={total:<3} correct={correct:<3} wrong={known - correct:<3} unknown={unknown:<3} accuracy(known)={known_text} correct-overall={correct / total:.2f}"


def summarize_multi(outcomes: list[ImageOutcome]) -> str:
    matching = sum(o.matching_people for o in outcomes)
    known = sum(o.known_people for o in outcomes)
    people = sum(o.boxes_found for o in outcomes)
    known_text = f"{matching / known:.2f}" if known else "n/a"
    return f"images={len(outcomes):<3} people={people:<3} classified={known:<3} matching={matching:<3} agreement(known)={known_text}"


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("jacket").setLevel(logging.INFO)
    logging.getLogger("jacket.classifier").setLevel(logging.WARNING)
    with open(WEB_DIR / "labels.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    detector = PersonDetector(config.YOLO_WEIGHTS_PATH, config.YOLO_PERSON_CLASS_ID,
                              config.YOLO_MIN_CONFIDENCE, config.YOLO_IMAGE_SIZE, config.DEVICE)
    for model_key in config.CLIP_MODELS:
        classifier = ClothingClassifier(model_key, config.DEVICE)
        annotated_dir = WEB_DIR / "annotated" / model_key
        annotated_dir.mkdir(parents=True, exist_ok=True)
        outcomes = [score_image(row, detector, classifier, annotated_dir) for row in rows]
        with open(WEB_DIR / f"results_{model_key}.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(vars(outcomes[0])))
            writer.writeheader()
            writer.writerows(vars(o) for o in outcomes)

        single = [o for o in outcomes if o.people == "single"]
        lines = [f"== {model_key} ==", f"single-person, all     : {summarize_single(single)}"]
        for pose in ("sit", "stand"):
            lines.append(f"single-person, {pose:<8}: {summarize_single([o for o in single if o.pose == pose])}")
        for expected in ("warm", "light"):
            lines.append(f"single-person, {expected:<8}: {summarize_single([o for o in single if o.expected == expected])}")
        lines.append(f"multi-person, all      : {summarize_multi([o for o in outcomes if o.people == 'multi'])}")
        for o in single:
            if o.primary_label != o.expected:
                lines.append(f"  miss {o.file}: expected {o.expected}, got {o.primary_label} {o.primary_reason}")
        logger.info("\n".join(lines))


if __name__ == "__main__":
    main()
