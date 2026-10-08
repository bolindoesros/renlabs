import pytest

from jacket.eval import EvalDataError, load_samples, summarize


def make_folders(root, warm_files, light_files):
    for label, names in (("warm", warm_files), ("light", light_files)):
        (root / label).mkdir(parents=True)
        for name in names:
            (root / label / name).write_bytes(b"")


def test_summarize_counts_confusion_matrix():
    pairs = [("warm", "warm"), ("warm", "light"), ("light", "light"), ("light", "unknown")]
    summary = summarize(pairs)
    assert summary.confusion["warm"] == {"warm": 1, "light": 1, "unknown": 0}
    assert summary.confusion["light"] == {"warm": 0, "light": 1, "unknown": 1}


def test_summarize_rates():
    pairs = [("warm", "warm"), ("warm", "light"), ("light", "light"), ("light", "unknown")]
    summary = summarize(pairs)
    assert summary.accuracy_on_known == pytest.approx(2 / 3)  # unknown left out
    assert summary.unknown_rate == 0.25
    assert summary.correct_overall == 0.5  # unknown counts as wrong


def test_summarize_all_unknown_has_no_accuracy():
    summary = summarize([("warm", "unknown"), ("light", "unknown")])
    assert summary.accuracy_on_known is None and summary.unknown_rate == 1.0


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([])


def test_load_samples_labels_by_folder_and_skips_other_files(tmp_path):
    make_folders(tmp_path, ["a.jpg", "notes.txt"], ["b.PNG"])
    samples = load_samples(tmp_path)
    assert [(s.path.name, s.true_label) for s in samples] == [("a.jpg", "warm"), ("b.PNG", "light")]


def test_load_samples_missing_folder_raises(tmp_path):
    (tmp_path / "warm").mkdir()
    (tmp_path / "warm" / "a.jpg").write_bytes(b"")  # light/ is the missing one
    with pytest.raises(EvalDataError, match="missing folder"):
        load_samples(tmp_path)


def test_load_samples_empty_folder_raises(tmp_path):
    make_folders(tmp_path, ["a.jpg"], [])
    with pytest.raises(EvalDataError, match="no images"):
        load_samples(tmp_path)
