import json

import pytest

from fc_distill.audit import (
    agreement,
    auto_verdict,
    clean_ids,
    only_defaults_differ,
    pre_tags,
    prepare,
    read_labels,
    review_sample,
    summary,
    write_labels,
    write_test_clean,
)
from tests.factories import make_example

OPTIONAL = {"x": {"type": "int"}, "unit": {"type": "str, optional", "default": "km"}}


def _row(id, verdict="", category=""):
    return {
        "id": id,
        "verdict": verdict,
        "category": category,
        "reason": "",
        "models_wrong": "m",
        "tags": "",
    }


def test_only_defaults_differ_ignores_optional_arguments_at_their_default():
    ex = make_example(1, params=OPTIONAL, answers=[{"name": "f", "arguments": {"x": 1}}])
    with_default = [{"name": "f", "arguments": {"x": 1, "unit": "km"}}]
    other_value = [{"name": "f", "arguments": {"x": 1, "unit": "mi"}}]
    assert only_defaults_differ(with_default, ex)
    assert not only_defaults_differ(other_value, ex)
    assert not only_defaults_differ(None, ex)


def test_auto_verdict_marks_code_valued_labels_wrong():
    ex = make_example(
        1, params={"x": {"type": "float"}}, answers=[{"name": "f", "arguments": {"x": "math.pi"}}]
    )
    assert auto_verdict(ex, [None])[0] == "label_wrong"  # type: ignore[index]


def test_auto_verdict_needs_every_wrong_model_to_differ_only_by_defaults():
    ex = make_example(1, params=OPTIONAL, answers=[{"name": "f", "arguments": {"x": 1}}])
    by_default = [{"name": "f", "arguments": {"x": 1, "unit": "km"}}]
    wrong_value = [{"name": "f", "arguments": {"x": 2}}]
    assert auto_verdict(ex, [by_default])[:2] == ("label_sound", "default made explicit")  # type: ignore[index]
    assert auto_verdict(ex, [by_default, wrong_value]) is None


def test_pre_tags_flags_percentages_and_explicit_defaults():
    ex = make_example(
        1,
        query="grow 3% a year",
        params=OPTIONAL,
        answers=[{"name": "f", "arguments": {"x": 3, "unit": "km"}}],
    )
    assert pre_tags(ex) == ["explicit_default", "percentage"]


def test_prepare_audits_rows_any_model_gets_wrong_and_keeps_existing_verdicts():
    test = [make_example(i) for i in range(3)]
    right, wrong = (True, '[{"name": "f", "arguments": {"x": 1}}]'), (False, "nonsense")
    predictions = {
        "small": {0: wrong, 1: wrong, 2: right},
        "big": {0: right, 1: wrong, 2: right},
    }
    existing = {
        0: _row(0, "label_wrong") | {"reason": "auto: stale rule"},
        1: _row(1, "label_wrong") | {"reason": "kept"},
    }
    rows = prepare(test, predictions, existing)
    assert [(r["id"], r["models_wrong"]) for r in rows] == [(0, "small"), (1, "small;big")]
    assert rows[0]["verdict"] == ""  # automatic verdicts are recomputed, not kept
    assert (rows[1]["verdict"], rows[1]["reason"]) == ("label_wrong", "kept")


def test_clean_ids_drops_ambiguous_and_wrong_and_requires_every_verdict():
    labels = {1: _row(1, "label_sound"), 2: _row(2, "label_ambiguous"), 3: _row(3, "label_wrong")}
    assert clean_ids([0, 1, 2, 3], labels) == [0, 1]
    with pytest.raises(ValueError, match="no verdict"):
        clean_ids([0, 1], {1: _row(1)})


def test_summary_reports_the_ceiling():
    labels = {1: _row(1, "label_sound", "default made explicit"), 2: _row(2, "label_wrong")}
    s = summary([0, 1, 2, 3], labels)
    assert (s["test_clean_rows"], s["ceiling"]) == (3, 0.75)
    assert s["categories"] == {"default made explicit": 1}


def test_labels_round_trip_and_reject_unknown_verdicts(tmp_path):
    path = tmp_path / "labels.csv"
    write_labels([_row(5, "label_sound")], path)
    assert read_labels(path) == {5: _row(5, "label_sound")}
    write_labels([_row(5, "typo")], path)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="unknown verdict"):
        read_labels(path)


def test_write_test_clean_adds_the_key(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"test": [1, 2]}))
    write_test_clean([2], path)
    assert json.loads(path.read_text()) == {"test": [1, 2], "test-clean": [2]}


def test_review_sample_is_fixed_and_drawn_from_audited_rows():
    labels = {i: _row(i, "label_sound") for i in range(100)}
    sample = review_sample(labels, n=30, seed=0)
    assert sample == review_sample(labels, n=30, seed=0)
    assert len(set(sample)) == 30
    assert set(sample) <= set(labels)


def test_agreement_separates_verdict_and_soundness_disagreements():
    labels = {1: _row(1, "label_sound"), 2: _row(2, "label_wrong"), 3: _row(3, "label_ambiguous")}
    review = {1: "label_sound", 2: "label_ambiguous", 3: "label_sound", 4: ""}
    result = agreement(labels, review)
    assert result == {
        "reviewed": 3,
        "verdict_disagreements": [2, 3],
        "soundness_disagreements": [3],
    }
