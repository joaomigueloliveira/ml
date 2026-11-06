import json
import math
from pathlib import Path
from typing import Any

import pytest

from fc_distill import evaluate
from fc_distill.data import Example
from fc_distill.slices import (
    argument_source,
    call_pattern,
    distractors,
    query_numbers,
    slice_scores,
    value_in_query,
)
from tests.factories import make_example


def _calls(*names: str) -> list[dict[str, Any]]:
    return [{"name": n, "arguments": {}} for n in names]


def test_call_pattern_merges_mixed_into_multiple():
    assert call_pattern(make_example(1, answers=_calls("f"))) == "single"
    assert call_pattern(make_example(1, answers=_calls("f", "f"))) == "parallel"
    assert call_pattern(make_example(1, answers=_calls("f", "g"))) == "multiple"
    assert call_pattern(make_example(1, answers=_calls("f", "f", "g"))) == "multiple"


def _with_tools(n_tools: int, called: int) -> Example:
    ex = make_example(1, answers=_calls(*[f"t{i}" for i in range(called)]))
    ex["tools"] = [{"name": f"t{i}", "parameters": {}} for i in range(n_tools)]
    return ex


def test_distractor_buckets():
    assert distractors(_with_tools(1, 1)) == "0"
    assert distractors(_with_tools(3, 1)) == "1-2"
    assert distractors(_with_tools(5, 1)) == "3+"


def test_query_numbers_reads_thousands_separators_and_decimals():
    assert query_numbers("Invest $10,000 at 2.5% for 8 years") == {10000.0, 2.5, 8.0}


def test_value_in_query_strings_and_numbers():
    q = "Weather in  New York for 3 days"
    assert value_in_query("new york", q)
    assert value_in_query(3, q)
    assert value_in_query(3.0, q)
    assert not value_in_query("NYC", q)
    assert not value_in_query(0.03, "at 3% interest")


def test_value_in_query_recurses_and_ignores_bools_and_null():
    q = "Sort 5, 2 and 9"
    assert value_in_query([5, 2, 9], q)
    assert not value_in_query([5, 2, 7], q)
    assert value_in_query({"desc": True, "key": None}, q)


def test_argument_source():
    copied = make_example(1, query="factorial of 7", answers=[{"name": "f", "arguments": {"x": 7}}])
    derived = make_example(
        1, query="factorial of seven", answers=[{"name": "f", "arguments": {"x": 7}}]
    )
    no_args = make_example(1, answers=_calls("f"))
    assert argument_source(copied) == "from_query"
    assert argument_source(derived) == "not_in_query"
    assert argument_source(no_args) == "from_query"


def test_slice_scores_counts_flags_and_intervals():
    single = make_example(1, answers=_calls("f"))
    parallel = make_example(2, answers=_calls("f", "f"))
    examples = [single] * 4 + [parallel] * 2
    scores = [{"exact": e} for e in (True, True, True, False, False, True)]
    out = slice_scores(examples, scores, ["exact"], min_rows=3)
    single_block = out["call_pattern"]["single"]
    assert single_block["n"] == 4
    assert single_block["too_small"] is False
    assert single_block["exact"] == 0.75
    assert single_block["exact_ci"] == round(1.96 * math.sqrt(0.75 * 0.25 / 4), 4)
    assert out["call_pattern"]["parallel"]["too_small"] is True
    assert out["call_pattern"]["multiple"] == {
        "n": 0,
        "too_small": True,
        "exact": None,
        "exact_ci": None,
    }
    assert sum(b["n"] for b in out["distractors"].values()) == 6


def test_rescore_rewrites_summary_from_saved_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ex = make_example(7, answers=[{"name": "f", "arguments": {"x": 1}}])
    monkeypatch.setattr(evaluate, "load_split", lambda split: [ex])
    (tmp_path / "summary.json").write_text(
        json.dumps({"model": "m", "split": "test", "seconds": 1.0, "exact": 0.0})
    )
    prediction = json.dumps([{"name": "f", "arguments": {"x": 1}}])
    (tmp_path / "predictions.jsonl").write_text(
        json.dumps({"id": 7, "prediction": prediction}) + "\n"
    )
    evaluate.rescore(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["exact"] == 1
    assert summary["lenient_exact"] == 1
    assert (summary["model"], summary["split"], summary["seconds"]) == ("m", "test", 1.0)
    assert summary["slices"]["call_pattern"]["single"]["n"] == 1
    record = json.loads((tmp_path / "predictions.jsonl").read_text())
    assert record["exact"] is True
