import json
import sys
from pathlib import Path

import pytest

from fc_distill import evaluate
from tests.factories import make_example


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
    assert set(summary["rescored_by"]) == {"commit", "dirty", "at"}
    record = json.loads((tmp_path / "predictions.jsonl").read_text())
    assert record["exact"] is True


def test_model_run_refuses_a_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate, "git_state", lambda: {"commit": "abc", "dirty": True})
    monkeypatch.setattr(sys, "argv", ["fc-eval", "--model", "m"])
    with pytest.raises(SystemExit):
        evaluate.parse_args()
    monkeypatch.setattr(sys, "argv", ["fc-eval", "--model", "m", "--allow-dirty"])
    assert evaluate.parse_args().allow_dirty


def test_rescore_needs_no_clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate, "git_state", lambda: {"commit": "abc", "dirty": True})
    monkeypatch.setattr(sys, "argv", ["fc-eval", "--rescore", "runs/x"])
    assert evaluate.parse_args().rescore == Path("runs/x")
