import json
import sys
from pathlib import Path
from typing import Any

import pytest

from fc_distill import evaluate
from tests.factories import make_example
from tests.fakes import FakeTracker


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


PROVENANCE = {"commit": "abc", "dirty": False, "model_revision": "r1"}


def test_log_eval_sends_scores_slices_and_provenance() -> None:
    summary = {
        "n": 3,
        "exact": 0.5,
        "exact_ci": [0.2, 0.8],
        "slices": {
            "call_pattern": {
                "single": {"too_small": False, "n": 2, "exact": 1.0, "exact_ci": [0.3, 1.0]},
                "parallel": {"too_small": True, "n": 1, "exact": 0.0, "exact_ci": [0.0, 0.8]},
                "multiple": {"too_small": True, "n": 0, "exact": None, "exact_ci": None},
            }
        },
        "model": "Qwen/Qwen3.5-0.8B",
        "split": "test",
        "seconds": 1.5,
        "provenance": PROVENANCE,
    }
    tracker = FakeTracker()
    evaluate.log_eval(tracker, summary)
    assert tracker.configured == {
        "model": "Qwen/Qwen3.5-0.8B",
        "split": "test",
        "provenance": PROVENANCE,
    }
    assert tracker.summarised["exact"] == 0.5
    assert tracker.summarised["seconds"] == 1.5
    assert tracker.summarised["slices"]["call_pattern"]["single"]["exact"] == 1.0
    assert "provenance" not in tracker.summarised
    assert tracker.tables["slices"] == (
        ["group", "slice", "too_small", "n", "exact", "exact_lo", "exact_hi"],
        [
            ["call_pattern", "single", False, 2, 1.0, 0.3, 1.0],
            ["call_pattern", "parallel", True, 1, 0.0, 0.0, 0.8],
            ["call_pattern", "multiple", True, 0, None, None, None],
        ],
    )


def test_log_eval_reads_bfcl_slices_and_categories() -> None:
    summary = {
        "overall": {"n": 2, "bfcl_exact": 0.5},
        "slices": {"simple": {"n": 1, "bfcl_exact": 1.0}},
        "categories": {"parallel": {"n": 1, "bfcl_exact": 0.0}},
        "model": "m",
        "split": "bfcl",
        "provenance": PROVENANCE,
        "rescored_by": {"commit": "def", "dirty": False, "at": "2026-11-06T00:00:00+00:00"},
    }
    tracker = FakeTracker()
    evaluate.log_eval(tracker, summary)
    assert tracker.configured["rescored_by"]["commit"] == "def"
    assert tracker.summarised["overall"] == {"n": 2, "bfcl_exact": 0.5}
    assert tracker.tables["slices"] == (
        ["group", "slice", "n", "bfcl_exact"],
        [["slices", "simple", 1, 1.0], ["categories", "parallel", 1, 0.0]],
    )


def test_relog_mirrors_a_saved_run_without_changing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "eval-test-Qwen_Qwen3.5-0.8B"
    run_dir.mkdir()
    saved = json.dumps({"exact": 0.5, "model": "m", "split": "test", "provenance": PROVENANCE})
    (run_dir / "summary.json").write_text(saved)
    tracker = FakeTracker()
    opened: dict[str, Any] = {}

    def start_run(**kwargs: Any) -> FakeTracker:
        opened.update(kwargs)
        return tracker

    monkeypatch.setattr(evaluate, "start_run", start_run)
    monkeypatch.setattr(evaluate, "git_state", lambda: {"commit": "abc", "dirty": True})
    monkeypatch.setattr(
        sys, "argv", ["fc-eval", "--relog", str(run_dir), "--wandb-group", "baseline"]
    )
    evaluate.main()
    assert opened == {
        "enabled": True,
        "job_type": "eval",
        "group": "baseline",
        "run_dir": run_dir,
        "name": "eval-test-Qwen_Qwen3.5-0.8B",
        "tags": ["split:test"],
    }
    assert tracker.configured["provenance"] == PROVENANCE
    assert tracker.summarised["exact"] == 0.5
    assert tracker.finished
    assert (run_dir / "summary.json").read_text() == saved


def test_wandb_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["fc-eval", "--rescore", "runs/x"])
    assert not evaluate.parse_args().wandb
    monkeypatch.setattr(sys, "argv", ["fc-eval", "--rescore", "runs/x", "--wandb"])
    assert evaluate.parse_args().wandb
