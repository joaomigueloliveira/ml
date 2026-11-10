import json
import sys
from pathlib import Path
from typing import Any

import pytest

from fc_distill import evaluate
from fc_distill.checkpoints import Checkpoint
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


def test_log_to_wandb_puts_provenance_in_config_and_scores_under_the_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "eval-test-clean-Qwen_Qwen3.5-0.8B"
    run_dir.mkdir()
    summary = {
        "n": 2,
        "exact": 0.5,
        "exact_ci": [0.1, 0.9],
        "slices": {"call_pattern": {"single": {"n": 2, "exact": 0.5}}},
        "model": "m",
        "split": "test-clean",
        "seconds": 1.5,
        "provenance": PROVENANCE,
    }
    saved = json.dumps(summary)
    (run_dir / "summary.json").write_text(saved)
    tracker = FakeTracker()
    opened: dict[str, Any] = {}

    def start_run(**kwargs: Any) -> FakeTracker:
        opened.update(kwargs)
        return tracker

    monkeypatch.setattr(evaluate, "start_run", start_run)
    evaluate.log_to_wandb(run_dir, "sft")
    assert opened == {
        "enabled": True,
        "job_type": "eval",
        "group": "sft",
        "run_dir": run_dir,
        "name": "eval-test-clean-Qwen_Qwen3.5-0.8B",
        "tags": ["split:test-clean"],
    }
    assert tracker.configured == {"model": "m", "split": "test-clean", "provenance": PROVENANCE}
    assert tracker.summarised["test_clean/exact_lo"] == 0.1
    assert tracker.summarised["test_clean/seconds"] == 1.5
    assert tracker.summarised["test_clean/slice/call_pattern/single/exact"] == 0.5
    assert not any("provenance" in k or k.endswith("/model") for k in tracker.summarised)
    assert tracker.finished
    assert (run_dir / "summary.json").read_text() == saved


def test_wandb_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["fc-eval", "--rescore", "runs/x"])
    assert not evaluate.parse_args().wandb
    monkeypatch.setattr(sys, "argv", ["fc-eval", "--rescore", "runs/x", "--wandb"])
    assert evaluate.parse_args().wandb


def test_each_checkpoint_of_a_training_run_gets_its_own_eval_run() -> None:
    def ckpt(step: int) -> Checkpoint:
        run = Path("runs/train-sft-gold-r16-b4x4-n54561-gl-s0")
        return Checkpoint(run / f"checkpoint-{step}", "Qwen/Qwen3.5-0.8B", 16, run.name, step)

    names = {evaluate.run_name("validation", "Qwen/Qwen3.5-0.8B", ckpt(s)) for s in (500, 1000)}

    assert names == {
        "eval-validation-train-sft-gold-r16-b4x4-n54561-gl-s0-checkpoint-500",
        "eval-validation-train-sft-gold-r16-b4x4-n54561-gl-s0-checkpoint-1000",
    }
    # Without an adapter the name is the baselines' as before.
    assert evaluate.run_name("validation", "Qwen/Qwen3.5-0.8B", None) == (
        "eval-validation-Qwen_Qwen3.5-0.8B"
    )


def test_adapter_check_refuses_an_adapter_that_changed_too_few_outputs() -> None:
    base = ["same"] * 32
    texts = ["other"] * 6 + ["same"] * 26  # 6 of 32 changed, under the default 25%
    with pytest.raises(RuntimeError, match="changed 6 of 32 check rows"):
        evaluate.adapter_check(base, texts, evaluate.ADAPTER_MIN_CHANGED)


def test_adapter_check_with_a_lower_threshold_passes_and_records_what_it_saw() -> None:
    # An adapter checked offline to be loaded (its other splits changed ~70% of rows) can pass
    # with an explicit lower threshold; the counts go into the run's summary either way.
    base = ["same"] * 32
    texts = ["other"] * 6 + ["same"] * 26
    assert evaluate.adapter_check(base, texts, 0.15) == {
        "rows": 32,
        "changed": 6,
        "min_changed": 0.15,
    }


def test_check_rows_are_a_fixed_sample_spread_over_the_split() -> None:
    # The first rows of BFCL are all "simple" ones the raw model already gets right (#71), so the
    # adapter-load check samples rows from the whole split instead, the same rows on every run.
    rows = evaluate.check_rows(1000)
    assert rows == evaluate.check_rows(1000)
    assert len(rows) == evaluate.ADAPTER_CHECK_ROWS == len(set(rows))
    assert max(rows) > 500
    assert evaluate.check_rows(10) == list(range(10))
