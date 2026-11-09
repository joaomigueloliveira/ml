import json
from pathlib import Path
from typing import Any

import pytest

from fc_distill import methods
from fc_distill.methods import METHODS, build, gains, render, select, validity
from tests.factories import make_example

TOOL = {
    "name": "weather",
    "description": "",
    "parameters": {
        "city": {"type": "str", "description": ""},
        "units": {"type": "str, optional", "description": "", "default": "C"},
    },
}


def _ckpt(step: int, exact: float, ci: tuple[float, float]) -> dict[str, Any]:
    return {"step": step, "exact": exact, "exact_ci": list(ci)}


def test_select_takes_the_best_and_breaks_an_exact_tie_towards_the_earlier_step() -> None:
    checkpoints = [
        _ckpt(500, 0.60, (0.57, 0.63)),
        _ckpt(1000, 0.70, (0.67, 0.73)),
        _ckpt(1500, 0.70, (0.67, 0.73)),
        _ckpt(2000, 0.68, (0.65, 0.71)),
    ]
    assert select(checkpoints) == (1000, [1500, 2000])


def test_gains_against_the_teacher_gap_and_against_method_a() -> None:
    # raw 50, teacher 70, A 90: a method at 80 recovers 150% of the gap and 75% of A's gain
    assert gains(0.80, 0.50, 0.70, 0.90) == {"gap_recovered": 1.5, "share_of_a": 0.75}


@pytest.mark.parametrize(
    ("prediction", "verdict"),
    [
        ('[{"name": "weather", "arguments": {"city": "Porto"}}]', "valid"),
        ('[{"name": "weather", "arguments": {"units": "F"}}]', "missing required"),
        ('[{"name": "forecast", "arguments": {"city": "Porto"}}]', "unknown function"),
        ("Sure! The weather in Porto", "unparsed"),
    ],
)
def test_validity_checks_teacher_calls_without_gold(prediction: str, verdict: str) -> None:
    assert validity(prediction, [TOOL]) == verdict


def _write_run(path: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(summary))
    (path / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def _rows(ids: range, right: set[int]) -> list[dict[str, Any]]:
    return [
        {"id": i, "parse_ok": True, "exact": i in right, "lenient_exact": i in right} for i in ids
    ]


def test_build_selects_per_method_compares_and_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"test-clean": [0, 1, 2, 3]}))
    val = range(10)
    # validation: raw 0.8B gets 2 right, the 2B 6, A 10, B's best checkpoint 7
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-0.8B", _rows(val, {0, 1}), {})
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-2B", _rows(val, set(range(6))), {})
    a, b, *_ = METHODS
    for method, scores in ((a, {500: 8, 1000: 10}), (b, {500: 7, 1000: 6})):
        for step, k in scores.items():
            _write_run(
                tmp_path / f"eval-validation-{method.train_run}-checkpoint-{step}",
                _rows(val, set(range(k))),
                {"adapter": {"gpu_hours": step / 1000}},
            )
    # official runs exist only for A's selected checkpoint and the raw 0.8B
    test = range(6)
    _write_run(tmp_path / f"eval-test-{a.train_run}-checkpoint-1000", _rows(test, set(test)), {})
    _write_run(tmp_path / "eval-test-Qwen_Qwen3.5-0.8B", _rows(test, {0}), {})
    teacher = [
        {"id": 0, "prediction": '[{"name": "weather", "arguments": {"city": "P"}}]'}
        | {"exact": True, "lenient_exact": True},
        {"id": 1, "prediction": "no", "exact": False, "lenient_exact": False},
    ]
    _write_run(tmp_path / "eval-train-Qwen_Qwen3.5-2B", teacher, {"seconds": 3600.0})
    examples = [make_example(i) | {"tools": [TOOL]} for i in range(2)]
    monkeypatch.setattr(methods, "load_split", lambda split, path: examples)

    data = build(tmp_path, splits)

    assert data["methods"]["A"]["selected"] == 1000
    assert data["methods"]["B"]["selected"] == 500
    assert data["methods"]["A"]["official"]["test-clean"]["n"] == 4
    assert "test" not in data["methods"]["B"]["official"]
    # B at 70: (70 - 20) / (60 - 20) of the gap, (70 - 20) / (100 - 20) of A's gain
    assert data["headline"]["B/validation"] == {"gap_recovered": 1.25, "share_of_a": 0.625}
    a_vs_b = next(c for c in data["mcnemar"] if (c["a"], c["b"]) == ("A", "B"))
    assert (a_vs_b["only_a_right"], a_vs_b["only_b_right"]) == (3, 0)
    assert data["teacher"]["variants"]["unfiltered"] == {"kept": 1, "exact": 1.0}
    assert data["teacher"]["unparsed"] == 1
    md = render(data)
    assert "**B**, SFT on teacher outputs (step 500)" in md
    assert "Inside its interval" in md


def test_matched_cost_scores_each_label_free_method_nearest_bs_gpu_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"test-clean": []}))
    val = range(10)
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-0.8B", _rows(val, {0}), {})
    _write_run(tmp_path / "eval-validation-Qwen_Qwen3.5-2B", _rows(val, set(range(5))), {})
    by_key = {m.key: m for m in METHODS}
    # B's best is step 1000 at 3.1 GPU-h. D's checkpoints sit at 2.0, 4.0 and 6.0 GPU-h, and D's
    # best is the last; the matched-cost point is 4.0 (|4.0 - 3.1| < |2.0 - 3.1|), not the best.
    runs = {
        "B": {500: (1.6, 5), 1000: (3.1, 7)},
        "D": {500: (2.0, 6), 1000: (4.0, 5), 1500: (6.0, 8)},
    }
    for key, checkpoints in runs.items():
        for step, (hours, k) in checkpoints.items():
            _write_run(
                tmp_path / f"eval-validation-{by_key[key].train_run}-checkpoint-{step}",
                _rows(val, set(range(k))),
                {"adapter": {"gpu_hours": hours}},
            )
    monkeypatch.setattr(methods, "load_split", lambda split, path: [])

    data = build(tmp_path, splits)

    assert data["methods"]["D"]["selected"] == 1500
    matched = data["matched_cost"]["D"]
    assert (matched["step"], matched["gpu_hours"], matched["exact"]) == (1000, 4.0, 0.5)
    assert (matched["vs_b"]["only_a_right"], matched["vs_b"]["only_b_right"]) == (2, 0)
    assert "B" not in data["matched_cost"]
    md = render(data)
    assert "**D**, on-policy distillation (step 1500)" in md
    assert "| D | 1000 | 4.00 |" in md
