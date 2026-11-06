import json
from pathlib import Path
from typing import Any

import pytest

from fc_distill import report
from fc_distill.report import MODELS, build, render
from tests.factories import make_example


def _write_run(path: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(summary))
    (path / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def _bfcl_block(n: int) -> dict[str, Any]:
    keys = ("parse_ok", "exact", "bfcl_exact", "names")
    block = {"n": n} | {k: 0.5 for k in keys} | {f"{k}_ci": [0.4, 0.6] for k in keys}
    return block


def test_build_scores_test_clean_from_the_test_run_and_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    examples = [make_example(i) for i in range(4)]
    monkeypatch.setattr(report, "load_split", lambda split, path: examples)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"test-clean": [0, 1, 2]}))
    metrics = ("parse_ok", "exact", "lenient_exact", "names", "arg_f1")
    for k, model in enumerate(MODELS):
        tag = model.id.replace("/", "_")
        # model k gets rows 0..k right
        records = [{"id": i} | {m: float(i <= k) for m in metrics} for i in range(4)]
        for r in records:
            r["exact"] = bool(r["exact"])
        _write_run(tmp_path / f"eval-test-{tag}", records, {"seconds": 1.0})
        bfcl_records = [{"id": f"simple_{i}", "exact": i <= k} for i in range(4)]
        bfcl_summary = {
            "overall": _bfcl_block(4),
            "slices": {"simple": _bfcl_block(4), "multiple_parallel": _bfcl_block(0)},
            "categories": {},
            "seconds": 2.0,
            "rows_calling_a_train_function_name": 0,
        }
        _write_run(tmp_path / f"eval-bfcl-{tag}", bfcl_records, bfcl_summary)

    data = build(tmp_path, splits)
    student = data["models"]["Qwen/Qwen3.5-0.8B"]
    assert student["test"]["n"] == 4
    assert student["test-clean"]["n"] == 3
    assert student["test-clean"]["exact"] == round(1 / 3, 4)
    first = data["mcnemar"][0]
    assert (first["eval"], first["n"], first["only_a_right"], first["only_b_right"]) == (
        "test-clean",
        3,
        0,
        1,
    )
    md = render(data)
    assert "## Starting line for Phase 2" in md
    assert "| Qwen/Qwen3.5-2B | teacher |" in md
