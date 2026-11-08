import json
from pathlib import Path

import pytest

from fc_distill.data import to_messages
from fc_distill.opd import init_adapter, opd_record, probe_exact, probe_rows, rollout_stats
from tests.factories import make_example


def test_opd_record_is_the_eval_prompt_with_no_answer() -> None:
    ex = make_example(7, query="weather in Porto", answers=[{"name": "f", "arguments": {"x": 42}}])
    record = opd_record(ex)
    assert record == {"prompt": to_messages(ex)}
    # Label-free (ADR 0007): the gold answer appears nowhere in what the trainer sees.
    assert '"x": 42' not in json.dumps(record)


def test_rollout_stats_count_parsed_calls() -> None:
    texts = [
        '[{"name": "f", "arguments": {"x": 1}}]',
        '```json\n[{"name": "f", "arguments": {}}]\n```',  # fenced: the scorer accepts it
        '[{"name": "f", "arguments": {"x": 1}}], {',  # bracket slip
        "I can't help with that.",
    ]
    assert rollout_stats(texts) == {"rollout/parse_ok": 0.5}


def test_probe_exact_scores_each_output_against_its_own_row() -> None:
    rows = [
        make_example(1, answers=[{"name": "f", "arguments": {"x": 1}}]),
        make_example(2, answers=[{"name": "f", "arguments": {"x": 2}}]),
    ]
    # The second output answers row 1, not row 2: wrong for its own row.
    outputs = ['[{"name": "f", "arguments": {"x": 1}}]', '[{"name": "f", "arguments": {"x": 1}}]']
    assert probe_exact(outputs, rows) == 0.5


def test_probe_rows_are_the_same_rows_every_time() -> None:
    validation = [make_example(i) for i in range(1000)]
    first = probe_rows(validation)
    assert len(first) == 200
    assert [ex["id"] for ex in probe_rows(list(reversed(validation)))] == [ex["id"] for ex in first]


def _checkpoint(tmp_path: Path, base: str = "Qwen/Qwen3.5-0.8B") -> Path:
    path = tmp_path / "train-sft-teacher-r16-b4x4-n52424-gl-s0" / "checkpoint-1500"
    path.mkdir(parents=True)
    config = {"base_model_name_or_path": base, "r": 8, "lora_alpha": 24}
    (path / "adapter_config.json").write_text(json.dumps(config))
    return path


def test_init_adapter_names_where_the_run_starts_and_keeps_the_adapters_rank(
    tmp_path: Path,
) -> None:
    # B then D (ADR 0008): the run is named after B's run and step, so it never reads as method D.
    start = init_adapter(_checkpoint(tmp_path), "Qwen/Qwen3.5-0.8B")
    assert start == {"tag": "from-sft-teacher-r16-b4x4-n52424-gl-s0-1500", "r": 8, "alpha": 24}


def test_init_adapter_refuses_an_adapter_trained_on_another_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"Qwen/Qwen3\.5-2B"):
        init_adapter(_checkpoint(tmp_path, base="Qwen/Qwen3.5-2B"), "Qwen/Qwen3.5-0.8B")
