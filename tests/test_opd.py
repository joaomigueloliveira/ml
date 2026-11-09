import json
from pathlib import Path

import pytest

from fc_distill.data import to_messages
from fc_distill.opd import (
    checkpoint_action,
    init_adapter,
    opd_record,
    probe_exact,
    probe_rows,
    rollout_stats,
    rows_trained,
)
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


@pytest.mark.parametrize(
    ("step", "action"),
    [
        (50, (False, False)),
        (100, (True, False)),
        (500, (True, False)),
        (700, (False, False)),
        (1000, (True, True)),
    ],
)
def test_checkpoint_action_saves_at_the_listed_steps_and_stops_at_the_last(
    step: int, action: tuple[bool, bool]
) -> None:
    # B then D's first look (#58): the schedule spans the epoch, but the run saves early and often,
    # and stops at 1000 with a checkpoint it can resume from.
    assert checkpoint_action(step, save_at=(100, 200, 300, 400, 500, 1000), stop_at=1000) == action


def test_checkpoint_action_without_a_list_leaves_saving_to_save_steps() -> None:
    assert checkpoint_action(500, save_at=(), stop_at=None) == (False, False)


def test_rows_trained_stops_counting_where_the_run_stopped() -> None:
    # B then D (#58) stopped at step 1000 of a 54,561-row epoch: 16,000 rows trained, not 54,561.
    assert rows_trained(steps=1000, rows_per_step=16, epoch_rows=54561) == 16000
    assert rows_trained(steps=3411, rows_per_step=16, epoch_rows=54561) == 54561
