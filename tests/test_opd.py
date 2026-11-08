import json

from fc_distill.data import to_messages
from fc_distill.opd import opd_record, probe_exact, probe_rows, rollout_stats
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
