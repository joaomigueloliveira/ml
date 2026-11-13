import math
from pathlib import Path

import numpy as np
import pytest

from fc_distill.teacher_topk import (
    RowTopk,
    answer_positions,
    coverage,
    load_topk,
    save_topk,
    top_k_entries,
)


def test_answer_positions_are_the_positions_with_a_label() -> None:
    # TRL's labels: -100 over the prompt, the token id over the completion.
    assert answer_positions([-100, -100, -100, 7, 8, 9]) == [3, 4, 5]


def test_top_k_entries_sorts_by_rank_and_keeps_a_written_token_outside_the_top_k_apart() -> None:
    # vLLM's entries at one position: the top k by rank, plus the written token (rank 9) if it
    # isn't among them.
    position = top_k_entries(
        token_ids=[11, 40, 12, 13],
        logprobs=[-1.2, -6.0, -0.5, -2.0],
        ranks=[2, 9, 1, 3],
        token=40,
        k=3,
    )

    assert position.ids == [12, 11, 13]
    assert position.logprobs == [-0.5, -1.2, -2.0]
    assert (position.token_logprob, position.token_rank) == (-6.0, 9)


def test_top_k_entries_finds_the_written_token_inside_the_top_k() -> None:
    position = top_k_entries(token_ids=[5, 6], logprobs=[-0.1, -2.5], ranks=[1, 2], token=6, k=2)

    assert position.ids == [5, 6]
    assert (position.token_logprob, position.token_rank) == (-2.5, 2)


def test_top_k_entries_needs_the_written_token() -> None:
    with pytest.raises(ValueError, match="written token 7"):
        top_k_entries(token_ids=[5, 6], logprobs=[-0.1, -2.5], ranks=[1, 2], token=7, k=2)


def _row(
    row_id: int, tokens: list[int], ids: list[list[int]], logprobs: list[list[float]]
) -> RowTopk:
    ranks = [row.index(t) + 1 if t in row else 99 for t, row in zip(tokens, ids, strict=True)]
    return RowTopk(
        row_id=row_id,
        positions=np.arange(10, 10 + len(tokens), dtype=np.int32),
        tokens=np.array(tokens, dtype=np.int32),
        ids=np.array(ids, dtype=np.int32),
        logprobs=np.array(logprobs, dtype=np.float32),
        token_logprob=np.array(
            [lp[r - 1] if r != 99 else -9.0 for lp, r in zip(logprobs, ranks, strict=True)],
            dtype=np.float32,
        ),
        token_rank=np.array(ranks, dtype=np.int32),
    )


_ROWS = [
    # Sure of its written token, and of the next one.
    _row(3, [1, 2], [[1, 9], [2, 9]], [[math.log(0.99), math.log(0.01)], [0.0, -30.0]]),
    # Prefers token 9 to the written 1: the top 2 hold 90%.
    _row(8, [1], [[9, 1]], [[math.log(0.6), math.log(0.3)]]),
]


def test_saved_top_k_loads_back_row_by_row(tmp_path: Path) -> None:
    path = tmp_path / "topk.npz"

    save_topk(path, _ROWS)
    loaded = load_topk(path)

    assert sorted(loaded) == [3, 8]
    for row in _ROWS:
        got = loaded[row.row_id]
        for field in ("positions", "tokens", "ids", "logprobs", "token_logprob", "token_rank"):
            np.testing.assert_array_equal(getattr(got, field), getattr(row, field))


def test_coverage_reports_top_1_agreement_and_the_mass_the_top_k_hold() -> None:
    stats = coverage(_ROWS)

    assert stats["rows"] == 2
    assert stats["positions"] == 3
    assert stats["written_is_top1"] == pytest.approx(2 / 3)
    assert stats["mass_min"] == pytest.approx(0.9)
    assert stats["mass_median"] == pytest.approx(1.0)
