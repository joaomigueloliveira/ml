from collections import Counter

import pytest

from fc_distill.eda import (
    call_pattern,
    ci_half_width,
    declared_type_counts,
    example_stats,
    percentiles,
    train_overlap,
)
from tests.factories import make_example


def test_call_pattern():
    one = {"name": "f", "arguments": {}}
    other = {"name": "g", "arguments": {}}
    assert call_pattern(make_example(1, answers=[one])) == "single"
    assert call_pattern(make_example(1, answers=[one, one])) == "parallel"
    assert call_pattern(make_example(1, answers=[one, other])) == "multiple"
    assert call_pattern(make_example(1, answers=[one, one, other])) == "mixed"


def test_example_stats_counts_calls_and_arguments():
    ex = make_example(7, query="abc", answers=[{"name": "f", "arguments": {"x": 1, "y": 2}}] * 2)
    (stats,) = example_stats([ex])
    assert stats == {
        "id": 7,
        "calls": 2,
        "distinct_functions_called": 1,
        "tools_offered": 1,
        "arguments": 4,
        "query_chars": 3,
    }


def test_declared_type_counts_uses_base_types():
    ex = make_example(1, params={"a": {"type": "List[int]"}, "b": {"type": "str, optional"}})
    assert declared_type_counts([ex]) == Counter({"list": 1, "str": 1})


def test_percentiles():
    assert percentiles(list(range(1, 101))) == {"p50": 50, "p90": 90, "p99": 99, "max": 100}


def test_percentiles_of_a_single_value():
    assert percentiles([7]) == {"p50": 7, "p90": 7, "p99": 7, "max": 7}


def test_ci_half_width_matches_the_adr_figure_for_2000_rows():
    assert ci_half_width(2000) == pytest.approx(2.19, abs=0.01)


def test_train_overlap_counts_unseen_functions_and_shared_templates():
    train = [make_example(1, query="factorial of 5")]
    held_out = [
        make_example(2, query="Factorial of 10."),
        make_example(3, query="something else", answers=[{"name": "g", "arguments": {}}]),
    ]
    assert train_overlap(train, held_out) == {
        "rows": 2,
        "calls_function_unseen_in_train": 1,
        "query_template_in_train": 1,
    }
