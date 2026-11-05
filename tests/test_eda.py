from collections import Counter

from fc_distill.eda import call_pattern, declared_type_counts, example_stats, percentiles
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
    assert percentiles(list(range(1, 101))) == {"p50": 51, "p90": 91, "p99": 100, "max": 100}
