import pytest

from fc_distill.quality import (
    base_type,
    code_arguments,
    conflicting_labels,
    exact_duplicates,
    missing_required,
    type_mismatches,
)
from tests.factories import make_example


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("str", "str"),
        ("str, optional", "str"),
        ("List[int]", "list"),
        ("Tuple[float, float]", "tuple"),
        ("Callable[[float], float]", "callable"),
        ("", ""),
    ],
)
def test_base_type(declared, expected):
    assert base_type(declared) == expected


def test_type_mismatch_flags_float_for_int_and_bool_for_int():
    exs = [
        make_example(1, answers=[{"name": "f", "arguments": {"x": 0.3}}]),
        make_example(2, answers=[{"name": "f", "arguments": {"x": True}}]),
        make_example(3, answers=[{"name": "f", "arguments": {"x": 4}}]),
    ]
    assert [i.row_id for i in type_mismatches(exs)] == [1, 2]


def test_float_parameter_accepts_int():
    ex = make_example(1, params={"x": {"type": "float"}})
    assert type_mismatches([ex]) == []


def test_code_argument_is_an_expression_not_a_quoted_literal():
    params = {"x": {"type": "float"}}
    exs = [
        make_example(1, params=params, answers=[{"name": "f", "arguments": {"x": "math.pi / 2"}}]),
        make_example(2, params=params, answers=[{"name": "f", "arguments": {"x": "5"}}]),
    ]
    assert [i.row_id for i in code_arguments(exs)] == [1]


def test_code_strings_in_text_and_callable_parameters_are_expected():
    exs = [
        make_example(
            1, params={"x": {"type": "str"}}, answers=[{"name": "f", "arguments": {"x": "a*b"}}]
        ),
        make_example(
            2,
            params={"x": {"type": "Callable[[float], float]"}},
            answers=[{"name": "f", "arguments": {"x": "lambda v: v"}}],
        ),
    ]
    assert code_arguments(exs) == []


def test_missing_required_ignores_optional_and_defaulted_parameters():
    params = {
        "x": {"type": "int"},
        "y": {"type": "int, optional"},
        "z": {"type": "int", "default": 3},
    }
    ex = make_example(1, params=params, answers=[{"name": "f", "arguments": {}}])
    assert missing_required([ex]) == [(1, "f", "x")]


def test_exact_duplicates_groups_identical_rows_only():
    exs = [make_example(1), make_example(2), make_example(3, query="other")]
    assert exact_duplicates(exs) == [[1, 2]]


def test_conflicting_labels_same_query_and_tools_different_answers():
    a = [{"name": "f", "arguments": {"x": 1}}]
    b = [{"name": "f", "arguments": {"x": 2}}]
    exs = [
        make_example(1, "Swap them", a),
        make_example(2, "swap  THEM", b),
        make_example(3, "unrelated", a),
    ]
    assert conflicting_labels(exs) == [[1, 2]]


def test_same_answer_in_another_order_is_not_a_conflict():
    calls = [{"name": "f", "arguments": {"x": 1}}, {"name": "f", "arguments": {"x": 2}}]
    exs = [make_example(1, answers=calls), make_example(2, answers=calls[::-1])]
    assert conflicting_labels(exs) == []


@pytest.mark.parametrize("value", ["movie", "N/A", "inf"])
def test_plain_words_for_numeric_parameters_are_mismatches_not_code(value):
    ex = make_example(1, answers=[{"name": "f", "arguments": {"x": value}}])
    assert code_arguments([ex]) == []
    assert [i.row_id for i in type_mismatches([ex])] == [1]


@pytest.mark.parametrize("value", ["math.pi / 2", "[0.015] * 12", "(float('inf'), 0)", "math.e"])
def test_expressions_are_code(value):
    ex = make_example(1, params={"x": {"type": "List[float]"}})
    ex["answers"] = [{"name": "f", "arguments": {"x": value}}]
    assert [i.row_id for i in code_arguments([ex])] == [1]
