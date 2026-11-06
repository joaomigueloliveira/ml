import pytest

from fc_distill.quality import (
    base_type,
    code_arguments,
    conflicting_labels,
    exact_duplicates,
    explicit_defaults,
    missing_required,
    query_template,
    type_mismatches,
    unknown_calls,
    without_defaults,
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


def test_missing_required_ignores_optional_parameters_only():
    # xLAM gives most required parameters an example value under "default"; that's not optional.
    params = {
        "x": {"type": "int"},
        "y": {"type": "int, optional"},
        "z": {"type": "int", "default": 3},
    }
    ex = make_example(1, params=params, answers=[{"name": "f", "arguments": {}}])
    assert missing_required([ex]) == [(1, "f", "x"), (1, "f", "z")]


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


def test_conflicting_labels_ignores_tool_order():
    t1 = {"name": "f", "description": "", "parameters": {"x": {"type": "int"}}}
    t2 = {"name": "g", "description": "", "parameters": {}}
    a, b = make_example(1), make_example(2, answers=[{"name": "f", "arguments": {"x": 2}}])
    a["tools"], b["tools"] = [t1, t2], [t2, t1]
    assert conflicting_labels([a, b]) == [[1, 2]]


def test_unknown_calls_flags_unoffered_functions_and_undeclared_arguments():
    exs = [
        make_example(1, answers=[{"name": "g", "arguments": {}}]),
        make_example(2, answers=[{"name": "f", "arguments": {"x": 1, "y": 2}}]),
        make_example(3),
    ]
    assert unknown_calls(exs) == [(1, "g", None), (2, "f", "y")]


def test_explicit_defaults_flags_default_values_and_nulls_of_optional_params_only():
    params = {
        "a": {"type": "str, optional", "default": "game"},
        "b": {"type": "str, optional"},
        "c": {"type": "int"},
        "d": {"type": "str", "default": "example"},  # required, "default" is only an example
        "n": {"type": "int, optional", "default": "17"},  # xLAM often stringifies defaults
    }
    calls = [
        {"name": "f", "arguments": {"a": "game", "b": None, "c": 1, "n": 17}},  # a, b, n flagged
        {"name": "f", "arguments": {"a": "beta", "c": None, "d": "example"}},  # c, d required
    ]
    issues = explicit_defaults([make_example(1, params=params, answers=calls)])
    assert [(i.argument, i.value) for i in issues] == [("a", "game"), ("b", None), ("n", 17)]


def test_query_template_masks_numbers_strings_and_punctuation():
    assert query_template("Calculate the factorial of 5.") == query_template(
        "calculate the  factorial of 10"
    )
    assert query_template("Weather in 'Paris'?") == query_template('weather in "Lisbon"')
    assert query_template("factorial of 5") != query_template("fibonacci of 5")


def test_without_defaults_drops_only_optional_arguments_at_their_default_or_null():
    params = {
        "x": {"type": "int"},
        "unit": {"type": "str, optional", "default": "km"},
        "note": {"type": "str, optional"},
        "q": {"type": "str", "default": "google.com"},
    }
    tools = [{"name": "f", "parameters": params}]
    calls = [{"name": "f", "arguments": {"x": None, "unit": "km", "note": None, "q": "google.com"}}]
    assert without_defaults(calls, tools) == [
        {"name": "f", "arguments": {"x": None, "q": "google.com"}}
    ]  # type: ignore[arg-type]
