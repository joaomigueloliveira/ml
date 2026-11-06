import json
from typing import Any

import pytest

from fc_distill.bfcl import (
    BfclRow,
    PossibleCall,
    calls_match,
    normalize_string,
    parse_row,
    score,
    seen_in_train,
    summarize,
    to_xlam_tool,
    type_string,
)

AREA: PossibleCall = {"area": {"base": [10], "height": [5], "unit": ["units", ""]}}


def _pred(*calls: dict[str, Any]) -> str:
    return json.dumps(list(calls))


def test_converts_spec_to_xlam_style():
    fn = {
        "name": "area",
        "description": "Area of a triangle.",
        "parameters": {
            "type": "dict",
            "properties": {
                "base": {"type": "integer", "description": "Base."},
                "unit": {"type": "string", "description": "Unit.", "default": "units"},
                "shape": {"type": "string", "description": "Shape.", "enum": ["a", "b"]},
                "sides": {"type": "array", "items": {"type": "float"}, "description": "Sides."},
                "where": {
                    "type": "dict",
                    "description": "Filter.",
                    "properties": {"school": {"type": "string", "description": "School."}},
                    "required": ["school"],
                },
            },
            "required": ["base", "shape"],
        },
    }
    params = to_xlam_tool(fn)["parameters"]
    assert params["base"] == {"description": "Base.", "type": "int"}
    assert params["unit"] == {"description": "Unit.", "type": "str, optional", "default": "units"}
    assert params["shape"] == {"description": "Shape. One of: a, b.", "type": "str"}
    assert params["sides"]["type"] == "List[float], optional"
    assert params["where"]["type"] == "Dict, optional"
    assert params["where"]["properties"] == {"school": {"description": "School.", "type": "str"}}


def test_nested_array_type_string():
    spec = {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}
    assert type_string(spec) == "List[List[int]]"
    assert type_string({"type": "array"}) == "List"


def test_optional_argument_may_be_left_out_or_passed():
    assert score(_pred({"name": "area", "arguments": {"base": 10, "height": 5}}), [AREA])["exact"]
    with_unit = {"name": "area", "arguments": {"base": 10, "height": 5, "unit": "units"}}
    assert score(_pred(with_unit), [AREA])["exact"]


def test_required_argument_left_out_fails():
    assert not score(_pred({"name": "area", "arguments": {"base": 10}}), [AREA])["exact"]


def test_unknown_argument_or_wrong_value_fails():
    extra = {"name": "area", "arguments": {"base": 10, "height": 5, "color": "red"}}
    assert not score(_pred(extra), [AREA])["exact"]
    wrong = {"name": "area", "arguments": {"base": 10, "height": 6}}
    assert not score(_pred(wrong), [AREA])["exact"]


def test_int_and_float_compare_equal_but_bool_is_not_a_number():
    assert score(_pred({"name": "area", "arguments": {"base": 10.0, "height": 5}}), [AREA])["exact"]
    gold: PossibleCall = {"f": {"flag": [True]}}
    assert not score(_pred({"name": "f", "arguments": {"flag": 1}}), [gold])["exact"]


def test_parallel_calls_match_in_any_order():
    gold: list[PossibleCall] = [{"w": {"city": ["Paris"]}}, {"w": {"city": ["Rome"]}}]
    pred = _pred(
        {"name": "w", "arguments": {"city": "Rome"}}, {"name": "w", "arguments": {"city": "Paris"}}
    )
    assert score(pred, gold)["exact"]
    twice_paris = _pred(*[{"name": "w", "arguments": {"city": "Paris"}}] * 2)
    assert not score(twice_paris, gold)["exact"]


def test_pairing_backtracks_when_a_call_fits_two_golds():
    gold: list[PossibleCall] = [{"f": {"x": [1, 2]}}, {"f": {"x": [1]}}]
    calls: Any = [{"name": "f", "arguments": {"x": 1}}, {"name": "f", "arguments": {"x": 2}}]
    assert calls_match(calls, gold, normalize=False)


def test_string_normalisation_only_in_bfcl_exact():
    gold: PossibleCall = {"f": {"city": ["New York"]}}
    s = score(_pred({"name": "f", "arguments": {"city": "new-york"}}), [gold])
    assert (s["exact"], s["bfcl_exact"]) == (False, True)
    assert normalize_string("It's A_b.c") == 'it"sabc'


def test_list_values_compare_element_by_element():
    gold: PossibleCall = {"f": {"xs": [[3, 5]]}}
    assert score(_pred({"name": "f", "arguments": {"xs": [3, 5]}}), [gold])["exact"]
    assert not score(_pred({"name": "f", "arguments": {"xs": [5, 3]}}), [gold])["exact"]


def test_dict_value_holds_acceptable_values_per_key():
    gold: PossibleCall = {"f": {"where": [{"dept": ["Science"], "school": ["Bluebird HS", ""]}]}}
    assert score(_pred({"name": "f", "arguments": {"where": {"dept": "Science"}}}), [gold])["exact"]
    other = {"name": "f", "arguments": {"where": {"dept": "Science", "school": "Elm"}}}
    assert not score(_pred(other), [gold])["exact"]


def test_parse_failure_scores_zero():
    assert score("Sure, here you go", [AREA]) == {
        "parse_ok": False,
        "exact": False,
        "bfcl_exact": False,
        "names": False,
    }


def test_parse_row_takes_the_single_user_turn():
    row = {
        "id": "simple_0",
        "question": [[{"role": "user", "content": "Area?"}]],
        "function": [{"name": "area", "description": "", "parameters": {"properties": {}}}],
    }
    ex = parse_row(row, [AREA], "simple")
    assert (ex["query"], ex["tools"][0]["name"], ex["possible"]) == ("Area?", "area", [AREA])
    row["question"] = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    with pytest.raises(ValueError, match="single user message"):
        parse_row(row, [AREA], "simple")


def _ex(i: int, cat: Any, name: str = "area") -> BfclRow:
    return {"id": f"{cat}_{i}", "category": cat, "query": "", "tools": [], "possible": [{name: {}}]}


def test_summarize_merges_small_categories_and_flags_them():
    examples = [_ex(i, "simple") for i in range(4)] + [_ex(i, "parallel") for i in range(2)]
    hit = {"parse_ok": True, "exact": True, "bfcl_exact": True, "names": True}
    miss = {"parse_ok": True, "exact": False, "bfcl_exact": False, "names": True}
    scores: Any = [hit, hit, hit, miss, miss, hit]
    out = summarize(examples, scores)
    assert out["overall"]["n"] == 6
    assert out["overall"]["exact"] == round(4 / 6, 4)
    assert out["slices"]["simple"]["exact"] == 0.75
    assert out["slices"]["multiple_parallel"]["n"] == 2
    assert out["categories"]["parallel"]["below_min_rows"] is True
    assert "multiple" not in out["categories"]


def test_seen_in_train_flags_rows_calling_a_train_name():
    examples = [_ex(0, "simple", "area"), _ex(1, "simple", "get_weather")]
    assert seen_in_train(examples, {"get_weather", "other"}) == ["simple_1"]
