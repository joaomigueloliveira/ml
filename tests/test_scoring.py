import json
import random
from typing import Any

from fc_distill.scoring import (
    canon,
    is_default,
    lenient_match,
    parse_calls,
    score,
    without_defaults,
)

WEATHER = {"name": "get_weather", "arguments": {"city": "Lisbon", "days": 3}}
TIME = {"name": "get_time", "arguments": {"tz": "Europe/Lisbon"}}
GOLD = [WEATHER, TIME]


def test_exact_match_ignores_call_order_and_int_float():
    weather = {"name": "get_weather", "arguments": {"days": 3.0, "city": "Lisbon"}}
    pred = json.dumps([TIME, weather])
    assert score(pred, GOLD) == {
        "parse_ok": True,
        "exact": True,
        "lenient_exact": True,
        "names": True,
        "arg_f1": 1.0,
    }


def test_strips_think_block_and_code_fence():
    pred = f"<think>hmm</think>\n```json\n{json.dumps([TIME])}\n```"
    assert parse_calls(pred) == [TIME]


def test_wrong_value_gives_partial_credit():
    weather = {"name": "get_weather", "arguments": {"city": "Porto", "days": 3}}
    s = score(json.dumps([weather, TIME]), GOLD)
    assert (s["exact"], s["names"]) == (False, True)
    assert abs(s["arg_f1"] - 2 / 3) < 1e-9


def test_extra_text_is_a_parse_failure():
    s = score(f"Sure! {json.dumps([TIME])}", GOLD)
    assert s == {
        "parse_ok": False,
        "exact": False,
        "lenient_exact": False,
        "names": False,
        "arg_f1": 0.0,
    }


def test_bool_is_not_number():
    gold = [{"name": "f", "arguments": {"flag": True}}]
    assert not score('[{"name": "f", "arguments": {"flag": 1}}]', gold)["exact"]


def test_single_object_is_accepted_as_one_call():
    gold = [{"name": "f", "arguments": {}}]
    assert score('{"name": "f", "arguments": {}}', gold)["exact"]


def test_missing_arguments_means_no_arguments():
    gold = [{"name": "f", "arguments": {}}]
    assert score('[{"name": "f"}]', gold)["exact"]


def test_wrong_function_scores_zero_args():
    s = score(json.dumps([{"name": "get_news", "arguments": {}}]), [TIME])
    assert (s["parse_ok"], s["exact"], s["names"], s["arg_f1"]) == (True, False, False, 0.0)


# Lenient exact (#15): optional arguments at their declared default, or null, are ignored.
TOOLS: list[dict[str, Any]] = [
    {
        "name": "search",
        "parameters": {
            "q": {"type": "str", "default": "hello"},
            "limit": {"type": "int, optional", "default": "10"},
            "lang": {"type": "str, optional", "default": "en"},
            "cursor": {"type": "str, optional"},
        },
    }
]


def _search(**arguments: Any) -> str:
    return json.dumps([{"name": "search", "arguments": arguments}])


def _lenient(pred: str, gold_args: dict[str, Any]) -> tuple[bool, bool]:
    s = score(pred, [{"name": "search", "arguments": gold_args}], TOOLS)
    return s["exact"], s["lenient_exact"]


def test_lenient_accepts_an_omitted_default_on_either_side():
    assert _lenient(_search(q="cats"), {"q": "cats", "lang": "en"}) == (False, True)
    assert _lenient(_search(q="cats", lang="en"), {"q": "cats"}) == (False, True)


def test_lenient_matches_a_default_written_as_a_string():
    # xLAM writes the default as "10" for an int parameter
    assert _lenient(_search(q="cats"), {"q": "cats", "limit": 10}) == (False, True)


def test_lenient_accepts_an_omitted_null():
    assert _lenient(_search(q="cats"), {"q": "cats", "cursor": None}) == (False, True)


def test_lenient_rejects_a_different_non_default_value():
    assert _lenient(_search(q="cats"), {"q": "cats", "lang": "fr"}) == (False, False)


def test_lenient_rejects_an_omitted_required_argument_even_at_its_default():
    # "q" has a default in the spec, but its type string doesn't say optional
    assert _lenient(_search(lang="en"), {"q": "hello", "lang": "en"}) == (False, False)


def test_lenient_without_tools_equals_strict():
    gold = [{"name": "search", "arguments": {"q": "cats", "lang": "en"}}]
    s = score(_search(q="cats"), gold)
    assert (s["exact"], s["lenient_exact"]) == (False, False)


def test_lenient_is_never_below_strict():
    rng = random.Random(0)
    values: dict[str, list[Any]] = {
        "q": ["cats", "hello"],
        "limit": [10, 5, "10", None],
        "lang": ["en", "fr", None],
        "cursor": [None, "abc"],
    }

    def draw() -> dict[str, Any]:
        return {k: rng.choice(v) for k, v in values.items() if rng.random() < 0.6}

    # lenient_match itself (not score(), which ORs it with exact) must accept every exact pair
    for _ in range(500):
        args = draw()
        call: Any = [{"name": "search", "arguments": args}]
        same: Any = [{"name": "search", "arguments": json.loads(json.dumps(args))}]
        assert lenient_match(call, same, TOOLS)


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


def test_canon_keeps_large_integers_exact():
    big = 2**53
    assert canon(big) != canon(big + 1)
    assert canon(5) == canon(5.0)
    assert canon(True) != canon(1)
    assert canon(0.5) == canon(0.5)


def test_is_default_bool_and_string_forms():
    one = {"type": "int, optional", "default": 1}
    assert is_default(one, 1)
    assert not is_default(one, True)
    flag = {"type": "bool, optional", "default": "True"}
    assert is_default(flag, True)
    assert not is_default(flag, False)
    lower = {"type": "bool, optional", "default": "true"}
    assert is_default(lower, True)
    assert not is_default({"type": "bool", "default": "true"}, True)  # required
