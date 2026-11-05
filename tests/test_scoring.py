import json

from xlam_distill.scoring import parse_calls, score

WEATHER = {"name": "get_weather", "arguments": {"city": "Lisbon", "days": 3}}
TIME = {"name": "get_time", "arguments": {"tz": "Europe/Lisbon"}}
GOLD = [WEATHER, TIME]


def test_exact_match_ignores_call_order_and_int_float():
    weather = {"name": "get_weather", "arguments": {"days": 3.0, "city": "Lisbon"}}
    pred = json.dumps([TIME, weather])
    assert score(pred, GOLD) == {"parse_ok": True, "exact": True, "names": True, "arg_f1": 1.0}


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
    assert s == {"parse_ok": False, "exact": False, "names": False, "arg_f1": 0.0}


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
