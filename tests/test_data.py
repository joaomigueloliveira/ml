import json

from xlam_distill.data import make_splits, parse_row, to_messages

ROW = {
    "id": 7,
    "query": "Weather in Lisbon for 3 days?",
    "tools": json.dumps([{"name": "get_weather", "parameters": {"city": {"type": "str"}}}]),
    "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Lisbon", "days": 3}}]),
}


def test_splits_are_deterministic_disjoint_and_complete():
    a, b = make_splits(100, eval_size=10, seed=0), make_splits(100, eval_size=10, seed=0)
    assert a == b
    assert len(a["eval"]) == 10
    assert set(a["eval"]).isdisjoint(a["train"])
    assert sorted(a["eval"] + a["train"]) == list(range(100))


def test_different_seed_gives_different_eval_set():
    assert make_splits(100, 10, seed=0)["eval"] != make_splits(100, 10, seed=1)["eval"]


def test_parse_row_decodes_json_fields():
    ex = parse_row(ROW)
    assert ex["tools"][0]["name"] == "get_weather"
    assert ex["answers"] == [{"name": "get_weather", "arguments": {"city": "Lisbon", "days": 3}}]


def test_messages_put_tools_in_system_and_query_in_user():
    system, user = to_messages(parse_row(ROW))
    assert system["role"] == "system"
    assert '"get_weather"' in system["content"]
    assert user == {"role": "user", "content": ROW["query"]}
