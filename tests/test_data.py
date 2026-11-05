import json

from fc_distill.data import parse_row, to_messages

ROW = {
    "id": 7,
    "query": "Weather in Lisbon for 3 days?",
    "tools": json.dumps([{"name": "get_weather", "parameters": {"city": {"type": "str"}}}]),
    "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Lisbon", "days": 3}}]),
}


def test_parse_row_decodes_json_fields():
    ex = parse_row(ROW)
    assert ex["tools"][0]["name"] == "get_weather"
    assert ex["answers"] == [{"name": "get_weather", "arguments": {"city": "Lisbon", "days": 3}}]


def test_messages_put_tools_in_system_and_query_in_user():
    system, user = to_messages(parse_row(ROW))
    assert system["role"] == "system"
    assert '"get_weather"' in system["content"]
    assert user == {"role": "user", "content": ROW["query"]}
