from typing import Any, cast

from fc_distill.data import Call, Example


def make_example(
    id: int,
    query: str = "q",
    answers: list[dict[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
) -> Example:
    """A one-tool example; `params` is the tool's parameter spec, `answers` the gold calls."""
    params = {"x": {"type": "int"}} if params is None else params
    answers = [{"name": "f", "arguments": {"x": 1}}] if answers is None else answers
    return {
        "id": id,
        "query": query,
        "tools": [{"name": "f", "description": "", "parameters": params}],
        "answers": cast(list[Call], answers),
    }
