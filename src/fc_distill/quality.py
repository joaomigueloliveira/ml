"""Detectors for label and schema problems in xLAM rows.

Each detector only reports. `fc_distill.splits` decides which findings remove train rows.
"""

import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fc_distill.data import Example
from fc_distill.scoring import call_multiset

# Python types a JSON value must have for each declared base type. float accepts ints.
_EXPECTED: dict[str, tuple[type, ...]] = {
    "str": (str,),
    "int": (int,),
    "float": (int, float),
    "bool": (bool,),
    "list": (list,),
    "tuple": (list,),
    "set": (list,),
    "dict": (dict,),
}
_NON_TEXT = {"int", "float", "list", "tuple", "set", "dict"}


def base_type(declared: str) -> str:
    """'List[int]' -> 'list', 'str, optional' -> 'str', 'Callable[[float], float]' -> 'callable'."""
    match = re.match(r"\s*([A-Za-z]+)", declared)
    return match.group(1).lower() if match else ""


def is_optional(spec: dict[str, Any]) -> bool:
    return "optional" in str(spec.get("type", "")).lower() or "default" in spec


@dataclass(frozen=True)
class ArgumentIssue:
    row_id: int
    function: str
    argument: str
    declared: str
    value: Any


def _arguments(example: Example) -> list[tuple[str, str, dict[str, Any], Any]]:
    """(function, argument, parameter spec, value) for every argument in the gold answer."""
    tools = {t["name"]: t for t in example["tools"]}
    return [
        (call["name"], arg, (tools[call["name"]].get("parameters") or {}).get(arg, {}), value)
        for call in example["answers"]
        for arg, value in call["arguments"].items()
    ]


def _matches(value: Any, base: str) -> bool:
    if base not in _EXPECTED:
        return True  # callable, any, unknown: nothing to check
    if isinstance(value, bool) and base != "bool":
        return False
    return isinstance(value, _EXPECTED[base])


def type_mismatches(examples: Sequence[Example]) -> list[ArgumentIssue]:
    """Arguments whose value doesn't have the declared type (e.g. 0.3 for an `int` parameter)."""
    return [
        ArgumentIssue(ex["id"], fn, arg, str(spec.get("type", "")), value)
        for ex in examples
        for fn, arg, spec, value in _arguments(ex)
        if not _matches(value, base_type(str(spec.get("type", ""))))
    ]


_CODE = re.compile(r"[()\[\]*]|\b[A-Za-z_]\w*\.[A-Za-z_]")


def is_code_string(value: Any, base: str) -> bool:
    """A string holding a Python expression where a non-text value is declared: 'math.pi / 2',
    '[0.015] * 12'. Plain words ('movie') are type mismatches, not code."""
    if base not in _NON_TEXT or not isinstance(value, str):
        return False
    try:
        json.loads(value)
    except json.JSONDecodeError:
        return bool(_CODE.search(value))
    return False


def code_arguments(examples: Sequence[Example]) -> list[ArgumentIssue]:
    """Arguments written as Python expressions instead of values; a model must copy the text."""
    return [
        ArgumentIssue(ex["id"], fn, arg, str(spec.get("type", "")), value)
        for ex in examples
        for fn, arg, spec, value in _arguments(ex)
        if is_code_string(value, base_type(str(spec.get("type", ""))))
    ]


def missing_required(examples: Sequence[Example]) -> list[tuple[int, str, str]]:
    """(row id, function, parameter) for required parameters the gold call leaves out."""
    found = []
    for ex in examples:
        tools = {t["name"]: t for t in ex["tools"]}
        for call in ex["answers"]:
            for param, spec in (tools[call["name"]].get("parameters") or {}).items():
                if not is_optional(spec) and param not in call["arguments"]:
                    found.append((ex["id"], call["name"], param))
    return found


def normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def exact_duplicates(examples: Sequence[Example]) -> list[list[int]]:
    """Groups of row ids with identical query, tools and answers (2+ rows per group)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for ex in examples:
        key = json.dumps([ex["query"], ex["tools"], ex["answers"]], sort_keys=True)
        groups[key].append(ex["id"])
    return [ids for ids in groups.values() if len(ids) > 1]


def conflicting_labels(examples: Sequence[Example]) -> list[list[int]]:
    """Groups of row ids with the same query and the same tools but different answers."""
    groups: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        groups[json.dumps([normalize_query(ex["query"]), ex["tools"]], sort_keys=True)].append(ex)
    return [
        [ex["id"] for ex in group]
        for group in groups.values()
        if len({call_multiset(ex["answers"]) for ex in group}) > 1
    ]
