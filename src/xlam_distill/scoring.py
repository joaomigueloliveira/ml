"""Deterministic scoring of function-call predictions against xLAM gold answers.

A prediction is the raw model text. It must contain a JSON list of
{"name": str, "arguments": dict} objects. Call order is ignored; everything else is strict.
"""

import json
import re
from collections import Counter
from collections.abc import Hashable, Sequence
from statistics import mean
from typing import Any, TypedDict, cast

from xlam_distill.data import Call

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class Score(TypedDict):
    parse_ok: bool
    exact: bool
    names: bool
    arg_f1: float


def parse_calls(text: str) -> list[Call] | None:
    """Extract the list of calls from model output, or None if it isn't valid."""
    text = _THINK.sub("", text).strip()
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        calls = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(calls, dict):
        calls = [calls]
    if not isinstance(calls, list):
        return None
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            return None
        if not isinstance(call.setdefault("arguments", {}), dict):
            return None
    return cast(list[Call], calls)


def _canon(value: Any) -> Hashable:
    """Canonical, hashable form of a JSON value. 5 and 5.0 compare equal, but True != 1
    (Python says True == 1.0, so bools and numbers are tagged to keep them apart)."""
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int | float):
        return ("num", float(value))
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        return tuple(_canon(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _canon(v)) for k, v in value.items()))
    return repr(value)


def _call_key(call: Call) -> Hashable:
    return (call["name"], _canon(call["arguments"]))


def _arg_triples(calls: Sequence[Call]) -> Counter[Hashable]:
    return Counter((c["name"], k, _canon(v)) for c in calls for k, v in c["arguments"].items())


def score(prediction: str, gold: Sequence[Call]) -> Score:
    """Score one example: valid format, exact match, right functions, argument-level F1."""
    calls = parse_calls(prediction)
    if calls is None:
        return {"parse_ok": False, "exact": False, "names": False, "arg_f1": 0.0}

    exact = Counter(map(_call_key, calls)) == Counter(map(_call_key, gold))
    names = Counter(c["name"] for c in calls) == Counter(c["name"] for c in gold)

    pred_args, gold_args = _arg_triples(calls), _arg_triples(gold)
    if not pred_args and not gold_args:
        arg_f1 = 1.0 if names else 0.0
    else:
        hit = (pred_args & gold_args).total()
        precision = hit / max(pred_args.total(), 1)
        recall = hit / max(gold_args.total(), 1)
        arg_f1 = 2 * precision * recall / (precision + recall) if hit else 0.0

    return {"parse_ok": True, "exact": exact, "names": names, "arg_f1": arg_f1}


def summarize(scores: Sequence[Score]) -> dict[str, float]:
    """Mean of every metric over a set of scored examples."""
    return {
        "n": len(scores),
        "parse_ok": mean(s["parse_ok"] for s in scores),
        "exact": mean(s["exact"] for s in scores),
        "names": mean(s["names"] for s in scores),
        "arg_f1": mean(s["arg_f1"] for s in scores),
    }
