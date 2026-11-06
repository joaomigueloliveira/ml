"""BFCL non-live single-turn subset (issue #7): tools the student never trained on.

The Berkeley Function-Calling Leaderboard's simple, multiple, parallel and parallel-multiple
categories (1,000 rows) at a pinned revision. Tool specs are converted to xLAM's shape, so the only
new thing for a model trained on xLAM is the tools themselves. Each gold argument is a list of
acceptable values; `""` in the list means the argument may be left out. Notes: docs/bfcl.md.
"""

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict, get_args

from fc_distill.data import Call, Message, prompt_messages
from fc_distill.eda import ci_half_width
from fc_distill.scoring import _canon, parse_calls

REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
Category = Literal["simple", "multiple", "parallel", "parallel_multiple"]
CATEGORIES: tuple[Category, ...] = get_args(Category)

_TYPES = {
    "string": "str",
    "integer": "int",
    "float": "float",
    "number": "float",
    "boolean": "bool",
    "dict": "Dict",
    "tuple": "Tuple",
    "any": "Any",
}

# A gold call: {function name: {argument: [acceptable values]}}.
PossibleCall = dict[str, dict[str, list[Any]]]


class BfclExample(TypedDict):
    id: str
    category: Category
    query: str
    tools: list[dict[str, Any]]
    possible: list[PossibleCall]


class BfclScore(TypedDict):
    parse_ok: bool
    exact: bool
    bfcl_exact: bool
    names: bool


def load(categories: Iterable[Category] = CATEGORIES) -> list[BfclExample]:
    """Every row of the given categories at the pinned revision, tools converted to xLAM style."""
    from huggingface_hub import hf_hub_download

    def read(name: str) -> list[dict[str, Any]]:
        path = hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION)
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    out: list[BfclExample] = []
    for cat in categories:
        answers = {r["id"]: r["ground_truth"] for r in read(f"possible_answer/BFCL_v3_{cat}.json")}
        out += [parse_row(r, answers[r["id"]], cat) for r in read(f"BFCL_v3_{cat}.json")]
    return out


def parse_row(row: Mapping[str, Any], possible: list[PossibleCall], cat: Category) -> BfclExample:
    turns = row["question"]
    if len(turns) != 1 or len(turns[0]) != 1 or turns[0][0]["role"] != "user":
        raise ValueError(f"{row['id']}: expected a single user message")
    return {
        "id": row["id"],
        "category": cat,
        "query": turns[0][0]["content"],
        "tools": [to_xlam_tool(f) for f in row["function"]],
        "possible": possible,
    }


def to_messages(ex: BfclExample) -> list[Message]:
    return prompt_messages(ex["query"], ex["tools"])


def to_xlam_tool(fn: Mapping[str, Any]) -> dict[str, Any]:
    """A BFCL (JSON-schema style) function spec in xLAM's shape: {name, description, parameters:
    {arg: {description, type, default?}}}, with Python-style type strings and ", optional" on every
    parameter that isn't required."""
    params = fn["parameters"]
    return {
        "name": fn["name"],
        "description": fn["description"],
        "parameters": _convert_properties(params["properties"], params.get("required", [])),
    }


def _convert_properties(props: Mapping[str, Any], required: Sequence[str]) -> dict[str, Any]:
    out = {}
    for name, p in props.items():
        optional = name not in required or p.get("optional") is True
        spec: dict[str, Any] = {
            "description": _description(p),
            "type": type_string(p) + (", optional" if optional else ""),
        }
        if "default" in p:
            spec["default"] = p["default"]
        if "properties" in p:  # nested dict: keep its fields, converted the same way
            spec["properties"] = _convert_properties(p["properties"], p.get("required", []))
        out[name] = spec
    return out


def _description(p: Mapping[str, Any]) -> str:
    text = str(p.get("description", ""))
    allowed = p.get("enum") or p.get("items", {}).get("enum")
    if allowed:
        text = f"{text} One of: {', '.join(map(str, allowed))}.".strip()
    return text


def type_string(p: Mapping[str, Any]) -> str:
    """Python-style type string as xLAM writes them: integer → int, array of float → List[float]."""
    if p["type"] == "array":
        items = p.get("items")
        return f"List[{type_string(items)}]" if items and "type" in items else "List"
    return _TYPES.get(p["type"], str(p["type"]))


# BFCL's string standardisation (ast_checker): drop spaces and , . / - _ * ^, lowercase,
# and treat single and double quotes alike.
_STRIP = re.compile(r"[ ,./\-_*^]")


def normalize_string(s: str) -> str:
    return _STRIP.sub("", s).lower().replace("'", '"')


def _value_ok(value: Any, acceptable: Any, normalize: bool) -> bool:
    """Is `value` equal to one acceptable value? Dict acceptable values hold, per key, a list of
    acceptable values again (with "" meaning the key may be left out)."""
    if isinstance(acceptable, dict):
        return isinstance(value, dict) and _args_ok(value, acceptable, normalize)
    if isinstance(acceptable, list):
        return (
            isinstance(value, list)
            and len(value) == len(acceptable)
            and all(_value_ok(v, a, normalize) for v, a in zip(value, acceptable, strict=True))
        )
    if normalize and isinstance(value, str) and isinstance(acceptable, str):
        return normalize_string(value) == normalize_string(acceptable)
    return _canon(value) == _canon(acceptable)


def _args_ok(args: Mapping[str, Any], possible: Mapping[str, list[Any]], normalize: bool) -> bool:
    if set(args) - set(possible):
        return False
    for name, acceptable in possible.items():
        if name not in args:
            if "" not in acceptable:
                return False
        elif not any(_value_ok(args[name], a, normalize) for a in acceptable):
            return False
    return True


def _call_ok(call: Call, gold: PossibleCall, normalize: bool) -> bool:
    ((name, possible),) = gold.items()
    return call["name"] == name and _args_ok(call["arguments"], possible, normalize)


def calls_match(calls: Sequence[Call], gold: Sequence[PossibleCall], normalize: bool) -> bool:
    """Is there a one-to-one pairing of predicted and gold calls, in any order, where every pair
    matches?"""
    if len(calls) != len(gold):
        return False

    def assign(i: int, used: frozenset[int]) -> bool:
        if i == len(calls):
            return True
        return any(
            j not in used and _call_ok(calls[i], g, normalize) and assign(i + 1, used | {j})
            for j, g in enumerate(gold)
        )

    return assign(0, frozenset())


def score(prediction: str, gold: Sequence[PossibleCall]) -> BfclScore:
    """`exact`: our strict rule over BFCL's acceptable values. `bfcl_exact`: also ignores case,
    spaces and some punctuation in strings, as BFCL's checker does."""
    calls = parse_calls(prediction)
    if calls is None:
        return {"parse_ok": False, "exact": False, "bfcl_exact": False, "names": False}
    gold_names = sorted(next(iter(g)) for g in gold)
    return {
        "parse_ok": True,
        "exact": calls_match(calls, gold, normalize=False),
        "bfcl_exact": calls_match(calls, gold, normalize=True),
        "names": sorted(c["name"] for c in calls) == gold_names,
    }


Metric = Literal["parse_ok", "exact", "bfcl_exact", "names"]
METRICS: tuple[Metric, ...] = get_args(Metric)
# ADR 0001: a slice is reported on its own only with at least 300 rows. Simple has 400; the other
# three categories have 200 each and are reported merged.
SLICES: dict[str, tuple[Category, ...]] = {
    "simple": ("simple",),
    "multiple_parallel": ("multiple", "parallel", "parallel_multiple"),
}


def _block(scores: Sequence[BfclScore]) -> dict[str, Any]:
    n = len(scores)
    out: dict[str, Any] = {"n": n}
    for m in METRICS:
        p = sum(bool(s[m]) for s in scores) / n if n else 0.0
        out[m] = round(p, 4)
        # ci_half_width is in points; keep the interval in the same unit as the score.
        out[f"{m}_ci"] = round(ci_half_width(n, p) / 100, 4) if n else None
    return out


def summarize(examples: Sequence[BfclExample], scores: Sequence[BfclScore]) -> dict[str, Any]:
    """Overall and per-slice metrics with 95% interval half-widths. Each category is listed too,
    marked as below the 300-row minimum when it is."""
    pairs = list(zip(examples, scores, strict=True))
    per_cat = {
        cat: _block([s for ex, s in pairs if ex["category"] == cat])
        | {"below_min_rows": sum(ex["category"] == cat for ex in examples) < 300}
        for cat in CATEGORIES
        if any(ex["category"] == cat for ex in examples)
    }
    return {
        "overall": _block(scores),
        "slices": {
            name: _block([s for ex, s in pairs if ex["category"] in cats])
            for name, cats in SLICES.items()
        },
        "categories": per_cat,
    }


def seen_in_train(examples: Iterable[BfclExample], train_tool_names: set[str]) -> list[str]:
    """Ids of rows whose gold answer calls a function name that also appears in xLAM train."""
    return [
        ex["id"]
        for ex in examples
        if any(next(iter(g)) in train_tool_names for g in ex["possible"])
    ]
