"""Error taxonomy for xLAM and BFCL predictions (shared by the analysis scripts)."""

import re
from collections import Counter

from fc_distill.scoring import canon, is_optional, parse_calls, without_defaults
from fc_distill.slices import value_in_query

_NORM = re.compile(r"[\s,./\-_*^'\"]")


def norm_str(v):
    return _NORM.sub("", str(v)).lower()


def value_kind(pv, gv, query):
    """How a wrong value differs from gold."""
    if (
        isinstance(pv, str) != isinstance(gv, str)
        and str(pv).strip().lower() == str(gv).strip().lower()
    ):
        return "type (str vs number/bool)"
    if isinstance(pv, (int, float)) and isinstance(gv, (int, float)) and not isinstance(pv, bool):
        return "number" + (
            " (gold in query)" if value_in_query(gv, query) else " (gold not in query)"
        )
    if norm_str(pv) == norm_str(gv):
        return "string format/case only"
    return "content" + (" (gold in query)" if value_in_query(gv, query) else " (gold not in query)")


def _align(pred, gold):
    """Pair predicted and gold calls with the same name, greedily by argument overlap."""
    pairs, used = [], set()
    for g in gold:
        best, bscore = None, -1
        for j, p in enumerate(pred):
            if j in used or p["name"] != g["name"]:
                continue
            s = len(set(map(str, p["arguments"].items())) & set(map(str, g["arguments"].items())))
            if s > bscore:
                best, bscore = j, s
        if best is not None:
            used.add(best)
            pairs.append((pred[best], g))
    return pairs


def classify_xlam(prediction, ex):
    """(coarse, fine) error class for one xLAM prediction against ex's gold."""
    calls = parse_calls(prediction)
    gold = ex["answers"]
    tools = ex["tools"]
    if calls is None:
        return "unparsed", "unparsed"
    from fc_distill.scoring import call_multiset, lenient_match

    if call_multiset(calls) == call_multiset(gold):
        return "right", "right"
    if lenient_match(calls, gold, tools):
        return "default-only", "optional default added/dropped"
    pn, gn = Counter(c["name"] for c in calls), Counter(c["name"] for c in gold)
    if pn != gn:
        if sum(pn.values()) < sum(gn.values()) and not (pn - gn):
            return "function", "missing call"
        if sum(pn.values()) > sum(gn.values()) and not (gn - pn):
            return "function", "extra call"
        if any(n not in {t["name"] for t in tools} for n in pn):
            return "function", "unknown function"
        return "function", "wrong function"
    specs = {t["name"]: t.get("parameters") or {} for t in tools}
    P, G = without_defaults(calls, tools), without_defaults(gold, tools)
    kinds = []
    for p, g in _align(P, G):
        sp = specs.get(g["name"], {})
        pk, gk = set(p["arguments"]), set(g["arguments"])
        for k in pk - gk:
            kinds.append(
                (
                    "arg key",
                    "extra optional arg"
                    if is_optional(sp.get(k, {}))
                    else ("extra unknown arg" if k not in sp else "extra required arg"),
                )
            )
        for k in gk - pk:
            kinds.append(
                (
                    "arg key",
                    "missing optional arg"
                    if is_optional(sp.get(k, {}))
                    else "missing required arg",
                )
            )
        for k in pk & gk:
            if canon(p["arguments"][k]) != canon(g["arguments"][k]):
                kinds.append(
                    ("value", value_kind(p["arguments"][k], g["arguments"][k], ex["query"]))
                )
    if not kinds:
        return "value", "other"
    # Report keys before values; the first fine class of the coarse class chosen.
    for coarse in ("arg key", "value"):
        for c, f in kinds:
            if c == coarse:
                return c, f
    return kinds[0]


def classify_bfcl(prediction, row):
    calls = parse_calls(prediction)
    if calls is None:
        return "unparsed", "unparsed"
    gold = row["possible"]
    from fc_distill.bfcl import _value_ok, calls_match

    if calls_match(calls, gold, normalize=False):
        return "right", "right"
    pn = Counter(c["name"] for c in calls)
    gn = Counter(next(iter(g)) for g in gold)
    if pn != gn:
        if sum(pn.values()) < sum(gn.values()) and not (pn - gn):
            return "function", "missing call"
        if sum(pn.values()) > sum(gn.values()) and not (gn - pn):
            return "function", "extra call"
        return "function", "wrong function"
    if calls_match(calls, gold, normalize=True):
        return "value", "string format/case only"
    # align by name, then find the first problem
    kinds = []
    used = set()
    for c in calls:
        cand = [j for j, g in enumerate(gold) if j not in used and next(iter(g)) == c["name"]]

        # pick the gold call with most matching args
        def nmatch(j):
            poss = gold[j][c["name"]]
            return sum(
                1
                for k, v in c["arguments"].items()
                if k in poss and any(_value_ok(v, a, True) for a in poss[k])
            )

        j = max(cand, key=nmatch)
        used.add(j)
        poss = gold[j][c["name"]]
        for k in set(c["arguments"]) - set(poss):
            kinds.append(("arg key", "extra unknown arg"))
        for k, acc in poss.items():
            if k not in c["arguments"]:
                if "" not in acc:
                    kinds.append(("arg key", "missing required arg"))
            elif not any(_value_ok(c["arguments"][k], a, True) for a in acc):
                v = c["arguments"][k]
                a0 = next((a for a in acc if a != ""), None)
                if "" in acc:
                    kinds.append(("value", "optional arg given a non-default value"))
                elif (
                    isinstance(v, str) != isinstance(a0, str)
                    and a0 is not None
                    and not isinstance(a0, (list, dict))
                ):
                    kinds.append(("value", "type (str vs number/bool)"))
                elif isinstance(a0, (list, dict)):
                    kinds.append(("value", "list/dict value"))
                elif isinstance(a0, (int, float)):
                    kinds.append(
                        (
                            "value",
                            "number"
                            + (
                                " (gold in query)"
                                if value_in_query(a0, row["query"])
                                else " (gold not in query)"
                            ),
                        )
                    )
                else:
                    kinds.append(
                        (
                            "value",
                            "content"
                            + (
                                " (gold in query)"
                                if value_in_query(a0, row["query"])
                                else " (gold not in query)"
                            ),
                        )
                    )
    for coarse in ("arg key", "value"):
        for c, f in kinds:
            if c == coarse:
                return c, f
    return "value", "other"
