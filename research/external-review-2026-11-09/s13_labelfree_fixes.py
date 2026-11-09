"""s13: two zero-GPU, label-free probes.
(a) Canonicalise each model's output (drop optional arguments passed at their declared default or
    null, using only the tool spec) and rescore strict exact: how much of the B/D/B->D ordering is
    the default convention?
(b) Label-free filters on the teacher's train outputs: rows kept and their agreement with gold
    (lenient), to see whether any label-free filter gets near filtered B's 94% precision."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import examples, mcnemar, pct, preds, splits

from fc_distill.scoring import is_optional, parse_calls, score, without_defaults
from fc_distill.slices import value_in_query

EX = examples()
out = []
P = out.append


def canon_pred(text, tools):
    c = parse_calls(text)
    if c is None:
        return text
    return json.dumps(without_defaults(c, tools))


P("## (a) strict exact after dropping the model's own default-valued optional arguments")
CAN = {}
for split in ["validation", "test-clean"]:
    P(f"\n### {split}")
    for m in ["0.8B", "2B", "A", "B", "Bf", "D", "D500", "BD"]:
        p = preds(m, split)
        new = {
            i: score(canon_pred(r["prediction"], EX[i]["tools"]), EX[i]["answers"], EX[i]["tools"])[
                "exact"
            ]
            for i, r in p.items()
        }
        CAN[m, split] = new
        P(
            f"{m}: {pct(sum(r['exact'] for r in p.values()) / len(p))} -> {pct(sum(new.values()) / len(new))}"
        )
    for a, b in [("D", "B"), ("BD", "B"), ("D500", "B"), ("B", "2B"), ("Bf", "B")]:
        ids = sorted(CAN[a, split])
        oa, ob, pv = mcnemar([CAN[a, split][i] for i in ids], [CAN[b, split][i] for i in ids])
        P(f"  canonicalised {a} vs {b}: {oa}/{ob} p={pv:.3g}")

# (b) label-free filters on train
T = preds("2B", "train")
train = splits()["train"]


def validity(calls, ex):
    specs = {t["name"]: t.get("parameters") or {} for t in ex["tools"]}
    for c in calls:
        if c["name"] not in specs:
            return False
        for k, s in specs[c["name"]].items():
            if not is_optional(s) and k not in c["arguments"]:
                return False
    return True


def grounded(calls, ex):
    """Every string/number value of the teacher's output is written in the query (slices rule)."""
    return all(value_in_query(v, ex["query"]) for c in calls for v in c["arguments"].values())


rows = []
for i in train:
    r = T[i]
    c = parse_calls(r["prediction"])
    if c is None:
        continue
    ex = EX[i]
    rows.append((i, r["lenient_exact"], validity(c, ex), grounded(c, ex), len(c)))
P(f"\n## (b) label-free filters on the teacher's {len(rows)} parsed train outputs (B's data)")
P("filter | rows kept | share lenient-matching gold")
for name, f in [
    ("parse only (B)", lambda r: True),
    ("validity", lambda r: r[2]),
    ("grounded values", lambda r: r[3]),
    ("validity and grounded", lambda r: r[2] and r[3]),
]:
    kept = [r for r in rows if f(r)]
    P(f"{name} | {len(kept)} | {pct(sum(r[1] for r in kept) / len(kept))}")
P(
    f"lenient match (filtered B, uses labels) | {sum(r[1] for r in rows)} | 100 by construction (94.0 strict)"
)
open(os.path.join(os.path.dirname(__file__), "out/s13_labelfree_fixes.md"), "w").write(
    "\n".join(out)
)
print("\n".join(out))
