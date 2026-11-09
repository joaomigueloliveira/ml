"""s4: are the teacher's unparsed outputs truncations or bracket slips? Rescore every model after a
deterministic closing-bracket repair (strip trailing ]} and re-close from the bracket stack)."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import bfcl_rows, examples, mcnemar, pct, preds

from fc_distill import bfcl
from fc_distill.scoring import parse_calls, score

EX = examples()
BF = bfcl_rows()


def repair(text):
    s = text.strip()
    if parse_calls(s) is not None:
        return s
    body = s.rstrip("]} \n")
    stack, in_str, esc = [], False, False
    for ch in body:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if stack:
                stack.pop()
    if in_str:
        return s
    fixed = body + "".join("]" if c == "[" else "}" for c in reversed(stack))
    return fixed if parse_calls(fixed) is not None else s


out = []
P = out.append
MODELS = ["0.8B", "2B", "4B", "A", "B", "Bf", "D", "BD"]
for split in ["validation", "test-clean", "bfcl"]:
    P(
        f"\n## {split}: exact before / after bracket repair (rows repaired to parse; of those now exact)"
    )
    rep = {}
    for m in MODELS:
        try:
            p = preds(m, split)
        except FileNotFoundError:
            continue
        new = {}
        nrep = nright = 0
        for i, r in p.items():
            if r["parse_ok"]:
                new[i] = r["exact"]
                continue
            f = repair(r["prediction"])
            if split == "bfcl":
                s = bfcl.score(f, BF[i]["possible"])
            else:
                s = score(f, EX[i]["answers"], EX[i]["tools"])
            new[i] = s["exact"]
            nrep += s["parse_ok"]
            nright += s["exact"]
        rep[m] = new
        before = sum(r["exact"] for r in p.values()) / len(p)
        P(
            f"{m} | {pct(before)} -> {pct(sum(new.values()) / len(new))} | unparsed {sum(not r['parse_ok'] for r in p.values())}, repaired {nrep}, now right {nright}"
        )
    for a, b in [("B", "2B"), ("D", "2B"), ("Bf", "2B")]:
        ids = sorted(rep[a])
        oa, ob, pv = mcnemar([rep[a][i] for i in ids], [rep[b][i] for i in ids])
        P(f"  after repair {a} vs {b}: only {a} {oa} / only {b} {ob}, p={pv:.3g}")
    if split == "test-clean":
        raw, t = 0.548, sum(rep["2B"].values()) / len(rep["2B"])
        for m in ["B", "D", "Bf"]:
            mm = sum(rep[m].values()) / len(rep[m])
            P(
                f"  gap recovered vs repaired raw models: {m} {(mm - raw) / (t - raw) * 100:.0f}% (repaired 2B {pct(t)})"
            )

# The teacher on train: how many of the 2,137 unparsed outputs repair, and of those match gold
t = preds("2B", "train")
n = ok = right = 0
for i, r in t.items():
    if r["parse_ok"]:
        continue
    n += 1
    f = repair(r["prediction"])
    s = score(f, EX[i]["answers"], EX[i]["tools"])
    ok += s["parse_ok"]
    right += s["lenient_exact"]
P(f"\n## train: 2B unparsed {n}, repaired to parse {ok}, of which lenient-match gold {right}")
open(os.path.join(os.path.dirname(__file__), "out/s4_repair.md"), "w").write("\n".join(out))
print("\n".join(out))
