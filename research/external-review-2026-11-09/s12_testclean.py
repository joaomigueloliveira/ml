"""s12: test-clean rows the audit calls sound but 'default made explicit' (strict marks equivalent
calls wrong there): per-model exact and lenient; B vs 2B by whether the row's tool set was seen in
train (README's memorisation claim); quirk matching on test vs test-clean."""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO, examples, mcnemar, pct, preds, splits

EX = examples()
out = []
P = out.append
audit = {int(r["id"]): r for r in csv.DictReader(open(REPO / "data/audit/test_labels.csv"))}
clean = set(splits()["test-clean"])
dme = [i for i in clean if i in audit and audit[i]["category"] == "default made explicit"]
P(f"## test-clean rows audited sound with category 'default made explicit': {len(dme)}")
for m in ["0.8B", "2B", "4B", "A", "B", "Bf", "D", "D500", "BD"]:
    p = preds(m, "test-clean")
    P(
        f"{m}: exact {sum(p[i]['exact'] for i in dme)}/{len(dme)}, lenient {sum(p[i]['lenient_exact'] for i in dme)}/{len(dme)}"
    )
for a, b in [("D", "B"), ("BD", "B"), ("D500", "B")]:
    pa, pb = preds(a, "test-clean"), preds(b, "test-clean")
    rest = sorted(clean - set(dme))
    oa, ob, pv = mcnemar([pa[i]["exact"] for i in rest], [pb[i]["exact"] for i in rest])
    P(f"{a} vs {b}, test-clean without those rows (n={len(rest)}): {oa}/{ob} p={pv:.3g}")

# Tool sets seen in train
train_sets = {frozenset(t["name"] for t in EX[i]["tools"]) for i in splits()["train"]}
train_q = {" ".join(EX[i]["query"].lower().split()) for i in splits()["train"]}
P("\n## test-clean by whether the exact tool set appears in train")
for seen in (True, False):
    ids = [i for i in clean if (frozenset(t["name"] for t in EX[i]["tools"]) in train_sets) == seen]
    b, t, d = preds("B", "test-clean"), preds("2B", "test-clean"), preds("D", "test-clean")
    ob, ot, pv = mcnemar([b[i]["exact"] for i in ids], [t[i]["exact"] for i in ids])
    P(
        f"tool set {'seen' if seen else 'unseen'} n={len(ids)}: B {pct(sum(b[i]['exact'] for i in ids) / len(ids))}, D {pct(sum(d[i]['exact'] for i in ids) / len(ids))}, 2B {pct(sum(t[i]['exact'] for i in ids) / len(ids))}; B vs 2B {ob}/{ot} p={pv:.3g}"
    )
P(
    f"test-clean queries whose normalised text appears in train: {sum(' '.join(EX[i]['query'].lower().split()) in train_q for i in clean)}"
)
open(os.path.join(os.path.dirname(__file__), "out/s12_testclean.md"), "w").write("\n".join(out))
print("\n".join(out))
