"""s2: error taxonomy per model and split; D vs B discordant rows by error class of the loser, by
slice (call pattern, distractors, argument source) and BFCL category; output length and call
counts."""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from common import bfcl_rows, examples, mcnemar, pct, preds
from taxonomy import classify_bfcl, classify_xlam

from fc_distill.scoring import parse_calls
from fc_distill.slices import DIMENSIONS

MODELS = ["0.8B", "2B", "A", "B", "Bf", "D", "D500", "BD"]
EX = examples()
BF = bfcl_rows()
out = []
P = out.append


def classes(model, split):
    p = preds(model, split)
    if split == "bfcl":
        return {i: classify_bfcl(r["prediction"], BF[i]) for i, r in p.items()}
    return {i: classify_xlam(r["prediction"], EX[i]) for i, r in p.items()}


CL = {}
for split in ["validation", "test-clean", "bfcl"]:
    for m in MODELS:
        CL[m, split] = classes(m, split)

for split in ["validation", "test-clean", "bfcl"]:
    P(f"\n## Error classes, {split} (rows; coarse then fine)")
    coarse_all = sorted({c for m in MODELS for c, _ in CL[m, split].values()} - {"right"})
    P("model | " + " | ".join(coarse_all))
    for m in MODELS:
        cc = Counter(c for c, _ in CL[m, split].values())
        P(f"{m} | " + " | ".join(str(cc[c]) for c in coarse_all))
    fine_all = Counter(f for m in MODELS for c, f in CL[m, split].values() if c != "right")
    fines = [f for f, _ in fine_all.most_common()]
    P("\nfine | " + " | ".join(MODELS))
    for f in fines:
        P(
            f"{f} | "
            + " | ".join(str(sum(1 for c, ff in CL[m, split].values() if ff == f)) for m in MODELS)
        )

# D vs B discordant rows: class of the loser's error
for split in ["validation", "test-clean", "bfcl"]:
    pb, pd = preds("B", split), preds("D", split)
    ids = sorted(pb)
    bw = [i for i in ids if pb[i]["exact"] and not pd[i]["exact"]]
    dw = [i for i in ids if pd[i]["exact"] and not pb[i]["exact"]]
    P(
        f"\n## {split}: rows only B right ({len(bw)}): D's error | rows only D right ({len(dw)}): B's error"
    )
    cb = Counter(CL["D", split][i][1] for i in bw)
    cd = Counter(CL["B", split][i][1] for i in dw)
    for f in sorted(set(cb) | set(cd), key=lambda f: -(cb[f] + cd[f])):
        P(f"{f} | D wrong (B right): {cb[f]} | B wrong (D right): {cd[f]}")

# By slice, D vs B
for split in ["validation", "test-clean"]:
    pb, pd, p2 = preds("B", split), preds("D", split), preds("2B", split)
    P(f"\n## {split}: B vs D by slice (exact; only B / only D, p) and 2B")
    for dim, (assign, buckets) in DIMENSIONS.items():
        for bk in buckets:
            ids = [i for i in pb if assign(EX[i]) == bk]
            if not ids:
                continue
            ob, od, pv = mcnemar([pb[i]["exact"] for i in ids], [pd[i]["exact"] for i in ids])
            P(
                f"{dim}={bk} | n={len(ids)} | B {pct(sum(pb[i]['exact'] for i in ids) / len(ids))} | D {pct(sum(pd[i]['exact'] for i in ids) / len(ids))} | 2B {pct(sum(p2[i]['exact'] for i in ids) / len(ids))} | {ob}/{od} p={pv:.3g}"
            )

pb, pd = preds("B", "bfcl"), preds("D", "bfcl")
P("\n## BFCL: B vs D by category")
for cat in ["simple", "multiple", "parallel", "parallel_multiple"]:
    ids = [i for i in pb if BF[i]["category"] == cat]
    ob, od, pv = mcnemar([pb[i]["exact"] for i in ids], [pd[i]["exact"] for i in ids])
    row = " | ".join(
        f"{m} {pct(sum(preds(m, 'bfcl')[i]['exact'] for i in ids) / len(ids))}" for m in MODELS
    )
    P(f"{cat} n={len(ids)} | {row} | B/D only {ob}/{od} p={pv:.3g}")

# Output length and call counts
P("\n## Output length (chars) and calls per row vs gold")
for split in ["validation", "bfcl"]:
    for m in MODELS:
        p = preds(m, split)
        L = [len(r["prediction"]) for r in p.values()]
        diffs = Counter()
        for i, r in p.items():
            c = parse_calls(r["prediction"])
            g = len(r["gold"])
            if c is None:
                diffs["unparsed"] += 1
            else:
                diffs["fewer" if len(c) < g else "more" if len(c) > g else "same"] += 1
        P(
            f"{split} {m}: mean chars {sum(L) / len(L):.0f}, max {max(L)} | calls vs gold: {dict(diffs)}"
        )

open(os.path.join(os.path.dirname(__file__), "out/s2_taxonomy.md"), "w").write("\n".join(out))
print("\n".join(out))
