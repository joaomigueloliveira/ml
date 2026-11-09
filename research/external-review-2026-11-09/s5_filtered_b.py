"""s5: what filtered B's lenient filter removed from the teacher's train outputs (by slice and the
teacher's error class), where Bf gains over B, and what is left between A and Bf: label conventions
vs capability on rows the filter dropped (proxy: the drop rate of train rows sharing the row's tool
set)."""

import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from common import examples, mcnemar, pct, preds, splits
from taxonomy import classify_xlam

from fc_distill.slices import DIMENSIONS

EX = examples()
out = []
P = out.append
T = preds("2B", "train")
train_ids = splits()["train"]
cls = {i: classify_xlam(T[i]["prediction"], EX[i]) for i in train_ids}
kept_B = {i for i in train_ids if T[i]["parse_ok"]}
kept_Bf = {i for i in train_ids if T[i]["lenient_exact"]}
P(f"train {len(train_ids)}; B keeps {len(kept_B)}; Bf keeps {len(kept_Bf)}")

P("\n## Teacher's train outputs by class (B's training rows), and what Bf keeps")
cc = Counter(cls[i][1] for i in train_ids)
for f, n in cc.most_common():
    P(
        f"{f} | {n} | in B {sum(1 for i in kept_B if cls[i][1] == f)} | in Bf {sum(1 for i in kept_Bf if cls[i][1] == f)}"
    )

P("\n## Train rows by slice: share kept by B and by Bf")
for dim, (assign, buckets) in DIMENSIONS.items():
    for bk in buckets:
        ids = [i for i in train_ids if assign(EX[i]) == bk]
        P(
            f"{dim}={bk} | n={len(ids)} | B keeps {pct(sum(i in kept_B for i in ids) / len(ids))} | Bf keeps {pct(sum(i in kept_Bf for i in ids) / len(ids))} | share of B's rows {pct(sum(i in kept_B for i in ids) / len(kept_B))} | share of Bf's rows {pct(sum(i in kept_Bf for i in ids) / len(kept_Bf))}"
        )

# Bf vs B by slice on validation and test-clean
for split in ["validation", "test-clean"]:
    pb, pf, pa = preds("B", split), preds("Bf", split), preds("A", split)
    P(f"\n## {split}: Bf vs B and A vs Bf by slice")
    for dim, (assign, buckets) in DIMENSIONS.items():
        for bk in buckets:
            ids = [i for i in pb if assign(EX[i]) == bk]
            ob, of, pv = mcnemar([pb[i]["exact"] for i in ids], [pf[i]["exact"] for i in ids])
            oa, of2, pv2 = mcnemar([pa[i]["exact"] for i in ids], [pf[i]["exact"] for i in ids])
            P(
                f"{dim}={bk} n={len(ids)} | B {pct(sum(pb[i]['exact'] for i in ids) / len(ids))} Bf {pct(sum(pf[i]['exact'] for i in ids) / len(ids))} A {pct(sum(pa[i]['exact'] for i in ids) / len(ids))} | B/Bf only {ob}/{of} p={pv:.3g} | A/Bf only {oa}/{of2} p={pv2:.3g}"
            )
    rows = [i for i in pb if pf[i]["exact"] and not pb[i]["exact"]]
    P(
        "  B's error where Bf right, B wrong: "
        + str(Counter(classify_xlam(pb[i]["prediction"], EX[i])[1] for i in rows).most_common(8))
    )
    rows = [i for i in pb if pa[i]["exact"] and not pf[i]["exact"]]
    P(
        f"  Bf's error where A right, Bf wrong ({len(rows)}): "
        + str(Counter(classify_xlam(pf[i]["prediction"], EX[i])[1] for i in rows).most_common(10))
    )

# Tool-set drop-rate proxy
by_tools = defaultdict(list)
for i in train_ids:
    by_tools[frozenset(t["name"] for t in EX[i]["tools"])].append(i)
by_name = defaultdict(list)
for i in train_ids:
    for c in EX[i]["answers"]:
        by_name[c["name"]].append(i)
for split in ["validation", "test-clean"]:
    pa, pf = preds("A", split), preds("Bf", split)
    groups = {"A right, Bf wrong": [], "both right": [], "both wrong": [], "Bf right, A wrong": []}
    for i in pa:
        g = (
            "both right"
            if pa[i]["exact"] and pf[i]["exact"]
            else "A right, Bf wrong"
            if pa[i]["exact"]
            else "Bf right, A wrong"
            if pf[i]["exact"]
            else "both wrong"
        )
        groups[g].append(i)
    P(
        f"\n## {split}: Bf's training coverage of the functions gold calls (train rows calling the same function names)"
    )
    for g, ids in groups.items():
        cov, keep, none = [], [], 0
        for i in ids:
            rows = set()
            for c in EX[i]["answers"]:
                rows |= set(by_name.get(c["name"], []))
            if not rows:
                none += 1
                continue
            keep.append(sum(r in kept_Bf for r in rows) / len(rows))
            cov.append(len(rows))
        if keep:
            P(
                f"{g} | n={len(ids)} | mean share of those train rows Bf keeps {pct(sum(keep) / len(keep))} | median train rows {sorted(cov)[len(cov) // 2]} | no train row {none}"
            )

open(os.path.join(os.path.dirname(__file__), "out/s5_filtered_b.md"), "w").write("\n".join(out))
print("\n".join(out))
