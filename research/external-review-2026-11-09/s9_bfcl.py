"""s9: BFCL rows D gets right and B misses: what B did; the missing-required-argument pattern of the
gold-trained models; does D track the 2B on BFCL row by row (2x2 of D/B right vs 2B right)."""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from common import bfcl_rows, preds
from taxonomy import classify_bfcl

from fc_distill.scoring import parse_calls

BF = bfcl_rows()
out = []
P = out.append
b, d, t, a, f = (preds(m, "bfcl") for m in ["B", "D", "2B", "A", "Bf"])

P("## 2x2 on BFCL vs the 2B: rows where the 2B is right / wrong")
for name, p in [("B", b), ("D", d), ("A", a), ("Bf", f)]:
    tr = [i for i in t if t[i]["exact"]]
    tw = [i for i in t if not t[i]["exact"]]
    P(
        f"{name}: right on 2B-right rows {sum(p[i]['exact'] for i in tr)}/{len(tr)}; right on 2B-wrong rows {sum(p[i]['exact'] for i in tw)}/{len(tw)}"
    )

# Missing required args: which argument, and does its spec description mention a default?
P(
    "\n## 'missing required arg' on BFCL: arg names and whether the spec description mentions a default"
)
for name, p in [("A", a), ("Bf", f), ("B", b), ("D", d)]:
    c = Counter()
    mention = 0
    tot = 0
    for i, r in p.items():
        cl = classify_bfcl(r["prediction"], BF[i])
        if cl[1] != "missing required arg":
            continue
        calls = parse_calls(r["prediction"]) or []
        for call in calls:
            for g in BF[i]["possible"]:
                ((gn, poss),) = g.items()
                if gn != call["name"]:
                    continue
                spec = next((tl for tl in BF[i]["tools"] if tl["name"] == gn), {})
                for k, acc in poss.items():
                    if k not in call["arguments"] and "" not in acc:
                        tot += 1
                        desc = json.dumps(spec.get("parameters", {}).get(k, {})).lower()
                        mention += "default" in desc
                        c[k] += 1
    P(
        f"{name}: missing required args {tot}, spec mentions 'default' for {mention}; top names {c.most_common(6)}"
    )

# Rows only D right: B's error (fine) by category
P("\n## Rows only D right / only B right, by category and loser's error")
for cat in ["simple", "multiple", "parallel", "parallel_multiple"]:
    ids = [i for i in b if BF[i]["category"] == cat]
    dw = [i for i in ids if d[i]["exact"] and not b[i]["exact"]]
    bw = [i for i in ids if b[i]["exact"] and not d[i]["exact"]]
    P(
        f"{cat}: only D {len(dw)} {Counter(classify_bfcl(b[i]['prediction'], BF[i])[1] for i in dw).most_common(5)} | only B {len(bw)} {Counter(classify_bfcl(d[i]['prediction'], BF[i])[1] for i in bw).most_common(4)}"
    )
# of only-D-right rows, how many have D == 2B output?
dw = [i for i in b if d[i]["exact"] and not b[i]["exact"]]
P(f"\nonly-D-right rows where the 2B is also right: {sum(t[i]['exact'] for i in dw)}/{len(dw)}")
bw = [i for i in b if b[i]["exact"] and not d[i]["exact"]]
P(f"only-B-right rows where the 2B is also right: {sum(t[i]['exact'] for i in bw)}/{len(bw)}")
# a few short examples of B's errors on only-D rows
P("\nexamples (B output | D output), truncated:")
for i in dw[:8]:
    P(
        f"{i}: {classify_bfcl(b[i]['prediction'], BF[i])[1]} | B {b[i]['prediction'][:120]} | D {d[i]['prediction'][:120]}"
    )
open(os.path.join(os.path.dirname(__file__), "out/s9_bfcl.md"), "w").write("\n".join(out))
print("\n".join(out))
