"""s11: GPU-hour accounting per checkpoint, rows consumed, the teacher pass's share, and what a
'smallest checkpoint statistically tied with the best' rule would pick; D500 vs D3000 on every
split; the adapter-load check on BFCL (rows changed vs the raw 0.8B, first 32 and all)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import RUN, RUNS, mcnemar, preds, run_dir

out = []
P = out.append
CK = {
    "A": [500, 1000, 1500, 2000, 2500, 3000, 3411],
    "B": [500, 1000, 1500, 2000, 2500, 3000, 3277],
    "Bf": [500, 1000, 1500, 2000, 2356],
    "D": [500, 1000, 1500, 2000, 2500, 3000, 3410],
    "BD": [100, 200, 300, 400, 500, 1000],
}
for m, steps in CK.items():
    cost = json.load(open(RUNS / RUN[m] / "cost.json"))
    P(
        f"\n## {m}: cost.json total {cost.get('total_gpu_hours', cost.get('cost', {}).get('gpu_hours')):.2f} GPU-h; rows {cost.get('cost', {}).get('rows')}; keys {sorted(cost)}"
    )
    best = max(
        steps, key=lambda s: (sum(r["exact"] for r in preds(m, "validation", s).values()), -s)
    )
    bp = preds(m, "validation", best)
    for s in steps:
        p = preds(m, "validation", s)
        sm = json.load(open(run_dir(m, "validation", s) / "summary.json"))
        ids = sorted(p)
        _, _, pv = mcnemar([bp[i]["exact"] for i in ids], [p[i]["exact"] for i in ids])
        P(
            f"step {s}: GPU-h {sm['adapter']['gpu_hours']:.2f}, rows seen {s * 16}, val exact {100 * sum(r['exact'] for r in p.values()) / 1000:.1f}, McNemar vs selected ({best}) p={pv:.3g}"
        )

P("\n## D step 500 vs step 3000")
for split in ["validation", "test", "test-clean", "bfcl"]:
    a, b = preds("D500", split), preds("D", split)
    ids = sorted(a)
    oa, ob, pv = mcnemar([a[i]["exact"] for i in ids], [b[i]["exact"] for i in ids])
    P(f"{split}: only D500 {oa} / only D3000 {ob}, p={pv:.3g}")

P(
    "\n## Adapter check on BFCL: rows whose output differs from the raw 0.8B's (fc-eval reruns the first 32 without the adapter)"
)
raw = preds("0.8B", "bfcl")
order = [json.loads(line)["id"] for line in open(run_dir("0.8B", "bfcl") / "predictions.jsonl")]
for m in ["A", "B", "Bf", "D", "D500", "BD"]:
    p = preds(m, "bfcl")
    sm = json.load(open(run_dir(m, "bfcl") / "summary.json"))
    first = order[:32]
    P(
        f"{m}: first 32 differ from saved raw run {sum(p[i]['prediction'] != raw[i]['prediction'] for i in first)}/32 (fc-eval's check: {sm['adapter'].get('check')}); all {sum(p[i]['prediction'] != raw[i]['prediction'] for i in p)}/1000; raw-0.8B exact on first 32 {sum(raw[i]['exact'] for i in first)}/32"
    )
open(os.path.join(os.path.dirname(__file__), "out/s11_cost.md"), "w").write("\n".join(out))
print("\n".join(out))
