"""s7b: the training probe (HF generate, 200 fixed validation rows) vs fc-eval (vLLM, saved adapter)
on the same 200 rows, for every D and B->D checkpoint."""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import RUNS, examples, preds, splits

from fc_distill.opd import probe_rows

EX = examples()
val = [EX[i] for i in splits()["validation"]]
rows = [ex["id"] for ex in probe_rows(val)]
out = []
P = out.append


def probe_log(path):
    vals = {}
    for line in open(path, errors="ignore"):
        m = re.search(r"probe, step (\d+): \{\"probe/validation_exact\": ([0-9.]+)", line)
        if m:
            vals[int(m.group(1))] = float(m.group(2))
    return vals


for name, log, steps in [
    ("D", "log-train-opd.txt", [500, 1000, 1500, 2000, 2500, 3000, 3410]),
    ("BD", "log-train-b-then-d.txt", [100, 200, 300, 400, 500, 1000]),
]:
    pl = probe_log(RUNS / log)
    P(f"\n## {name}: probe (HF) vs fc-eval (vLLM) on the same 200 rows; fc-eval on all 1,000")
    for s in steps:
        p = preds(name, "validation", s)
        fe = sum(p[i]["exact"] for i in rows)
        P(
            f"step {s} | probe {pl.get(s, float('nan')) * 200:.0f}/200 | fc-eval {fe}/200 | diff {pl.get(s, float('nan')) * 200 - fe:+.0f} | fc-eval full {sum(r['exact'] for r in p.values())}/1000"
        )
b = preds("B", "validation")
P(
    f"\nB step 1500 fc-eval on the probe rows: {sum(b[i]['exact'] for i in rows)}/200; full {sum(r['exact'] for r in b.values())}/1000"
)
open(os.path.join(os.path.dirname(__file__), "out/s7b_probe.md"), "w").write("\n".join(out))
print("\n".join(out))
