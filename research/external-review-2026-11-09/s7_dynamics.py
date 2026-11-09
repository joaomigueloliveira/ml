"""s7: D's and B->D's training dynamics from trainer_state.json: loss, rollout parse rate, rollout
length, entropy, grad norm, by 250-step window; and the step-time accounting (GPU-hours)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import RUN, RUNS

out = []
P = out.append
for name, ck in [("D", "checkpoint-3410"), ("BD", "checkpoint-1000")]:
    h = json.load(open(RUNS / RUN[name] / ck / "trainer_state.json"))["log_history"]
    h = [r for r in h if "loss" in r]
    P(f"\n## {name}: {len(h)} log rows, every {h[1]['step'] - h[0]['step']} steps")
    keys = [
        "loss",
        "rollout/parse_ok",
        "completions/mean_length",
        "completions/clipped_ratio",
        "entropy",
        "grad_norm",
        "step_time",
        "learning_rate",
    ]
    P("window | " + " | ".join(keys))
    W = 250 if name == "D" else 100
    last = h[-1]["step"]
    for lo in range(0, last, W):
        rows = [r for r in h if lo < r["step"] <= lo + W]
        if not rows:
            continue
        P(
            f"{lo + 1}-{lo + W} | "
            + " | ".join(f"{sum(r.get(k, 0) for r in rows) / len(rows):.4g}" for k in keys)
        )
    # per-step time total
    st = sum(r.get("step_time", 0) for r in h) * (h[1]["step"] - h[0]["step"]) / 3600
    P(f"approx GPU-hours from step_time x logging interval: {st:.2f}")

# Probe vs fc-eval at B->D (log)
log = open(RUNS / "log-train-b-then-d.txt", errors="ignore").read().splitlines()
P("\n## B->D probe lines")
for line in log:
    if "probe" in line.lower():
        P(line.strip()[:200])
log = open(RUNS / "log-train-opd.txt", errors="ignore").read().splitlines()
P("\n## D probe lines")
for line in log:
    if "probe" in line.lower():
        P(line.strip()[:200])
open(os.path.join(os.path.dirname(__file__), "out/s7_dynamics.md"), "w").write("\n".join(out))
print("\n".join(out))
