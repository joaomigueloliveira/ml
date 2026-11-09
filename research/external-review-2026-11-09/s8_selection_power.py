"""s8: checkpoint selection on 1,000 validation rows: selection stability (bootstrap), winner's curse
(split-half: select on one half, score on the other), power of paired McNemar at the observed
discordance, Holm correction over the README's headline tests, and paired CIs behind 'tie' claims."""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import mcnemar, paired_diff_ci, preds

CKPTS = {
    "A": [500, 1000, 1500, 2000, 2500, 3000, 3411],
    "B": [500, 1000, 1500, 2000, 2500, 3000, 3277],
    "Bf": [500, 1000, 1500, 2000, 2356],
    "D": [500, 1000, 1500, 2000, 2500, 3000, 3410],
    "BD": [100, 200, 300, 400, 500, 1000],
}
out = []
P = out.append
rng = random.Random(0)
for m, steps in CKPTS.items():
    V = {s: preds(m, "validation", s) for s in steps}
    ids = sorted(V[steps[0]])
    M = [[int(V[s][i]["exact"]) for i in ids] for s in steps]
    n = len(ids)
    # bootstrap selection frequency
    freq = {s: 0 for s in steps}
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        sc = [sum(row[j] for j in idx) for row in M]
        best = max(range(len(steps)), key=lambda k: (sc[k], -k))
        freq[steps[best]] += 1
    # split-half winner's curse
    opt = []
    gap_to_best = []
    for _ in range(2000):
        perm = list(range(n))
        rng.shuffle(perm)
        h1, h2 = perm[: n // 2], perm[n // 2 :]
        s1 = [sum(row[j] for j in h1) / len(h1) for row in M]
        s2 = [sum(row[j] for j in h2) / len(h2) for row in M]
        k = max(range(len(steps)), key=lambda k: (s1[k], -k))
        opt.append(s1[k] - s2[k])
        gap_to_best.append(max(s2) - s2[k])
    full = [sum(r) / n for r in M]
    P(f"{m}: full-validation exact by step {dict(zip(steps, [round(100 * x, 1) for x in full]))}")
    P(
        f"   bootstrap selection frequency: { {s: round(100 * f / 2000) for s, f in freq.items()} } (%)"
    )
    P(
        f"   split-half optimism of the selected checkpoint (500 rows): {100 * sum(opt) / len(opt):.2f} pts; scaled to 1,000 rows ~{100 * sum(opt) / len(opt) / math.sqrt(2):.2f}; mean regret vs the half-2 best {100 * sum(gap_to_best) / len(gap_to_best):.2f} pts"
    )

# Power: minimum detectable difference for paired McNemar, alpha .05 two-sided, power .8
P("\n## Minimum detectable paired difference (80% power, alpha 0.05)")
for name, n, disc in [
    ("validation, B vs D (76 discordant)", 1000, 76),
    ("validation, B vs D500 (106)", 1000, 106),
    ("test-clean, B vs D (119)", 1625, 119),
    ("test-clean, B vs D500 (184)", 1625, 184),
    ("BFCL, B vs D (90)", 1000, 90),
]:
    pd = disc / n
    za, zb = 1.959964, 0.841621
    # solve n = (za*sqrt(pd) + zb*sqrt(pd - d^2))^2 / d^2 for d
    lo, hi = 1e-4, pd
    for _ in range(100):
        d = (lo + hi) / 2
        need = (za * math.sqrt(pd) + zb * math.sqrt(max(pd - d * d, 0))) ** 2 / (d * d)
        lo, hi = (d, hi) if need > n else (lo, d)
    P(f"{name}: MDE ~{100 * d:.1f} pts")

# Holm over the README's headline tests (recomputed from predictions)
tests = [
    ("B vs 2B, test-clean", "B", "2B", "test-clean"),
    ("B vs 2B, BFCL", "B", "2B", "bfcl"),
    ("A vs 2B, BFCL (tie)", "A", "2B", "bfcl"),
    ("D vs B, test-clean (tie)", "D", "B", "test-clean"),
    ("D vs 2B, BFCL (tie)", "D", "2B", "bfcl"),
    ("D vs B, BFCL", "D", "B", "bfcl"),
    ("D500 vs B, test-clean", "D500", "B", "test-clean"),
    ("D500 vs B, BFCL (tie)", "D500", "B", "bfcl"),
    ("BD vs B, test-clean", "BD", "B", "test-clean"),
    ("BD vs B, BFCL (tie)", "BD", "B", "bfcl"),
    ("BD vs D, BFCL", "BD", "D", "bfcl"),
    ("A vs B, test-clean", "A", "B", "test-clean"),
    ("A vs D, test-clean", "A", "D", "test-clean"),
    ("Bf vs B, test-clean", "Bf", "B", "test-clean"),
    ("Bf vs B, BFCL (tie)", "Bf", "B", "bfcl"),
    ("A vs Bf, test-clean", "A", "Bf", "test-clean"),
    ("A vs Bf, BFCL", "A", "Bf", "bfcl"),
    ("A vs D, BFCL (tie)", "A", "D", "bfcl"),
]
res = []
for name, a, b, split in tests:
    pa, pb = preds(a, split), preds(b, split)
    ids = sorted(pa)
    oa, ob, p = mcnemar([pa[i]["exact"] for i in ids], [pb[i]["exact"] for i in ids])
    d, lo, hi = paired_diff_ci([pa[i]["exact"] for i in ids], [pb[i]["exact"] for i in ids])
    res.append((name, oa, ob, p, d, lo, hi))
m = len(res)
order = sorted(range(m), key=lambda k: res[k][3])
holm = {}
running = 0
for rank, k in enumerate(order):
    adj = min(1.0, (m - rank) * res[k][3])
    running = max(running, adj)
    holm[k] = running
P(f"\n## Headline tests (official runs as saved), Holm over {m}")
P("test | only first / only second | p | Holm p | diff [95% CI] pts")
for k, (name, oa, ob, p, d, lo, hi) in enumerate(res):
    P(
        f"{name} | {oa}/{ob} | {p:.3g} | {holm[k]:.3g} | {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    )
open(os.path.join(os.path.dirname(__file__), "out/s8_selection_power.md"), "w").write(
    "\n".join(out)
)
print("\n".join(out))
