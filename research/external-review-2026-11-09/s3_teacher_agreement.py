"""s3: does each student agree with the 2B teacher row by row? Where the teacher is wrong, who copies
the teacher's exact wrong answer? Also: B beats its teacher - format vs content."""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from common import examples, mcnemar, pct, preds
from taxonomy import classify_xlam

from fc_distill.scoring import call_multiset, parse_calls

EX = examples()
MODELS = ["0.8B", "A", "B", "Bf", "D", "D500", "BD"]
out = []
P = out.append


def key(text):
    c = parse_calls(text)
    if c is None:
        return None
    try:
        return call_multiset(c)
    except TypeError:
        return None


for split in ["validation", "test-clean", "bfcl"]:
    t = preds("2B", split)
    tk = {i: key(r["prediction"]) for i, r in t.items()}
    t_wrong_parsed = [i for i in t if not t[i]["exact"] and tk[i] is not None]
    t_right = [i for i in t if t[i]["exact"]]
    P(f"\n## {split}: agreement with the 2B (identical parsed calls)")
    P(
        f"teacher right {len(t_right)}, teacher wrong but parsed {len(t_wrong_parsed)}, teacher unparsed {sum(tk[i] is None for i in t)}"
    )
    P(
        "model | agree with 2B, all rows | agree on teacher-wrong-parsed rows (copies the error) | right on teacher-wrong-parsed rows | wrong on teacher-right rows"
    )
    for m in MODELS:
        p = preds(m, split)
        mk = {i: key(r["prediction"]) for i, r in p.items()}
        agree = sum(mk[i] is not None and mk[i] == tk[i] for i in t)
        copy = sum(mk[i] == tk[i] for i in t_wrong_parsed)
        right = sum(p[i]["exact"] for i in t_wrong_parsed)
        lose = sum(not p[i]["exact"] for i in t_right)
        P(
            f"{m} | {pct(agree / len(t))} | {copy}/{len(t_wrong_parsed)} ({pct(copy / len(t_wrong_parsed))}) | {right} | {lose}"
        )

# B beats its teacher: format vs content on validation and test-clean
for split in ["validation", "test-clean"]:
    t, b, d = preds("2B", split), preds("B", split), preds("D", split)
    both = [i for i in t if t[i]["parse_ok"] and b[i]["parse_ok"]]
    ob, ot, pv = mcnemar([b[i]["exact"] for i in both], [t[i]["exact"] for i in both])
    P(
        f"\n## {split}: B vs 2B where both parse (n={len(both)}): B {pct(sum(b[i]['exact'] for i in both) / len(both))}, 2B {pct(sum(t[i]['exact'] for i in both) / len(both))}, only B {ob} / only 2B {ot}, p={pv:.3g}"
    )
    # Same under lenient
    ob, ot, pv = mcnemar(
        [b[i]["lenient_exact"] for i in both], [t[i]["lenient_exact"] for i in both]
    )
    P(
        f"  lenient: B {pct(sum(b[i]['lenient_exact'] for i in both) / len(both))}, 2B {pct(sum(t[i]['lenient_exact'] for i in both) / len(both))}, {ob}/{ot}, p={pv:.3g}"
    )
    # Teacher's error class on rows B gets right and the teacher misses (parsed)
    rows = [i for i in both if b[i]["exact"] and not t[i]["exact"]]
    P(
        "  2B's error on rows B right, 2B wrong (both parse): "
        + str(Counter(classify_xlam(t[i]["prediction"], EX[i])[1] for i in rows).most_common(8))
    )
    rows = [i for i in both if t[i]["exact"] and not b[i]["exact"]]
    P(
        "  B's error on rows 2B right, B wrong (both parse): "
        + str(Counter(classify_xlam(b[i]["prediction"], EX[i])[1] for i in rows).most_common(8))
    )
    # Teacher unparsed: why? length
    un = [i for i in t if not t[i]["parse_ok"]]
    L = sorted(len(t[i]["prediction"]) for i in un)
    P(
        f"  2B unparsed {len(un)}: median chars {L[len(L) // 2]}, >2000 chars: {sum(x > 2000 for x in L)}; B right on {sum(b[i]['exact'] for i in un)}, D right on {sum(d[i]['exact'] for i in un)}"
    )

# Teacher on train: unparsed reasons
t = preds("2B", "train")
un = [r for r in t.values() if not r["parse_ok"]]
L = sorted(len(r["prediction"]) for r in un)
P(
    f"\n## train: 2B unparsed {len(un)} of {len(t)}; median chars {L[len(L) // 2]}; >3000 chars {sum(x > 3000 for x in L)}"
)
open(os.path.join(os.path.dirname(__file__), "out/s3_teacher_agreement.md"), "w").write(
    "\n".join(out)
)
print("\n".join(out))
