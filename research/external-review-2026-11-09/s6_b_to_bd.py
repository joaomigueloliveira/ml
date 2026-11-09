"""s6: what changed from B to B->D (every B->D checkpoint on validation): outputs changed, flips
right->wrong and wrong->right, the error classes, and whether the new output is the teacher's."""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from common import examples, pct, preds
from taxonomy import classify_xlam

from fc_distill.scoring import call_multiset, parse_calls

EX = examples()
out = []
P = out.append


def key(t):
    c = parse_calls(t)
    return None if c is None else call_multiset(c)


for split, steps in [("validation", [100, 200, 300, 400, 500, 1000]), ("test-clean", [400])]:
    b, t = preds("B", split), preds("2B", split)
    bk = {i: key(r["prediction"]) for i, r in b.items()}
    tk = {i: key(r["prediction"]) for i, r in t.items()}
    P(f"\n## {split}: B -> B->D")
    P(
        "step | outputs changed | right->wrong | wrong->right | of changed rows: new output = 2B's | old (B) output = 2B's | lenient r->w / w->r"
    )
    for s in steps:
        d = preds("BD", split, s)
        dk = {i: key(r["prediction"]) for i, r in d.items()}
        ch = [i for i in b if dk[i] != bk[i]]
        rw = [i for i in b if b[i]["exact"] and not d[i]["exact"]]
        wr = [i for i in b if d[i]["exact"] and not b[i]["exact"]]
        lrw = sum(b[i]["lenient_exact"] and not d[i]["lenient_exact"] for i in b)
        lwr = sum(d[i]["lenient_exact"] and not b[i]["lenient_exact"] for i in b)
        P(
            f"{s} | {len(ch)} | {len(rw)} | {len(wr)} | {sum(dk[i] == tk[i] and dk[i] is not None for i in ch)} | {sum(bk[i] == tk[i] and bk[i] is not None for i in ch)} | {lrw}/{lwr}"
        )
        if s in (400, 1000):
            P(
                f"  step {s} B->D's error on right->wrong rows: "
                + str(
                    Counter(classify_xlam(d[i]["prediction"], EX[i])[1] for i in rw).most_common(6)
                )
            )
            P(
                f"  step {s} B's error on wrong->right rows: "
                + str(
                    Counter(classify_xlam(b[i]["prediction"], EX[i])[1] for i in wr).most_common(6)
                )
            )
            P(
                f"  step {s} right->wrong rows where B->D now outputs the 2B's answer: {sum(dk[i] == tk[i] for i in rw)} of {len(rw)}"
            )

# D's own checkpoints: agreement with the teacher over training, validation
t = preds("2B", "validation")
tk = {i: key(r["prediction"]) for i, r in t.items()}
P("\n## D over training, validation: agreement with the 2B and exact")
for s in [500, 1000, 1500, 2000, 2500, 3000, 3410]:
    d = preds("D", "validation", s)
    dk = {i: key(r["prediction"]) for i, r in d.items()}
    agree = sum(dk[i] == tk[i] and dk[i] is not None for i in t)
    c = Counter(classify_xlam(d[i]["prediction"], EX[i])[0] for i in d)
    P(
        f"D {s} | exact {pct(sum(r['exact'] for r in d.values()) / 1000)} | lenient {pct(sum(r['lenient_exact'] for r in d.values()) / 1000)} | agree with 2B {pct(agree / 1000)} | classes {dict(c)}"
    )
for s in [500, 1000, 1500, 2000, 2500, 3000, 3277]:
    d = preds("B", "validation", s)
    dk = {i: key(r["prediction"]) for i, r in d.items()}
    agree = sum(dk[i] == tk[i] and dk[i] is not None for i in t)
    P(
        f"B {s} | exact {pct(sum(r['exact'] for r in d.values()) / 1000)} | agree with 2B {pct(agree / 1000)}"
    )

# The step-1000 dip of D: what broke between 500 and 1000 and came back at 1500?
d5, d10, d15 = (
    preds("D", "validation", 500),
    preds("D", "validation", 1000),
    preds("D", "validation", 1500),
)
rw = [i for i in d5 if d5[i]["exact"] and not d10[i]["exact"]]
P(
    f"\n## D step 500 -> 1000: right->wrong {len(rw)}, wrong->right {sum(d10[i]['exact'] and not d5[i]['exact'] for i in d5)}; recovered by 1500: {sum(d15[i]['exact'] for i in rw)}"
)
P(
    "  step-1000 error on those rows: "
    + str(Counter(classify_xlam(d10[i]["prediction"], EX[i])[1] for i in rw).most_common(8))
)
P(
    "  all step-1000 errors: "
    + str(
        Counter(
            classify_xlam(d10[i]["prediction"], EX[i])[1] for i in d10 if not d10[i]["exact"]
        ).most_common(8)
    )
)
P(
    "  all step-1500 errors: "
    + str(
        Counter(
            classify_xlam(d15[i]["prediction"], EX[i])[1] for i in d15 if not d15[i]["exact"]
        ).most_common(8)
    )
)
open(os.path.join(os.path.dirname(__file__), "out/s6_b_to_bd.md"), "w").write("\n".join(out))
print("\n".join(out))
