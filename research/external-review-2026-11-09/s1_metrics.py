"""s1: every model's metrics on every split; key pairwise comparisons under exact, lenient_exact,
names (McNemar) and arg_f1 (paired diff CI); accuracy on the audit's unsound test rows; run-to-run
nondeterminism between each trained model's test and test-clean runs (same rows, same weights)."""

import csv
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import REPO, mcnemar, paired_diff_ci, pct, preds, splits

MODELS = ["0.8B", "2B", "4B", "A", "B", "Bf", "D", "D500", "BD"]
SPLITS = ["validation", "test", "test-clean", "bfcl"]
lines = []
P = lines.append

P("## Metrics per model and split")
for split in SPLITS:
    P(f"\n### {split}")
    mets = (
        ["parse_ok", "exact", "lenient_exact", "names", "arg_f1"]
        if split != "bfcl"
        else ["parse_ok", "exact", "bfcl_exact", "names"]
    )
    P("model | n | " + " | ".join(mets))
    for m in MODELS:
        try:
            p = preds(m, split)
        except FileNotFoundError:
            continue
        rows = list(p.values())
        P(
            f"{m} | {len(rows)} | "
            + " | ".join(pct(sum(float(r[k]) for r in rows) / len(rows)) for k in mets)
        )

PAIRS = [
    ("B", "D"),
    ("B", "Bf"),
    ("B", "2B"),
    ("D", "2B"),
    ("A", "Bf"),
    ("B", "BD"),
    ("D500", "B"),
    ("Bf", "D"),
]
P("\n## Pairwise comparisons (first minus second)")
for split in SPLITS:
    P(f"\n### {split}")
    mets = (
        ["exact", "lenient_exact", "names"] if split != "bfcl" else ["exact", "bfcl_exact", "names"]
    )
    P(
        "pair | "
        + " | ".join(f"{k}: only1/only2 p" for k in mets)
        + (" | arg_f1 diff [95%]" if split != "bfcl" else "")
    )
    for a, b in PAIRS:
        try:
            pa, pb = preds(a, split), preds(b, split)
        except FileNotFoundError:
            continue
        ids = sorted(set(pa) & set(pb))
        cells = []
        for k in mets:
            oa, ob, pv = mcnemar([pa[i][k] for i in ids], [pb[i][k] for i in ids])
            cells.append(f"{oa}/{ob} p={pv:.3g}")
        if split != "bfcl":
            m_, lo, hi = paired_diff_ci(
                [pa[i]["arg_f1"] for i in ids], [pb[i]["arg_f1"] for i in ids]
            )
            cells.append(f"{100 * m_:+.1f} [{100 * lo:+.1f},{100 * hi:+.1f}]")
        P(f"{a} vs {b} | " + " | ".join(cells))

# Unsound rows (ambiguous / wrong) of test: who matches the published label there?
audit = {int(r["id"]): r for r in csv.DictReader(open(REPO / "data/audit/test_labels.csv"))}
clean = set(splits()["test-clean"])
test_ids = set(splits()["test"])
amb = {i for i, r in audit.items() if r["verdict"] == "label_ambiguous"}
wrong = {i for i, r in audit.items() if r["verdict"] == "label_wrong"}
unaudited = test_ids - set(audit)
P(
    f"\n## Test rows by audit status: unaudited {len(unaudited)}, ambiguous {len(amb)}, wrong {len(wrong)}, clean {len(clean)}"
)
P(
    "model | exact on ambiguous | exact on wrong | exact on audited-sound | exact on unaudited (assumed sound) | errors on unaudited"
)
for m in MODELS:
    p = preds(m, "test")

    def acc(s):
        return sum(p[i]["exact"] for i in s) / len(s)

    sound_aud = clean - unaudited
    P(
        f"{m} | {pct(acc(amb))} | {pct(acc(wrong))} | {pct(acc(sound_aud))} | {pct(acc(unaudited))} | {sum(not p[i]['exact'] for i in unaudited)}"
    )

# Run-to-run nondeterminism: test run vs test-clean run of the same checkpoint
P("\n## Same checkpoint, two runs (test run restricted to test-clean ids vs the test-clean run)")
P(
    "model | rows | prediction text differs | exact differs | exact in test run | exact in test-clean run"
)
for m in ["A", "B", "Bf", "D", "D500", "BD"]:
    pt, pc = preds(m, "test"), preds(m, "test-clean")
    ids = sorted(set(pc))
    dtext = sum(pt[i]["prediction"] != pc[i]["prediction"] for i in ids)
    dex = sum(pt[i]["exact"] != pc[i]["exact"] for i in ids)
    P(
        f"{m} | {len(ids)} | {dtext} | {dex} | {pct(sum(pt[i]['exact'] for i in ids) / len(ids))} | {pct(sum(pc[i]['exact'] for i in ids) / len(ids))}"
    )

open(
    __import__("os").path.join(__import__("os").path.dirname(__file__), "out/s1_metrics.md"), "w"
).write("\n".join(lines))
print("\n".join(lines))
