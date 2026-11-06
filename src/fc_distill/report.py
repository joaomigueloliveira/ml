"""Official baselines report (issue #10): results/baselines.json and results/baselines.md from runs.

uv run fc-report    # reads runs/eval-{test,bfcl}-Qwen_Qwen3.5-{0.8B,2B,4B}

Test-clean is scored from the test runs restricted to the test-clean ids, not from separate runs:
the predictions are the same, and a second vLLM run could differ slightly because batched greedy
decoding isn't perfectly deterministic.
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any

from fc_distill.data import DEFAULT_SPLITS_PATH, Example, load_split
from fc_distill.scoring import METRICS
from fc_distill.slices import mean_ci, slice_scores

MODELS = ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B")
SHORT = {"Qwen/Qwen3.5-0.8B": "0.8B", "Qwen/Qwen3.5-2B": "2B", "Qwen/Qwen3.5-4B": "4B"}
ROLE = {
    "Qwen/Qwen3.5-0.8B": "student",
    "Qwen/Qwen3.5-2B": "teacher",
    "Qwen/Qwen3.5-4B": "reference",
}
# (a, b, eval): paired comparisons reported with McNemar's test. 0.8B vs 2B on test-clean is the
# one #2's success criterion builds on.
COMPARISONS = (
    ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "test-clean"),
    ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "test"),
    ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "bfcl"),
    ("Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "test-clean"),
    ("Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "bfcl"),
)
DEFAULT_JSON = Path("results/baselines.json")
DEFAULT_MD = Path("results/baselines.md")


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired right/wrong outcomes: a binomial test on the rows
    where exactly one of the two is right."""
    if len(a) != len(b):
        raise ValueError("paired outcomes must have the same length")
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        tail = sum(Fraction(comb(n, k), 2**n) for k in range(min(only_a, only_b) + 1))
        p = float(min(Fraction(1), 2 * tail))
    return {"n": len(a), "only_a_right": only_a, "only_b_right": only_b, "p": p}


def read_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads((run_dir / "summary.json").read_text())
    with (run_dir / "predictions.jsonl").open() as f:
        records = [json.loads(line) for line in f]
    return summary, records


def xlam_block(
    records: Sequence[Mapping[str, Any]], examples: Mapping[int, Example]
) -> dict[str, Any]:
    """Every metric with its 95% interval half-width, and the slices, for a set of scored rows."""
    block: dict[str, Any] = {"n": len(records)}
    for m in METRICS:
        mean, ci = mean_ci([float(r[m]) for r in records])
        block[m], block[f"{m}_ci"] = round(mean, 4), round(ci, 4)
    rows = [examples[r["id"]] for r in records]
    block["slices"] = slice_scores(rows, records, METRICS)
    return block


def build(runs: Path, splits_path: Path = DEFAULT_SPLITS_PATH) -> dict[str, Any]:
    test = {ex["id"]: ex for ex in load_split("test", splits_path)}
    clean = set(json.loads(splits_path.read_text())["test-clean"])
    models: dict[str, Any] = {}
    exact: dict[tuple[str, str], dict[Any, bool]] = {}
    for model in MODELS:
        tag = model.replace("/", "_")
        t_summary, t_records = read_run(runs / f"eval-test-{tag}")
        b_summary, b_records = read_run(runs / f"eval-bfcl-{tag}")
        clean_records = [r for r in t_records if r["id"] in clean]
        models[model] = {
            "role": ROLE[model],
            "test": xlam_block(t_records, test) | {"seconds": t_summary["seconds"]},
            "test-clean": xlam_block(clean_records, test),
            "bfcl": {k: b_summary[k] for k in ("overall", "slices", "categories")}
            | {
                "seconds": b_summary["seconds"],
                "rows_calling_a_train_function_name": b_summary[
                    "rows_calling_a_train_function_name"
                ],
            },
            "provenance": {
                "test": t_summary.get("provenance"),
                "bfcl": b_summary.get("provenance"),
            },
        }
        exact[model, "test"] = {r["id"]: r["exact"] for r in t_records}
        exact[model, "test-clean"] = {r["id"]: r["exact"] for r in clean_records}
        exact[model, "bfcl"] = {r["id"]: r["exact"] for r in b_records}
    comparisons = []
    for a, b, split in COMPARISONS:
        ids = sorted(exact[a, split], key=str)
        result = mcnemar([exact[a, split][i] for i in ids], [exact[b, split][i] for i in ids])
        comparisons.append({"a": a, "b": b, "eval": split} | result)
    return {"models": models, "mcnemar": comparisons}


def _pct(x: float | None, ci: float | None = None) -> str:
    if x is None:
        return "-"
    return f"{100 * x:.1f}" + (f" ± {100 * ci:.1f}" if ci is not None else "")


def _p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def render(data: Mapping[str, Any]) -> str:
    models = data["models"]
    student, teacher = models["Qwen/Qwen3.5-0.8B"], models["Qwen/Qwen3.5-2B"]
    s_clean, t_clean = student["test-clean"]["exact"], teacher["test-clean"]["exact"]
    gap = t_clean - s_clean
    out = [
        "# Baselines",
        "",
        "Issue #10. Generated by `uv run fc-report` from the runs in `runs/`; numbers: "
        "[`baselines.json`](baselines.json). Greedy decoding, thinking off (ADR 0003). Every "
        "number is a percentage with its 95% interval half-width. Test-clean is scored from the "
        f"test runs, restricted to its {student['test-clean']['n']:,} rows.",
        "",
        "## Headline",
        "",
        "| Model | Role | Test | Test-clean | BFCL | Test-clean, lenient |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, r in models.items():
        out.append(
            f"| {model} | {r['role']} | {_pct(r['test']['exact'], r['test']['exact_ci'])} "
            f"| {_pct(r['test-clean']['exact'], r['test-clean']['exact_ci'])} "
            f"| {_pct(r['bfcl']['overall']['exact'], r['bfcl']['overall']['exact_ci'])} "
            f"| {_pct(r['test-clean']['lenient_exact'], r['test-clean']['lenient_exact_ci'])} |"
        )
    out += [
        "",
        "Strict exact match is the headline metric. Lenient exact also accepts optional arguments "
        "at their default (#15).",
        "",
        "## Starting line for Phase 2",
        "",
        f"On test-clean the student scores **{_pct(s_clean)}** and the teacher "
        f"**{_pct(t_clean)}**: a gap of **{100 * gap:.1f} points**. #2's success criterion, "
        f"recovering at least half of it, means a distilled 0.8B at **{_pct(s_clean + gap / 2)}** "
        "or better on test-clean, significant at p < 0.05 by McNemar against the baseline "
        "student, and also improving on BFCL.",
        "",
        "## Paired comparisons",
        "",
        "Exact McNemar test on strict exact match, row by row on the same eval.",
        "",
        "| A | B | Eval | Rows | Only A right | Only B right | p |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for c in data["mcnemar"]:
        out.append(
            f"| {SHORT[c['a']]} | {SHORT[c['b']]} | {c['eval']} | {c['n']:,} "
            f"| {c['only_a_right']} | {c['only_b_right']} | {_p(c['p'])} |"
        )
    out += ["", "## All metrics", ""]
    xlam_metrics = ("parse_ok", "exact", "lenient_exact", "names", "arg_f1")
    for split in ("test", "test-clean"):
        out += [
            f"### {split}",
            "",
            "| Model | Rows | " + " | ".join(xlam_metrics) + " |",
            "|---|---:|" + "---:|" * len(xlam_metrics),
        ]
        for model, r in models.items():
            b = r[split]
            cells = " | ".join(_pct(b[m], b[f"{m}_ci"]) for m in xlam_metrics)
            out.append(f"| {SHORT[model]} | {b['n']:,} | {cells} |")
        out.append("")
    bfcl_metrics = ("parse_ok", "exact", "bfcl_exact", "names")
    out += [
        "### BFCL (unseen tools)",
        "",
        "| Model | Rows | " + " | ".join(bfcl_metrics) + " |",
        "|---|---:|" + "---:|" * len(bfcl_metrics),
    ]
    for model, r in models.items():
        b = r["bfcl"]["overall"]
        cells = " | ".join(_pct(b[m], b[f"{m}_ci"]) for m in bfcl_metrics)
        out.append(f"| {SHORT[model]} | {b['n']:,} | {cells} |")
    rows_seen = next(iter(models.values()))["bfcl"]["rows_calling_a_train_function_name"]
    out += [
        "",
        f"{rows_seen} of the BFCL rows call a function name that also appears in xLAM train. "
        "`bfcl_exact` reimplements BFCL's string normalisation only; see "
        "[`docs/bfcl.md`](../docs/bfcl.md).",
        "",
        "## Slices (strict exact)",
        "",
        "Test-clean. Buckets under 300 rows are marked *.",
        "",
        "| Slice | Bucket | Rows | " + " | ".join(SHORT[m] for m in models) + " | 2B - 0.8B |",
        "|---|---|---:|" + "---:|" * len(models) + "---:|",
    ]
    first = next(iter(models.values()))["test-clean"]["slices"]
    for dim, buckets in first.items():
        for bucket, info in buckets.items():
            per_model = [
                _pct(
                    r["test-clean"]["slices"][dim][bucket]["exact"],
                    r["test-clean"]["slices"][dim][bucket]["exact_ci"],
                )
                for r in models.values()
            ]
            sb = student["test-clean"]["slices"][dim][bucket]["exact"]
            tb = teacher["test-clean"]["slices"][dim][bucket]["exact"]
            diff = "-" if sb is None or tb is None else f"{100 * (tb - sb):.1f}"
            star = "*" if info["too_small"] else ""
            head = f"| {dim} | {bucket}{star} | {info['n']:,} | "
            out.append(head + " | ".join(per_model) + f" | {diff} |")
    out += [
        "",
        "BFCL, by category group (simple has 400 rows; the other three are merged to 600, "
        "ADR 0001):",
        "",
        "| Model | simple | multiple + parallel |",
        "|---|---:|---:|",
    ]
    for model, r in models.items():
        s = r["bfcl"]["slices"]
        out.append(
            f"| {SHORT[model]} | {_pct(s['simple']['exact'], s['simple']['exact_ci'])} "
            f"| {_pct(s['multiple_parallel']['exact'], s['multiple_parallel']['exact_ci'])} |"
        )
    out += [
        "",
        "## Cost and provenance",
        "",
        "Wall-clock time for the full eval on one RTX 3060 (12 GB), vLLM, all rows in one batch. "
        "The 4B runs cap concurrency (`max_num_seqs` 24) to fit in memory, so its time is not "
        "comparable with the others.",
        "",
        "| Model | Test (s) | BFCL (s) | max_num_seqs | Model revision | Commit |",
        "|---|---:|---:|---:|---|---|",
    ]
    for model, r in models.items():
        prov = r["provenance"]["test"] or {}
        rev = (prov.get("model_revision") or "-")[:12]
        commit = (prov.get("commit") or "-")[:7] + (" (dirty)" if prov.get("dirty") else "")
        seqs = prov.get("max_num_seqs") or "default"
        out.append(
            f"| {SHORT[model]} | {r['test']['seconds']} | {r['bfcl']['seconds']} | {seqs} "
            f"| `{rev}` | `{commit}` |"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    args = ap.parse_args()
    data = build(args.runs)
    DEFAULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_JSON.write_text(json.dumps(data, indent=1) + "\n")
    DEFAULT_MD.write_text(render(data))
    print(f"wrote {DEFAULT_JSON} and {DEFAULT_MD}")


if __name__ == "__main__":
    main()
