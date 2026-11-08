"""Method results (#46): results/methods.json and results/methods.md from the runs.

uv run --no-sync fc-report --methods

For each method: every checkpoint's validation score against its GPU-hours, the selected checkpoint
(ADR 0006: best strict exact, ties to the earlier), and the selected checkpoint's official test,
test-clean and BFCL scores once they exist. Test-clean is scored from the test run restricted to
its ids, as in the baselines report. Label-free methods are the main question; methods with labels
are the reference (ADR 0007).
"""

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from fc_distill.data import DEFAULT_SPLITS_PATH, Example, load_split
from fc_distill.evaluate import read_run
from fc_distill.scoring import is_optional, parse_calls
from fc_distill.stats import mcnemar, metric_block


class Method(NamedTuple):
    key: str
    name: str
    train_run: str
    labels: bool


METHODS = (
    Method("A", "SFT on gold labels", "train-sft-gold-r16-b4x4-n54561-gl-s0", labels=True),
    Method("B", "SFT on teacher outputs", "train-sft-teacher-r16-b4x4-n52424-gl-s0", labels=False),
)
STUDENT, TEACHER = "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"
BASELINES = {STUDENT: "raw 0.8B", TEACHER: "raw 2B"}
OFFICIAL = ("test", "test-clean", "bfcl")
METRICS = ("parse_ok", "exact", "lenient_exact")
DEFAULT_JSON = Path("results/methods.json")
DEFAULT_MD = Path("results/methods.md")
_STEP = re.compile(r"-checkpoint-(\d+)$")
TEACHER_RUN = "eval-train-Qwen_Qwen3.5-2B"


def select(checkpoints: Sequence[Mapping[str, Any]]) -> tuple[int, list[int]]:
    """ADR 0006: the best strict exact wins, an exact tie goes to the earlier step. Also the steps
    whose score lies inside the winner's 95% interval: statistically tied with it."""
    best = max(checkpoints, key=lambda c: (c["exact"], -c["step"]))
    low, high = best["exact_ci"]
    tied = [c["step"] for c in checkpoints if c is not best and low <= c["exact"] <= high]
    return best["step"], sorted(tied)


def gains(score: float, raw: float, teacher: float, a: float) -> dict[str, float]:
    """Gap recovered towards the teacher, and share of A's gain (GLOSSARY)."""
    return {
        "gap_recovered": round((score - raw) / (teacher - raw), 4),
        "share_of_a": round((score - raw) / (a - raw), 4),
    }


def validity(prediction: str, tools: Sequence[Mapping[str, Any]]) -> str:
    """A label-free check of one teacher output (ADR 0007): "unparsed", "unknown function" (a call
    to a tool that wasn't offered), "missing required" (a required argument left out) or "valid"."""
    calls = parse_calls(prediction)
    if calls is None:
        return "unparsed"
    specs = {t["name"]: t.get("parameters") or {} for t in tools}
    for call in calls:
        if call["name"] not in specs:
            return "unknown function"
        required = {a for a, spec in specs[call["name"]].items() if not is_optional(spec)}
        if not required <= set(call["arguments"]):
            return "missing required"
    return "valid"


def teacher_counts(
    records: Sequence[Mapping[str, Any]], examples: Mapping[int, Example]
) -> dict[str, Any]:
    """How many teacher outputs each B variant keeps, and how many of those match gold."""
    out: dict[str, Any] = {"rows": len(records)}
    kept: dict[str, list[Mapping[str, Any]]] = {"unfiltered": [], "valid": [], "lenient": []}
    for r in records:
        verdict = validity(r["prediction"], examples[r["id"]]["tools"])
        out[verdict] = out.get(verdict, 0) + 1
        if verdict != "unparsed":
            kept["unfiltered"].append(r)
        if verdict == "valid":
            kept["valid"].append(r)
        if r["lenient_exact"]:
            kept["lenient"].append(r)
    out["variants"] = {
        name: {"kept": len(rows), "exact": round(sum(r["exact"] for r in rows) / len(rows), 4)}
        for name, rows in kept.items()
        if rows
    }
    return out


def _eval_dir(split: str, run: str) -> str:
    """The eval run an official split is read from: test-clean comes from the test run."""
    return f"eval-{'test' if split == 'test-clean' else split}-{run.replace('/', '_')}"


def _records(runs: Path, run: str, split: str, clean: set[int]) -> list[dict[str, Any]] | None:
    """A run's records for an official split (test-clean from the test run), or None if absent."""
    path = runs / run
    if not (path / "summary.json").exists():
        return None
    _, records = read_run(path)
    return [r for r in records if r["id"] in clean] if split == "test-clean" else records


def build(runs: Path, splits_path: Path = DEFAULT_SPLITS_PATH) -> dict[str, Any]:
    clean = set(json.loads(splits_path.read_text())["test-clean"])
    exact: dict[tuple[str, str], dict[Any, bool]] = {}

    def keep(who: str, split: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        exact[who, split] = {r["id"]: r["exact"] for r in records}
        return metric_block(records, ("exact",) if split == "bfcl" else METRICS)

    validation = {
        label: keep(
            model, "validation", read_run(runs / f"eval-validation-{model.replace('/', '_')}")[1]
        )
        for model, label in BASELINES.items()
    }
    official_baselines: dict[str, dict[str, Any]] = {}
    for model, label in BASELINES.items():
        official_baselines[label] = {}
        for split in OFFICIAL:
            records = _records(runs, _eval_dir(split, model), split, clean)
            if records is not None:
                official_baselines[label][split] = keep(model, split, records)

    methods: dict[str, Any] = {}
    for m in METHODS:
        checkpoints = []
        for path in sorted(runs.glob(f"eval-validation-{m.train_run}-checkpoint-*")):
            match = _STEP.search(path.name)
            if match is None:
                continue
            summary, records = read_run(path)
            step = int(match.group(1))
            block = keep(f"{m.key}@{step}", "validation", records)
            checkpoints.append(
                {"step": step, "gpu_hours": round(summary["adapter"]["gpu_hours"], 3)} | block
            )
        if not checkpoints:
            continue
        checkpoints.sort(key=lambda c: c["step"])
        step, tied = select(checkpoints)
        exact[m.key, "validation"] = exact[f"{m.key}@{step}", "validation"]
        official = {}
        for split in OFFICIAL:
            run = (
                f"eval-{'test' if split == 'test-clean' else split}-{m.train_run}-checkpoint-{step}"
            )
            records = _records(runs, run, split, clean)
            if records is not None:
                official[split] = keep(m.key, split, records)
        methods[m.key] = {
            "name": m.name,
            "labels": m.labels,
            "train_run": m.train_run,
            "checkpoints": checkpoints,
            "selected": step,
            "tied": tied,
            "official": official,
        }

    def score(who: str, split: str) -> float | None:
        if who in BASELINES.values():
            block = validation[who] if split == "validation" else official_baselines[who].get(split)
        else:
            m = methods.get(who)
            if m is None:
                return None
            block = (
                next(c for c in m["checkpoints"] if c["step"] == m["selected"])
                if split == "validation"
                else m["official"].get(split)
            )
        return None if block is None else float(block["exact"])

    headline = {}
    for key in methods:
        for split in ("validation", "test-clean"):
            values = [score(w, split) for w in (key, "raw 0.8B", "raw 2B", "A")]
            if all(v is not None for v in values):
                own, raw, t2b, a = (float(v) for v in values if v is not None)
                headline[f"{key}/{split}"] = gains(own, raw, t2b, a)

    ids = {"raw 0.8B": STUDENT, "raw 2B": TEACHER}
    comparisons = []
    for key in methods:
        for other in ("raw 0.8B", "raw 2B", *(k for k in methods if k < key)):
            for split in ("validation", *OFFICIAL):
                a_rows = exact.get((ids.get(other, other), split))
                b_rows = exact.get((key, split))
                if a_rows is None or b_rows is None:
                    continue
                order = sorted(b_rows, key=str)
                result = mcnemar([a_rows[i] for i in order], [b_rows[i] for i in order])
                comparisons.append({"a": other, "b": key, "eval": split} | result)
    teacher: dict[str, Any] | None = None
    if (runs / TEACHER_RUN / "summary.json").exists():
        t_summary, t_records = read_run(runs / TEACHER_RUN)
        train = {ex["id"]: ex for ex in load_split("train", splits_path)}
        teacher = teacher_counts(t_records, train) | {
            "gpu_hours": round(t_summary["seconds"] / 3600, 3)
        }
    return {
        "teacher": teacher,
        "baselines": {"validation": validation, "official": official_baselines},
        "methods": methods,
        "headline": headline,
        "mcnemar": comparisons,
    }


def _pct(block: Mapping[str, Any] | None, metric: str = "exact") -> str:
    if block is None or block.get(metric) is None:
        return "-"
    low, high = block[f"{metric}_ci"]
    return f"{100 * block[metric]:.1f} [{100 * low:.1f}, {100 * high:.1f}]"


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    align = ["---"] + ["---:"] * (len(head) - 1)
    return [
        "| " + " | ".join(head) + " |",
        "|" + "|".join(align) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def render(data: Mapping[str, Any]) -> str:
    methods = data["methods"]
    base = data["baselines"]
    out = [
        "# Method results",
        "",
        "Issues #42, #46. Generated by `uv run --no-sync fc-report --methods` from the runs in "
        "`runs/`; numbers: [`methods.json`](methods.json). Strict exact with 95% intervals. "
        "Checkpoints are chosen on validation (ADR 0006); test, test-clean and BFCL are run only "
        "for the selected one. Label-free methods are the main question; methods with labels are "
        "the reference (ADR 0007).",
        "",
        "## Headline",
        "",
    ]
    head = ["Model", "Labels", "GPU-h", "Validation", *OFFICIAL]
    rows = [
        [
            label,
            "-",
            "0",
            _pct(base["validation"][label]),
            *(_pct(base["official"][label].get(s)) for s in OFFICIAL),
        ]
        for label in BASELINES.values()
    ]
    for key, m in methods.items():
        sel = next(c for c in m["checkpoints"] if c["step"] == m["selected"])
        rows.append(
            [
                f"**{key}**, {m['name']} (step {m['selected']})",
                "yes" if m["labels"] else "no",
                f"{sel['gpu_hours']:.2f}",
                _pct(sel),
                *(_pct(m["official"].get(s)) for s in OFFICIAL),
            ]
        )
    out += _table(head, rows)
    out += [
        "",
        "GPU-hours are training step time plus, for B, the teacher's generation over the train "
        "split (1.43 h); validation scoring is not included (ADR 0006).",
        "",
        "Gap recovered = (method - raw 0.8B) / (raw 2B - raw 0.8B); share of A's gain = "
        "(method - raw 0.8B) / (A - raw 0.8B). #2's success criterion is a gap recovered of at "
        "least 50% on test-clean for the best label-free method.",
        "",
    ]
    out += _table(
        ["Method", "Eval", "Gap recovered", "Share of A's gain"],
        [
            [
                k.split("/")[0],
                k.split("/")[1],
                f"{100 * v['gap_recovered']:.0f}%",
                f"{100 * v['share_of_a']:.0f}%",
            ]
            for k, v in data["headline"].items()
        ],
    )
    out += ["", "## Validation curves", ""]
    for key, m in methods.items():
        out += [f"### {key}, {m['name']} (`{m['train_run']}`)", ""]
        out += _table(
            ["Step", "GPU-h", *METRICS],
            [
                [
                    f"**{c['step']}**" if c["step"] == m["selected"] else str(c["step"]),
                    f"{c['gpu_hours']:.2f}",
                    *(_pct(c, x) for x in METRICS),
                ]
                for c in m["checkpoints"]
            ],
        )
        tied = ", ".join(str(s) for s in m["tied"]) or "none"
        out += [
            "",
            f"Selected: step {m['selected']}. Inside its interval (statistically tied): {tied}.",
            "",
        ]
    out += ["## Paired comparisons", "", "Exact McNemar test on strict exact, row by row.", ""]
    out += _table(
        ["A", "B", "Eval", "Rows", "Only A right", "Only B right", "p"],
        [
            [
                c["a"],
                c["b"],
                c["eval"],
                f"{c['n']:,}",
                str(c["only_a_right"]),
                str(c["only_b_right"]),
                "< 0.0001" if c["p"] < 1e-4 else f"{c['p']:.4f}",
            ]
            for c in data["mcnemar"]
        ],
    )
    teacher = data["teacher"]
    if teacher:
        variants = teacher["variants"]
        out += [
            "",
            "## Teacher outputs (method B)",
            "",
            f"The 2B answered all {teacher['rows']:,} train rows in {teacher['gpu_hours']:.2f} "
            "GPU-hours (#44). Rows each variant would train on, and how many of them match gold:",
            "",
            *_table(
                ["Variant", "Uses labels", "Rows kept", "Exact vs gold"],
                [
                    [
                        "unfiltered (parses), method B",
                        "no",
                        f"{variants['unfiltered']['kept']:,}",
                        f"{100 * variants['unfiltered']['exact']:.1f}",
                    ],
                    [
                        "validity check (tools offered, required arguments present)",
                        "no",
                        f"{variants['valid']['kept']:,}",
                        f"{100 * variants['valid']['exact']:.1f}",
                    ],
                    [
                        "lenient match with gold, filtered B",
                        "yes",
                        f"{variants['lenient']['kept']:,}",
                        f"{100 * variants['lenient']['exact']:.1f}",
                    ],
                ],
            ),
            "",
            f"The validity check drops {teacher.get('unknown function', 0):,} outputs calling a "
            f"tool that wasn't offered and {teacher.get('missing required', 0):,} missing a "
            "required argument. It moves the kept rows' agreement with gold from "
            f"{100 * variants['unfiltered']['exact']:.1f}% "
            f"to {100 * variants['valid']['exact']:.1f}%, "
            "so it is reported, not trained (ADR 0007).",
        ]
    return "\n".join(out) + "\n"
