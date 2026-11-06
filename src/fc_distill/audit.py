"""Label audit of the test split (issue #8, ADR 0002). Rubric: docs/label-audit.md.

Every test row that at least one model gets wrong is audited. Each audited row gets a verdict on its
gold label: `model_wrong` (the label is sound), `label_ambiguous` or `label_wrong`. `test-clean` is
test minus the ambiguous and wrong rows.

uv run fc-audit prepare   # write data/audit/test_labels.csv: rows to audit, pre-tags, auto verdicts
uv run fc-audit build     # write the test-clean ids into data/splits.json and print the ceiling
"""

import argparse
import csv
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypedDict, get_args

from fc_distill.data import DEFAULT_SPLITS_PATH, Call, Example, load_split
from fc_distill.quality import (
    code_arguments,
    explicit_defaults,
    missing_required,
    type_mismatches,
    without_defaults,
)
from fc_distill.scoring import call_multiset, parse_calls

Verdict = Literal["model_wrong", "label_ambiguous", "label_wrong"]
VERDICTS: tuple[Verdict, ...] = get_args(Verdict)
UNSOUND: frozenset[Verdict] = frozenset({"label_ambiguous", "label_wrong"})

DEFAULT_LABELS_PATH = Path("data/audit/test_labels.csv")
DEFAULT_MODELS = ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B")
FIELDS = ("id", "verdict", "category", "reason", "models_wrong", "tags")

AUTO = "auto: "  # reason prefix of verdicts set by `auto_verdict`, recomputed on every prepare
_PERCENT = re.compile(r"\d\s*%|percent", re.IGNORECASE)


class AuditRow(TypedDict):
    id: int
    verdict: Verdict | Literal[""]
    category: str
    reason: str
    models_wrong: str  # ';'-joined model ids
    tags: str  # ';'-joined pre-tags


def is_auto(row: AuditRow) -> bool:
    return row["reason"].startswith(AUTO)


def pre_tags(example: Example) -> list[str]:
    """Deterministic hints about the gold label, shown to the labeller and kept in the CSV."""
    one = [example]
    found = {
        "code_argument": bool(code_arguments(one)),
        "type_mismatch": bool(type_mismatches(one)),
        "missing_required": bool(missing_required(one)),
        "explicit_default": bool(explicit_defaults(one)),
        "percentage": bool(_PERCENT.search(example["query"])),
    }
    return [tag for tag, hit in found.items() if hit]


def only_defaults_differ(prediction: Sequence[Call] | None, example: Example) -> bool:
    """The prediction is the gold call once optional arguments at their default are dropped."""
    if prediction is None:
        return False
    tools = example["tools"]
    return call_multiset(without_defaults(prediction, tools)) == call_multiset(
        without_defaults(example["answers"], tools)
    )


def auto_verdict(
    example: Example, predictions: Sequence[Sequence[Call] | None]
) -> tuple[Verdict, str, str] | None:
    """(verdict, category, reason) when the outcome doesn't depend on which model erred, else None.

    `predictions` are the parsed outputs of the models that got the row wrong.
    """
    if code_arguments([example]):
        return "label_wrong", "code-valued argument", AUTO + "gold argument is a Python expression"
    if predictions and all(only_defaults_differ(p, example) for p in predictions):
        return "model_wrong", "default made explicit", AUTO + "differs only by an optional default"
    return None


def load_predictions(run_dir: Path) -> dict[int, tuple[bool, str]]:
    """id -> (exact, raw prediction) from an `fc-eval` run."""
    with (run_dir / "predictions.jsonl").open() as f:
        records = [json.loads(line) for line in f]
    return {r["id"]: (r["exact"], r["prediction"]) for r in records}


def prepare(
    test: Sequence[Example],
    predictions: Mapping[str, Mapping[int, tuple[bool, str]]],
    existing: Mapping[int, AuditRow] | None = None,
) -> list[AuditRow]:
    """One row per test example that at least one model gets wrong. Verdicts already in
    `existing` are kept, so re-running after new model runs doesn't lose labelling work.
    Automatic verdicts are always recomputed, so a rule change reaches every row."""
    existing = existing or {}
    rows: list[AuditRow] = []
    for ex in test:
        wrong = [m for m, preds in predictions.items() if not preds[ex["id"]][0]]
        if not wrong:
            continue
        row: AuditRow = {
            "id": ex["id"],
            "verdict": "",
            "category": "",
            "reason": "",
            "models_wrong": ";".join(wrong),
            "tags": ";".join(pre_tags(ex)),
        }
        auto = auto_verdict(ex, [parse_calls(predictions[m][ex["id"]][1]) for m in wrong])
        if auto:
            row["verdict"], row["category"], row["reason"] = auto
        elif (
            (old := existing.get(ex["id"]))
            and old["verdict"]
            and not old["reason"].startswith(AUTO)
        ):
            row |= {"verdict": old["verdict"], "category": old["category"], "reason": old["reason"]}
        rows.append(row)
    return rows


def read_labels(path: Path = DEFAULT_LABELS_PATH) -> dict[int, AuditRow]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[int, AuditRow] = {}
    for r in rows:
        if r["verdict"] and r["verdict"] not in VERDICTS:
            raise ValueError(f"row {r['id']}: unknown verdict {r['verdict']!r}")
        out[int(r["id"])] = {**r, "id": int(r["id"])}  # type: ignore[typeddict-item]
    return out


def write_labels(rows: Sequence[AuditRow], path: Path = DEFAULT_LABELS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def clean_ids(test_ids: Sequence[int], labels: Mapping[int, AuditRow]) -> list[int]:
    """Test minus rows labelled ambiguous or wrong. Every audited row must have a verdict."""
    unlabelled = [i for i, r in labels.items() if not r["verdict"]]
    if unlabelled:
        raise ValueError(f"{len(unlabelled)} audited rows have no verdict, e.g. {unlabelled[:5]}")
    drop = {i for i, r in labels.items() if r["verdict"] in UNSOUND}
    return sorted(set(test_ids) - drop)


def summary(test_ids: Sequence[int], labels: Mapping[int, AuditRow]) -> dict[str, object]:
    clean = clean_ids(test_ids, labels)
    return {
        "test_rows": len(test_ids),
        "audited_rows": len(labels),
        "verdicts": dict(Counter(r["verdict"] for r in labels.values())),
        "categories": dict(Counter(r["category"] for r in labels.values() if r["category"])),
        "test_clean_rows": len(clean),
        "ceiling": round(len(clean) / len(test_ids), 4),
    }


def write_test_clean(ids: Sequence[int], splits_path: Path = DEFAULT_SPLITS_PATH) -> None:
    splits = json.loads(splits_path.read_text())
    splits["test-clean"] = sorted(ids)
    splits_path.write_text(json.dumps(splits, indent=0) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=("prepare", "build"))
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    args = ap.parse_args()

    test = load_split("test")
    if args.command == "prepare":
        predictions = {
            m: load_predictions(args.runs / f"eval-test-{m.replace('/', '_')}") for m in args.models
        }
        existing = read_labels(args.labels) if args.labels.exists() else {}
        rows = prepare(test, predictions, existing)
        write_labels(rows, args.labels)
        todo = sum(not r["verdict"] for r in rows)
        print(f"wrote {args.labels}: {len(rows)} rows to audit, {todo} still need a verdict")
    else:
        labels = read_labels(args.labels)
        ids = [ex["id"] for ex in test]
        write_test_clean(clean_ids(ids, labels))
        print(json.dumps(summary(ids, labels), indent=2))


if __name__ == "__main__":
    main()
