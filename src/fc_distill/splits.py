"""Building the train / validation / test splits (ADR 0001, 0002).

1. Exact duplicate rows are collapsed to their lowest id, in every split.
2. Rows are grouped by normalized query, and whole groups are assigned to a split, so the same
   query text can never appear in two splits.
3. Train only: rows with code-valued arguments or conflicting labels are removed. Validation and
   test keep their rows exactly as published.
"""

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from fc_distill.data import DEFAULT_SPLITS_PATH, Example, Split
from fc_distill.quality import code_arguments, conflicting_labels, exact_duplicates, normalize_query

VALIDATION_SIZE = 1000
TEST_SIZE = 2000
SEED = 0


class SplitReport(TypedDict):
    rows: int
    exact_duplicates_removed: int
    train_code_arguments_removed: int
    train_conflicting_labels_removed: int
    sizes: dict[str, int]
    # Rows validation and test keep as published although they'd fail a train filter (for #8).
    held_out_code_argument_rows: dict[str, int]
    held_out_conflicting_label_rows: dict[str, int]


def assign_groups(
    examples: Sequence[Example], validation_size: int, test_size: int, seed: int
) -> dict[Split, list[int]]:
    """Randomly assign whole normalized-query groups to test, then validation, then train.

    A group is placed in test or validation only if it fits the remaining room, so both get
    exactly the requested number of rows (groups are almost all of size 1).
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for ex in examples:
        groups[normalize_query(ex["query"])].append(ex["id"])
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    splits: dict[Split, list[int]] = {"train": [], "validation": [], "test": []}
    room: dict[Split, int] = {"test": test_size, "validation": validation_size}
    for key in keys:
        ids = groups[key]
        target: Split = next((s for s in ("test", "validation") if len(ids) <= room[s]), "train")
        if target != "train":
            room[target] -= len(ids)
        splits[target].extend(ids)
    return {name: sorted(ids) for name, ids in splits.items()}


def build_splits(
    examples: Sequence[Example],
    validation_size: int = VALIDATION_SIZE,
    test_size: int = TEST_SIZE,
    seed: int = SEED,
) -> tuple[dict[Split, list[int]], SplitReport]:
    """Deduplicate, split by query group, then filter train. Returns the id lists and a report."""
    duplicates = {i for group in exact_duplicates(examples) for i in sorted(group)[1:]}
    unique = [ex for ex in examples if ex["id"] not in duplicates]
    splits = assign_groups(unique, validation_size, test_size, seed)

    train_ids = set(splits["train"])
    train = [ex for ex in unique if ex["id"] in train_ids]
    code = {issue.row_id for issue in code_arguments(train)}
    conflicting = {i for group in conflicting_labels(train) for i in group}
    splits["train"] = sorted(train_ids - code - conflicting)

    by_id = {ex["id"]: ex for ex in unique}
    held_out = {name: [by_id[i] for i in splits[name]] for name in ("validation", "test")}
    report: SplitReport = {
        "rows": len(examples),
        "exact_duplicates_removed": len(duplicates),
        "train_code_arguments_removed": len(code),
        "train_conflicting_labels_removed": len(conflicting - code),
        "sizes": {name: len(ids) for name, ids in splits.items()},
        "held_out_code_argument_rows": {
            name: len({i.row_id for i in code_arguments(exs)}) for name, exs in held_out.items()
        },
        "held_out_conflicting_label_rows": {
            name: sum(len(g) for g in conflicting_labels(exs)) for name, exs in held_out.items()
        },
    }
    return splits, report


def write_splits(
    splits: dict[Split, list[int]], report: SplitReport, path: Path = DEFAULT_SPLITS_PATH
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**splits, "report": report}, indent=0) + "\n")
