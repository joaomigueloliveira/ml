"""Shared loaders for the review scripts. Read-only on the repo's runs/ and data/."""

import json
import math
import os
from fractions import Fraction
from functools import cache
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

REPO = Path(os.environ.get("FC_REPO", Path(__file__).resolve().parents[2]))
RUNS = REPO / "runs"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

RUN = {
    "A": "train-sft-gold-r16-b4x4-n54561-gl-s0",
    "B": "train-sft-teacher-r16-b4x4-n52424-gl-s0",
    "Bf": "train-sft-teacher-lenient-r16-b4x4-n37696-gl-s0",
    "D": "train-opd-r16-b4x4-n54561-s0",
    "BD": "train-opd-from-sft-teacher-r16-b4x4-n52424-gl-s0-1500-r16-b4x4-n54561-s0",
}
SEL = {"A": 3411, "B": 1500, "Bf": 1000, "D": 3000, "BD": 400, "D500": 500}
RAW = {"0.8B": "Qwen_Qwen3.5-0.8B", "2B": "Qwen_Qwen3.5-2B", "4B": "Qwen_Qwen3.5-4B"}


def run_dir(model: str, split: str, step: int | None = None) -> Path:
    if model in RAW:
        return RUNS / f"eval-{split}-{RAW[model]}"
    base = "D" if model == "D500" else model
    step = SEL[model] if step is None else step
    return RUNS / f"eval-{split}-{RUN[base]}-checkpoint-{step}"


@cache
def splits() -> dict:
    return json.loads((REPO / "data/splits.json").read_text())


def read_preds(path: Path) -> dict:
    out = {}
    with open(path / "predictions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


@cache
def preds(model: str, split: str, step: int | None = None) -> dict:
    """{id: row}. test-clean for raw models is their test run restricted to test-clean ids."""
    if split == "test-clean" and model in RAW:
        ids = set(splits()["test-clean"])
        return {k: v for k, v in preds(model, "test").items() if k in ids}
    return read_preds(run_dir(model, split, step))


@cache
def examples() -> dict:
    from fc_distill.data import load_all

    return {ex["id"]: ex for ex in load_all()}


@cache
def bfcl_rows() -> dict:
    from fc_distill import bfcl

    return {r["id"]: r for r in bfcl.load()}


def mcnemar(a, b):
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    tail = sum(Fraction(math.comb(n, k), 2**n) for k in range(min(only_a, only_b) + 1))
    return only_a, only_b, float(min(Fraction(1), 2 * tail))


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0, c - h), min(1, c + h)


def paired_diff_ci(a, b, z=1.959964):
    """Mean difference a-b of paired 0/1 (or real) outcomes with a normal 95% interval."""
    d = [float(x) - float(y) for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return m, m - z * sd / math.sqrt(n), m + z * sd / math.sqrt(n)


def pct(x):
    return f"{100 * x:.1f}"
