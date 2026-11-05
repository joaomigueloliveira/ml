"""xLAM-60k loading, a fixed train/eval split, and the prompt every model sees."""

import json
import random
from pathlib import Path
from typing import Any, Literal, TypedDict

from datasets import load_dataset

DATASET = "Salesforce/xlam-function-calling-60k"
DEFAULT_SPLITS_PATH = Path("data/splits.json")
EVAL_SIZE = 1000
SEED = 0

SYSTEM_PROMPT = """You are a function-calling assistant. You have access to these tools:

{tools}

Answer the user's request by calling the tools. Reply with ONLY a JSON list of calls, \
in the form [{{"name": "<tool name>", "arguments": {{<argument name>: <value>, ...}}}}]. \
No explanations, no markdown."""

Split = Literal["train", "eval"]


class Call(TypedDict):
    name: str
    arguments: dict[str, Any]


class Example(TypedDict):
    id: int
    query: str
    tools: list[dict[str, Any]]
    answers: list[Call]


class Message(TypedDict):
    role: str
    content: str


def make_splits(n: int, eval_size: int = EVAL_SIZE, seed: int = SEED) -> dict[Split, list[int]]:
    """Deterministically partition row indices 0..n-1 into eval and train."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return {"eval": sorted(idx[:eval_size]), "train": sorted(idx[eval_size:])}


def load_split(split: Split, splits_path: Path = DEFAULT_SPLITS_PATH) -> list[Example]:
    """Return the examples of one split. The split file is created on first use and committed,
    so every experiment is scored on exactly the same held-out rows."""
    ds = load_dataset(DATASET, split="train")
    if splits_path.exists():
        splits = json.loads(splits_path.read_text())
    else:
        splits = make_splits(len(ds))
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        splits_path.write_text(json.dumps(splits))
    return [parse_row(ds[i]) for i in splits[split]]


def parse_row(row: dict[str, Any]) -> Example:
    return {
        "id": row["id"],
        "query": row["query"],
        "tools": json.loads(row["tools"]),
        "answers": json.loads(row["answers"]),
    }


def to_messages(example: Example) -> list[Message]:
    tools = json.dumps(example["tools"], indent=1)
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(tools=tools)},
        {"role": "user", "content": example["query"]},
    ]
