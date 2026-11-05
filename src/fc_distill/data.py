"""xLAM-60k loading at a pinned revision, the committed splits, and the prompt every model sees."""

import json
from pathlib import Path
from typing import Any, Literal, TypedDict

from datasets import load_dataset

DATASET = "Salesforce/xlam-function-calling-60k"
DATASET_REVISION = "26d14ebfe18b1f7b524bd39b404b50af5dc97866"
DEFAULT_SPLITS_PATH = Path("data/splits.json")

SYSTEM_PROMPT = """You are a function-calling assistant. You have access to these tools:

{tools}

Answer the user's request by calling the tools. Reply with ONLY a JSON list of calls, \
in the form [{{"name": "<tool name>", "arguments": {{<argument name>: <value>, ...}}}}]. \
No explanations, no markdown."""

Split = Literal["train", "validation", "test"]


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


def load_all() -> list[Example]:
    """Every row of the dataset at the pinned revision, with tools and answers parsed."""
    ds = load_dataset(DATASET, split="train", revision=DATASET_REVISION)
    return [parse_row(row) for row in ds]


def load_split(split: Split, splits_path: Path = DEFAULT_SPLITS_PATH) -> list[Example]:
    """The examples of one committed split (see `fc_distill.splits` for how it was built)."""
    if not splits_path.exists():
        raise FileNotFoundError(
            f"{splits_path} not found; it is built by notebooks/01_dataset.ipynb "
            "(fc_distill.splits.write_splits)"
        )
    ids = set(json.loads(splits_path.read_text())[split])
    return [ex for ex in load_all() if ex["id"] in ids]


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
