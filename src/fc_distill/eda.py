"""Statistics and plots behind notebooks/01_dataset.ipynb."""

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypedDict

from fc_distill.data import Example, to_messages
from fc_distill.quality import base_type, query_template

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class ExampleStats(TypedDict):
    id: int
    calls: int
    distinct_functions_called: int
    tools_offered: int
    arguments: int
    query_chars: int


def example_stats(examples: Iterable[Example]) -> list[ExampleStats]:
    """One row of shape statistics per example."""
    return [
        {
            "id": ex["id"],
            "calls": len(ex["answers"]),
            "distinct_functions_called": len({c["name"] for c in ex["answers"]}),
            "tools_offered": len(ex["tools"]),
            "arguments": sum(len(c["arguments"]) for c in ex["answers"]),
            "query_chars": len(ex["query"]),
        }
        for ex in examples
    ]


def call_pattern(ex: Example) -> str:
    """'single', 'parallel' (one function repeated), 'multiple' (all different) or 'mixed'."""
    names = [c["name"] for c in ex["answers"]]
    if len(names) == 1:
        return "single"
    if len(set(names)) == 1:
        return "parallel"
    if len(set(names)) == len(names):
        return "multiple"
    return "mixed"


def declared_type_counts(examples: Iterable[Example]) -> Counter[str]:
    """How often each base parameter type is declared across all offered tools."""
    return Counter(
        base_type(str(spec.get("type", "")))
        for ex in examples
        for tool in ex["tools"]
        for spec in (tool.get("parameters") or {}).values()
    )


def called_function_counts(examples: Iterable[Example]) -> Counter[str]:
    return Counter(call["name"] for ex in examples for call in ex["answers"])


def token_lengths(
    examples: Sequence[Example], model: str = "Qwen/Qwen3.5-0.8B"
) -> tuple[list[int], list[int]]:
    """(prompt tokens, answer tokens) per example, with the chat template the models see."""
    import json

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    prompts = [
        len(
            tok.apply_chat_template(
                to_messages(ex),
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
            )["input_ids"]
        )
        for ex in examples
    ]
    answers = [len(tok(json.dumps(ex["answers"]))["input_ids"]) for ex in examples]
    return prompts, answers


def percentiles(
    values: Sequence[float], qs: Sequence[float] = (0.5, 0.9, 0.99)
) -> dict[str, float]:
    """Nearest-rank percentiles (always an observed value) plus the max."""
    ordered = sorted(values)
    out = {f"p{round(q * 100)}": ordered[max(math.ceil(q * len(ordered)) - 1, 0)] for q in qs}
    return out | {"max": ordered[-1]}


def ci_half_width(n: int, p: float = 0.5) -> float:
    """Half-width, in points, of the normal-approximation 95% interval for an accuracy on n rows."""
    return 100 * 1.96 * math.sqrt(p * (1 - p) / n)


class TrainOverlap(TypedDict):
    rows: int
    calls_function_unseen_in_train: int
    query_template_in_train: int


def train_overlap(train: Iterable[Example], held_out: Sequence[Example]) -> TrainOverlap:
    """How much of a held-out split train has already seen: rows calling a function no train
    answer calls, and rows whose query template (`quality.query_template`) occurs in train."""
    train = list(train)
    called = {c["name"] for ex in train for c in ex["answers"]}
    templates = {query_template(ex["query"]) for ex in train}
    return {
        "rows": len(held_out),
        "calls_function_unseen_in_train": sum(
            any(c["name"] not in called for c in ex["answers"]) for ex in held_out
        ),
        "query_template_in_train": sum(query_template(ex["query"]) in templates for ex in held_out),
    }


def plot_counts(counts: Mapping[Any, int], title: str, xlabel: str) -> "Figure":
    """Bar chart of a small distribution: integer keys in order, other keys by frequency."""
    import matplotlib.pyplot as plt

    items = list(counts.items())
    if all(isinstance(k, int) for k, _ in items):
        items.sort()
    else:
        items.sort(key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=90)
    ax.bar([str(k) for k, _ in items], [v for _, v in items], color="#0f6e64")
    ax.set(title=title, xlabel=xlabel, ylabel="rows")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_hist(values: Sequence[float], title: str, xlabel: str, bins: int = 50) -> "Figure":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=90)
    ax.hist(values, bins=bins, color="#0f6e64")
    ax.set(title=title, xlabel=xlabel, ylabel="rows")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_rank_frequency(counts: Counter[str], title: str) -> "Figure":
    """Log-log rank/frequency plot: shows how long the tail of rarely used tools is."""
    import matplotlib.pyplot as plt

    freqs = sorted(counts.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=90)
    ax.loglog(range(1, len(freqs) + 1), freqs, color="#0f6e64")
    ax.set(title=title, xlabel="rank", ylabel="calls")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
