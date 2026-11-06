"""Per-slice scores for xLAM splits (issue #16), so a student/teacher gap can be explained.

Three dimensions, each a function of the gold row only:

- call pattern: single / multiple / parallel; mixed (both) is merged into multiple.
- distractors: offered tools the gold answer doesn't call, bucketed 0 / 1-2 / 3+.
- argument source: every string and number in the gold answer appears in the query
  (`from_query`), or at least one doesn't (`not_in_query`: a date rewritten, "2%" as 0.02,
  a ticker for a company name, a value the query never gives).

ADR 0001: a slice is reported on its own only with at least 300 rows. Every bucket here shows its
row count and 95% intervals, and buckets under the minimum are flagged `too_small`.
"""

import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from fc_distill.data import Example
from fc_distill.eda import call_pattern as _raw_call_pattern
from fc_distill.stats import MIN_ROWS, metric_block

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def call_pattern(ex: Example) -> str:
    """single / multiple / parallel. Mixed (repeated and different functions) is too rare to report
    on its own (49 test rows) and counts as multiple."""
    pattern = _raw_call_pattern(ex)
    return "multiple" if pattern == "mixed" else pattern


def distractors(ex: Example) -> str:
    """Offered tools that the gold answer doesn't call: '0', '1-2' or '3+'."""
    called = {c["name"] for c in ex["answers"]}
    n = len({t["name"] for t in ex["tools"]} - called)
    return "0" if n == 0 else "1-2" if n <= 2 else "3+"


def _scalars(value: Any) -> Iterator[Any]:
    if isinstance(value, list | tuple):
        for v in value:
            yield from _scalars(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _scalars(v)
    else:
        yield value


def query_numbers(query: str) -> set[float]:
    """Every number written in the query with digits ('1,000' is 1000)."""
    out = set()
    for token in _NUMBER.findall(query):
        try:
            out.add(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


def value_in_query(value: Any, query: str) -> bool:
    """Is every string and number in `value` written in the query? Strings match as a
    case-insensitive substring (whitespace collapsed); numbers match a number in the query. Booleans
    and null say nothing about the query and always count as in it."""
    text = " ".join(query.lower().split())
    numbers = query_numbers(query)
    for s in _scalars(value):
        if isinstance(s, bool) or s is None:
            continue
        if isinstance(s, str) and " ".join(s.lower().split()) not in text:
            return False
        if isinstance(s, int | float) and float(s) not in numbers:
            return False
    return True


def argument_source(ex: Example) -> str:
    """'from_query' if every gold argument value is written in the query, else 'not_in_query'.
    A row with no arguments counts as from_query."""
    ok = all(
        value_in_query(v, ex["query"]) for call in ex["answers"] for v in call["arguments"].values()
    )
    return "from_query" if ok else "not_in_query"


DIMENSIONS: dict[str, tuple[Callable[[Example], str], tuple[str, ...]]] = {
    "call_pattern": (call_pattern, ("single", "multiple", "parallel")),
    "distractors": (distractors, ("0", "1-2", "3+")),
    "argument_source": (argument_source, ("from_query", "not_in_query")),
}


def slice_scores(
    examples: Sequence[Example],
    scores: Sequence[Mapping[str, Any]],
    metrics: Iterable[str],
    min_rows: int = MIN_ROWS,
) -> dict[str, dict[str, dict[str, Any]]]:
    """{dimension: {bucket: {n, too_small, <metric>, <metric>_ci, ...}}} for every dimension, with
    every bucket listed (an empty one as n = 0)."""
    metrics = list(metrics)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for dim, (assign, buckets) in DIMENSIONS.items():
        groups: dict[str, list[Mapping[str, Any]]] = {b: [] for b in buckets}
        for ex, s in zip(examples, scores, strict=True):
            groups[assign(ex)].append(s)
        out[dim] = {
            bucket: {"too_small": len(rows) < min_rows} | metric_block(rows, metrics)
            for bucket, rows in groups.items()
        }
    return out
