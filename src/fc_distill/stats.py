"""Intervals and paired tests for every reported number (#2: 95% CIs on every number, paired McNemar
for every "A beats B"; ADR 0001: a slice is reported on its own only with at least 300 rows)."""

import math
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Any

MIN_ROWS = 300
Z = 1.959964  # two-sided 95%


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson score 95% interval for k successes in n trials. Unlike the normal approximation it
    stays inside [0, 1] and holds up for small n and rates near 0 or 1."""
    if n == 0:
        raise ValueError("no rows")
    p = k / n
    centre = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


def interval(values: Sequence[float]) -> tuple[float, float, float]:
    """(mean, low, high). 0/1 values get a Wilson interval; other values in [0, 1] (arg_f1) a
    normal-approximation interval clipped to [0, 1]."""
    n = len(values)
    if n == 0:
        raise ValueError("no rows")
    mean = sum(values) / n
    if all(v in (0.0, 1.0) for v in values):
        low, high = wilson(round(sum(values)), n)
    else:
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
        low, high = max(0.0, mean - Z * sd / math.sqrt(n)), min(1.0, mean + Z * sd / math.sqrt(n))
    return mean, low, high


def metric_block(rows: Sequence[Mapping[str, Any]], metrics: Iterable[str]) -> dict[str, Any]:
    """{n, <metric>: mean, <metric>_ci: [low, high], ...}, rounded to 4 decimals. An empty set of
    rows gives None for every metric."""
    out: dict[str, Any] = {"n": len(rows)}
    for m in metrics:
        if rows:
            mean, low, high = interval([float(r[m]) for r in rows])
            out[m], out[f"{m}_ci"] = round(mean, 4), [round(low, 4), round(high, 4)]
        else:
            out[m] = out[f"{m}_ci"] = None
    return out


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
        tail = sum(Fraction(math.comb(n, k), 2**n) for k in range(min(only_a, only_b) + 1))
        p = float(min(Fraction(1), 2 * tail))
    return {"n": len(a), "only_a_right": only_a, "only_b_right": only_b, "p": p}
