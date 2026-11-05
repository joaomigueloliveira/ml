# 0002. Test labels are never edited; `test-clean` is a subset

Date: 2026-11-05 · Status: accepted

## Context

The scouting run showed wrong-argument errors staying flat from 0.8B to 4B (229 / 256 / 224), which
points at ambiguous or wrong gold labels (is "3% interest" `3` or `0.03`?). Correcting labels by our own
judgement would bend the benchmark toward what our models output and make our numbers incomparable to
anyone else's.

## Decision

- The test split keeps xLAM's labels exactly as published.
- A label audit marks each test row that a strong model gets wrong as *model wrong*, *label ambiguous*
  or *label wrong*. **`test-clean`** is test minus the ambiguous and wrong rows.
- Every result is reported on both test and `test-clean`.
- Train is cleaned only by documented filters that remove rows (unparseable, inconsistent, duplicated,
  leaking into validation/test, or holding Python code where a value is declared). No train label is
  rewritten either.

## Consequences

- Our raw-test numbers are comparable to other xLAM results; `test-clean` shows the realistic ceiling.
- The audit's rubric and per-row labels are committed so anyone can check our judgement.
