# Eval-risk review

A third reviewer for any PR that changes data, splits, scoring or reported numbers. It runs beside
the standards and spec review and asks a different question: **could this change make a reported
number wrong, or make a later comparison unfair?** A standards or spec review reads the diff; this one
tests claims against the data and saved model outputs.

## Running it

Spawn one independent subagent, read-only, with this file as its brief, plus:

- the diff (`git diff <base>...HEAD`) and the issues it implements
- the run directories it may test against (`runs/eval-*/predictions.jsonl`, `summary.json`)

It runs every Python command as `uv run --no-sync …` and uses no GPU.

## What to probe

Work through every area that the diff touches, plus anything the evidence leads to:

1. **Scoring.** Parsing (fences, trailing text, a dict instead of a list, empty answers), value
   equality (`canon`: int vs float, bool vs number, large integers, nested values), lenient defaults
   (`is_optional`, `is_default`), BFCL's acceptable-value lists and pairing.
2. **Statistics.** Interval method and its behaviour at small n and near 0 or 1; McNemar; whether a
   claim's test matches its wording in #2 (for example, "recovers half the gap" vs "beats the
   baseline").
3. **Splits and leakage.** Anything train shares with validation or test: query text, templates,
   identical gold answers, tools. Measure the effect on scores when there's overlap.
4. **Label quality.** Whether `test-clean` depends on which models were audited or on run-to-run
   noise; train label noise (ADR 0005).
5. **Comparability.** Prompt and chat template matching between eval and training; `max_model_len`
   and `max_tokens` against real lengths; decoding determinism; provenance enough to reproduce a run.
6. **What the headline number measures.** For each gap or target, break down where it comes from
   (output format vs content, slices, rows seen in train), so a method can't meet it by an
   unintended route.

## Report

A prioritised list (**High / Medium / Low**). For each finding:

- what's wrong or missing
- evidence: `file:line`, counts from real data, or a snippet that reproduces it
- the impact on later results
- a suggested fix

Mark each finding as a **confirmed** bug or a **risk**, and end with what was checked and found
sound. Keep it under 600 words. The implementing agent then verifies every High finding itself
before acting on it.
