# Test label audit

Issue #8. Rubric: [`docs/label-audit.md`](../docs/label-audit.md). Per-row labels:
[`data/audit/test_labels.csv`](../data/audit/test_labels.csv). `test-clean` ids: the `test-clean`
key in `data/splits.json` (`fc-eval --split test-clean`).

## Ceiling

**1,626 of the 2,000 test rows (81.3%) have a sound label.** That's the highest strict exact match a
model that's always right could score on raw test. The other 374 rows have a gold label that is
ambiguous (299) or wrong (75), and `test-clean` leaves them out.

| | Rows |
|---|---:|
| Test | 2,000 |
| Every model right (assumed sound) | 751 |
| Audited (at least one model wrong) | 1,249 |
| Model wrong: label sound | 875 |
| Label ambiguous | 299 |
| Label wrong | 75 |
| **`test-clean`** | **1,626** |

## Scores

Strict exact match, greedy decoding, thinking off (`fc-eval --split test`).

| Model | Test | `test-clean` |
|---|---:|---:|
| Qwen3.5-0.8B (student) | 46.4 | 55.1 |
| Qwen3.5-2B (teacher) | 65.3 | 76.9 |
| Qwen3.5-4B | 69.1 | 82.2 |

Removing the unsound rows raises every model by 9 to 13 points, and the 2B-to-4B gap grows from 3.8 to
5.3 points. The wrong-argument errors that stayed flat from 0.8B to 4B in scouting ([ADR 0002](../docs/adr/0002-test-labels-never-edited.md)) were mostly label noise: 4B gets 255
of the 299 ambiguous rows wrong, and every one of the 75 wrong ones.

## What makes a label unsound

| Category | Rows | Example |
|---|---:|---|
| Unclear units or format | 188 | country `UK` vs `GB`; `.jpg` vs `jpg`; casing when the spec example is lowercase; `9:30` as `930` or `9.5` |
| Underspecified query | 103 | "last week" with no reference date; coordinates of a landmark; a search string that could be worded several ways |
| Wrong value | 60 | placeholders (`cursor_value`, `<is_id>`, `YOUR_API_KEY`); wrong timestamps; `x**3` where the query and spec say `x^3`; missing `-` for descending order |
| Spec violation | 8 | a string where `int`/`bool` is declared; a dict where a list is declared |
| Several fitting tools | 8 | two offered tools that both validate an email |
| Code-valued argument | 6 | `"math.pi / 2"` where a float is declared (set automatically) |
| Wrong calls | 1 | a required argument given in the query left out |

Sound labels that a strict scorer still marks wrong for an equivalent call:

| Category | Rows | |
|---|---:|---|
| Default made explicit | 122 | gold passes an optional argument at its default, or the model does; 71 set automatically. #15 adds a lenient score for these |
| Convention not stated in spec | 6 | interest rates passed as decimals (`0.04` for 4%), consistent per tool across the dataset |

## Method

- The audit covers every row that **any** of 0.8B, 2B and 4B gets wrong, not only the teacher's
  errors as #8 first proposed. Auditing only 2B and 4B errors would have kept 17 unsound rows that only
  the student misses. That would have lowered the student's `test-clean` score by 0.6 points and
  widened the teacher-student gap by 0.9.
- Automatic verdicts are used only where the outcome doesn't depend on which model erred:
  code-valued gold arguments (label wrong), and rows where every wrong prediction differs from gold
  only by optional defaults (sound label).
- The remaining 1,172 rows were labelled by hand against the rubric. Model identities were hidden
  and predictions were shown in a fixed shuffled order.
- Repeated cases get the same verdict: a format choice the spec leaves open (casing, prefixes,
  separators, off-by-one page numbering when the spec is silent) is ambiguous every time it comes up.

## Review

The maintainer relabels 30 random audited rows blind (`uv run fc-audit review` writes
[the sheet](label-audit-review.md) and `data/audit/review.csv`, seed 0). More than 3 disagreements
that change soundness means the rubric is too loose: tighten it and relabel.

**Status: pending.** Once the review is filled in, `uv run fc-audit build` reports the disagreements.
Record the outcome here.

## Caveats

- One labeller. The 30-row review is the only check on consistency.
- Rows that every model gets right are assumed sound. An ambiguous label that all three models happen
  to match stays in `test-clean`, so 81.3% is an upper bound on the sound share.
- "Ambiguous" is applied generously to format choices the spec leaves open. A lenient scorer (#15)
  that normalises casing and `www.`/`https://` prefixes could recover some of these rows.
