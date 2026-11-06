# 0001. Train / validation / test, with test only for official numbers

Date: 2026-11-05 · Status: accepted · Amended 2026-11-06 (#25)

## Context

Every training method produces checkpoints and has settings (learning rate, steps, LoRA rank). Picking
them by looking at the same rows we report on tunes the method to those rows and inflates the final
number. xLAM-60k has far more rows than we need for evaluation.

## Decision

Three random splits of xLAM-60k: **validation** (1,000 rows), **test** (2,000 rows), **train** (the rest).

- Rows are grouped by normalized query text before splitting, so the same query never sits in two
  splits. Query *templates* (same sentence, other values) and tools are shared with train by how
  xLAM was generated (13–17% of held-out queries); that is kept, so xLAM test measures seen tools
  with new values and unseen-tool generalization is measured on BFCL.
- A reworded query can keep the values, and then its gold answer equals a held-out one. Train rows
  whose gold answer equals a validation or test gold answer (as strict exact compares calls) are
  removed (#25), so training never teaches a held-out answer itself. A held-out answer that is one
  call inside a larger train answer is kept: that train row is a different, harder task.

- Checkpoint selection, early stopping and setting comparisons use **validation** only.
- **Test** is run for the official baselines and for each method's final checkpoint, nothing else.
- 2,000 test rows put the 95% interval at about ±2.2 points near 50%. Comparisons between models use
  paired McNemar tests on the same rows.

## Consequences

- No result in the write-up was chosen by looking at test.
- A slice (e.g. a call pattern) is reported on its own only with at least 300 test rows (±5.7
  points); smaller ones are merged.
- Test runs are rare, so their cost doesn't matter; validation runs happen often and stay at 1,000 rows.
