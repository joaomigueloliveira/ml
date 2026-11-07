# 0005. Train and validation keep their label noise

Date: 2026-11-06 · Status: accepted

## Context

The test label audit (#8, [results](../../results/label-audit.md)) found that at least 18.75% of test
rows have an unsound gold label: 287 ambiguous and 88 wrong out of 2,000. The main causes are a format the
spec leaves open (`UK` vs `GB`, casing), a value the query doesn't fix, and a value gold makes up
(`zuid: "jkl012"` for "agent D"). Test is a random sample of xLAM, so train very likely has a similar
share, roughly 10,000 of its 56,392 rows. The train filters so far (ADR 0002, #6) remove only 213
code-valued and 20 contradicting rows.

Cleaning train the way test was cleaned would take hand-labelling tens of thousands of rows, and most
of the noise needs judgement (is `UK` acceptable?) that no rule captures. Rewriting labels would also
break ADR 0002.

## Decision

- **Train keeps its labels.** Rows leave train only through documented filters that detect a problem
  without judgement. Candidates for the made-up-value class are placeholders (`YOUR_API_KEY`,
  `cursor_value`, `<is_id>`) and string values that appear in neither the query nor the spec. #17
  measures them on train first, and a filter is added only if it removes rows the audit would call
  wrong.
- **Validation is not audited for now.** Label noise lowers every checkpoint's validation score by
  about the same amount, so it mostly shifts numbers rather than reordering checkpoints. The audit is
  triggered if a method's gain on raw test is much larger than its gain on `test-clean`: that means it
  learned xLAM's quirks, and quirk-learning could also steer checkpoint selection on validation.
- **`test-clean` is the target.** The #2 success criterion (≥ 50% of the gap recovered) is judged on
  `test-clean`. A model trained on gold learns xLAM's quirks, which raises raw test and not
  `test-clean`, because those rows aren't in it.
- **Teacher outputs are filtered with the lenient score, not strict.** If the filtered variant of SFT
  on teacher outputs keeps only outputs that match gold, strict exact match would drop the teacher's
  defensible answers (`GB`, a default left out) and keep the ones that copy gold's quirks. The filter
  uses lenient exact (#15).

## Considered options

- **Hand-audit train.** Rejected: tens of thousands of judgements, for a gain that `test-clean` already
  makes visible.
- **Train only on rows every baseline model gets right.** Rejected: it keeps the easy rows and removes
  the hard but sound ones, so train would no longer match the test distribution.
- **Audit validation now.** Deferred under the trigger above. About 600 rows by the same method.

## Consequences

- Every result is reported on test and `test-clean` (ADR 0002). The difference between a method's gain
  on the two is reported too, as the quirk-learning signal.
- SFT on gold labels starts with an advantage on raw test that it doesn't have on `test-clean`. The
  write-up compares methods on `test-clean`.
- #15 is a prerequisite for the filtered teacher-output arm.

## Addendum (2026-11-07): no made-up-value filter (#17)

#17 measured the two candidate filters named above against the test label audit. The prototype
was a quick, untuned measurement and was not committed:

- **placeholder:** a gold string value matches a placeholder pattern (`YOUR_API_KEY`,
  `api_key_here`, `cursor_value`, `<name>`, `example.com`, `xxx`, `your_…`);
- **unseen string:** a gold string value of at least 3 characters, compared case-insensitively,
  appears in neither the query nor the tool spec.

Rows that were never audited (every model right) count as sound, as in `test-clean`.

| Filter | Test rows flagged | Wrong value | Any unsound | Recall on the 73 wrong-value rows | Train rows removed |
|---|---:|---:|---:|---:|---:|
| placeholder | 60 | 11 (18%) | 21 (35%) | 15% | 1,400 of 54,561 (2.6%) |
| unseen string | 273 | 46 (17%) | 159 (58%) | 63% | 7,476 of 54,561 (13.7%) |

**Decision: neither filter is added.**
- Neither one mostly removes rows the audit calls wrong. About 5 in 6 of the rows each one flags are
  not wrong values.
- Most of the unseen-string filter's unsound hits are ambiguous labels, which this ADR keeps on
  purpose.
- Its sound hits are correct rewrites that the query doesn't spell out, e.g. "January 1, 2023" →
  `"2023-01-01"`.
- Tuning the normalisation could raise precision a little, but not plausibly to "mostly wrong".

Why this costs little:
- Only SFT arm A trains on gold labels. Arm B and the distillation arms learn from teacher
  outputs, so gold's made-up values don't reach them.
- If noisy gold makes arm A lose to the teacher-trained arms, that is a finding for the write-up.

**Trigger to revisit:** if arm A's gain on raw test is much larger than its gain on `test-clean`
(the quirk-learning signal above), measure a tuned filter again before further gold-label runs.

Settled while scoping #17, so no further study is needed:
- **Float tolerance:** a 1e-4 relative tolerance would turn 1 to 2 of the 2,000 test rows right per
  model, so the scorer stays exact.
- **Gold call order:** the scorer already ignores it.
