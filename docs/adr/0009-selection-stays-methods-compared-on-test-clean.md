# 0009. Checkpoint selection stays as is; methods are compared on test-clean, not picked on validation

Date: 2026-11-10 · Status: accepted (#73)

## Context

ADR 0006 picks each run's checkpoint as the best on validation (strict exact against gold, an exact
tie going to the earlier step). The external review of 2026-11-09 showed what that pick can and
can't do:

- **Validation is flat within noise after the first checkpoints.** 1,000 rows detect about 2.4
  points at 80% power. Within a run, most checkpoints are not distinguishable from the pick by a
  paired test. Across methods, B and D differ by 0.2 points (708 vs 706).
- **The pick moves on resampled validation.** In a bootstrap, B picks step 1500 in 67% of
  resamples, and D's pick spreads over four checkpoints (31% for step 3000). The pick's own
  validation score is inflated by 0.4–0.8 points.
- **Lenient exact would move three of five picks**, each within noise:

  | | A | B | Filtered B | D | B then D |
  |---|---|---|---|---|---|
  | Strict pick | 3411 | 1500 | 1000 | 3000 | 400 |
  | Lenient pick | 3411 | 2500 | 2000 | 1500 | 400 |

- **Validation can't see differences that test does.** It ties D at steps 500 and 3000, while
  test-clean separates step 500 from B (p = 0.006). And validation stops B before its BFCL score
  peaks (79.1 at the pick, 81.1 at the end of its epoch, #70).

The pick's *test* numbers are not biased by any of this: test is scored once, on the checkpoint
picked, so its number is fair for that checkpoint. What a noisy pick costs is the choice of
checkpoint, and any sentence that rests on it.

## Decision

- **The rule within a run stays ADR 0006's**, unchanged, for finished and future runs alike. No run
  is re-picked and no new test run is needed.
- **Methods are not picked against each other on validation.** Map #2's Phase 4 plan, to crown the
  best label-free method on validation, is replaced: every label-free method is reported on
  test-clean with paired tests (Holm). The README names a winner only where test-clean separates it
  significantly. Where nothing does, it says so ("B and D are not distinguishable"), with GPU-hours
  beside the scores. Of tied methods, the cheaper one is published.
- **A claim that depends on the pick names the checkpoint** (as "D beats B on BFCL" does since #70),
  and the README says once that the pick within a run is noisy.
- **Selection stays on strict exact, and validation is not audited.** The lenient table above shows
  the pick is noise-driven, not label-driven. An audit would remove about a fifth of the rows and
  make validation less sensitive, not more.

## Considered options

- **The cheapest checkpoint within noise of the best.** Rejected: it would move D to step 500, which
  trails B on test-clean (p = 0.006). It rewards validation's blind spot by choosing under-trained
  checkpoints, and it moves A and B to checkpoints never scored on test.
- **The last checkpoint, with no selection.** Rejected: honest about a fixed budget and free of
  optimism, but B then D's later checkpoints are all significantly worse, so the rule would need an
  exception at once.
- **Select on lenient exact.** Rejected: three picks move within noise, and three new test runs
  would follow from a rule chosen after seeing what it picks.
- **Audit validation (about 600 rows).** Rejected for the reason above; ADR 0005's trigger now
  produces a finding instead (its 2026-11-10 addendum).
- **Pick the headline on test-clean.** Rejected: it would break ADR 0001. Reporting every method
  and wording the comparison as it stands needs no pick.

Every alternative was seen applied to the finished runs before this decision, so keeping the rule
fixed in advance (ADR 0006) was the only option not chosen with the outcome in view.

## Consequences

- The README's method comparisons rest on test-clean, test and BFCL, never on a pick's validation
  score. Its wording is corrected in #72.
- Selection frequency stays an analysis on the review's branch; `fc-report` gains no bootstrap.
- Phase 4's headline is a statement about which methods test-clean separates, with cost, rather than
  a single winner by number.
