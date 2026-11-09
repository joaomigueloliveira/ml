# 0007. Label-free distillation is the main question; SFT on labels is the reference

Date: 2026-11-08 · Status: accepted · Amended 2026-11-08 by ADR 0008 (D starts from the raw student); 2026-11-09 (filtered B before C)

## Context

The first two Phase 2 runs (#42, #46) changed what the comparison can say:

| Validation, strict exact | Score | GPU-h |
|---|---:|---:|
| raw 0.8B (student) | 46.6 | 0 |
| raw 2B (teacher) | 68.0 | 0 |
| B, SFT on teacher outputs, best checkpoint (1500) | 70.8 | 3.12 |
| A, SFT on gold labels, best checkpoint (3411) | 84.7 | 3.84 |

A beats its own teacher by 17 points, so "gap recovered" towards the 2B can't separate A from
anything. B lands near its teacher, as sequence-level distillation students usually do: the teacher
is right on 64.9% of train rows (#44). Comparing A and B head to head mostly measures label quality.

In practice distillation is used when there are no labels: there are queries, and a model better
than the student. That is the question worth answering here.

## Decision

**Two regimes.** Every method is labelled by whether its training uses gold labels.

| Regime | Methods | Role |
|---|---|---|
| Label-free | B (unfiltered), D, C | The main question and the README's headline claim |
| With labels | A, filtered B, E | The reference: what labels buy |

- **Order:** B → D → C → filtered B → E. The label-free methods run first; the with-labels runs follow.
  Amended 2026-11-09: filtered B ran before C (#60), since it needed no new code and C's design
  wasn't settled; the results table, not the run order, tells the story.
- **D starts from the raw student**, like A, B and C (ADR 0008; first decided as "from B's
  selected checkpoint", which replaced #35's "the better of A and B"). Starting from A would bring
  labels into a label-free method. B then D is run as a pipeline, not as method D.
- **C, logit distillation, is a main method**, run after D. It is the off-policy, dense-signal point
  between B and D, which tells whether D's gain comes from the teacher's probabilities or from
  training on the student's own outputs.
- **Filtered B always runs** (amends ADR 0006, where it ran only if B lost to A). It is an analysis
  run: filtered B − B is what the teacher's mistakes cost, A − filtered B is what is left (fewer rows,
  the teacher's style).
- **E, GRPO, counts as with labels**: its reward is checked against gold. It answers whether RL adds
  anything on top of SFT. Its start point and reward are settled at Phase 3.
- **A label-free validity check** (every called function exists in the offered tools, required
  arguments present) is reported as a count of the teacher outputs it would drop, not trained on.

**Headline numbers.** For the best label-free method, two numbers on `test-clean`:
- **gap recovered**, (method − raw student) / (teacher − raw student). The ≥ 50% success criterion
  (#35) stays unchanged;
- **share of A's gain**, (method − raw student) / (A − raw student): how much of what labels buy is
  reached without them.

Every result is shown next to the baselines and A.

## Considered options

- **Keep A vs B as the core comparison.** Rejected: it measures label quality against a weak
  teacher, not distillation.
- **D from the better of A and B.** Rejected, see above.
- **Keep C as a stretch goal.** Rejected: without it the README covers two of the three classic kinds
  of distillation, and C separates D's two ingredients.
- **A label-free filtered B variant trained.** Deferred to the count: if the validity check drops
  only a handful of rows, a run would show nothing.

## Consequences

- "Label-free" is about training. Checkpoints are still chosen on labelled validation and scored on
  labelled test, as in practice, where a small labelled eval set is the norm.
- The teacher is a 2B. With a stronger teacher the label-free numbers would likely be higher; the
  results are about this pair (ADR 0004).
- C needs a teacher pass that stores top-k probabilities and a KL loss in `fc-sft`.
