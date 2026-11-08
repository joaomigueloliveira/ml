# Function-calling distillation

A small Qwen3.5 student learns single-turn tool calling on xLAM-60k from a larger Qwen3.5 teacher.

## Models

**Student**:
Qwen3.5-0.8B, the model being trained.

**Teacher**:
Qwen3.5-2B, the model whose outputs or logits the student learns from.

## Experiments

**Method**:
One way of training the student, compared with the others per GPU-hour: A, SFT on gold labels; B,
SFT on teacher outputs; C, logit distillation; D, on-policy distillation; E, GRPO.
_Avoid_: arm

**Label-free method**:
A method whose training never sees gold labels, only queries and the teacher: B (unfiltered), C, D.
The main question (ADR 0007).

**With-labels method**:
A method whose training uses gold labels, as targets, as a filter or in a reward: A, filtered B, E.
The reference for what labels buy.

**Variant**:
A named change within a method, e.g. method B filtered to the teacher outputs that match gold.

**Run**:
One training job of a method or variant, with its own checkpoints and its own W&B run.

**Baseline**:
An untrained model used as a reference point: the student, the teacher, the 4B.

## Data

**Row**:
One xLAM example: a query, the offered tools and the gold answer.
_Avoid_: sample, example (when a split row is meant)

**Gold answer**:
The list of calls xLAM publishes as correct for a row.
_Avoid_: label (alone), ground truth

**Offered tools**:
The tools a row's prompt lists. Some are distractors that the gold answer doesn't call.

**Query template**:
The same query sentence with different values, as xLAM's generator produced many per tool.

**Explicit default**:
An optional argument that a call passes at its declared default or as `null`.

**Made-up value**:
A gold argument value that the query doesn't give and the gold answer invents.

## Splits

**Train / validation / test**:
The three splits of xLAM-60k. Validation chooses checkpoints and settings; test only produces
official numbers.

**test-clean**:
Test minus the rows whose gold answer the label audit marks ambiguous or wrong.
_Avoid_: clean test, filtered test

**Train filter**:
A documented rule that removes rows from train. It never rewrites a gold answer.

## Label audit

**Audited row**:
A test row that at least one of the 0.8B, 2B and 4B baselines gets wrong.

**Sound label**:
A gold answer that a careful person, reading only the query and the tool specs, would write.
_Avoid_: model wrong (the old verdict name), correct label

**Ambiguous label**:
A gold answer that is one of several reasonable calls for the query and specs.

**Wrong label**:
A gold answer that contradicts the query or specs, or contains a made-up value.

**Ceiling**:
The share of test rows with a sound label: the best strict exact score an always-right model could
reach on test.

## Scoring

**Strict exact**:
A prediction matches when its multiset of calls equals the gold answer's. The headline metric.
_Avoid_: accuracy (alone)

**Lenient exact**:
Strict exact, except that explicit defaults on either side are ignored.

**Quirk-learning**:
A model gaining on raw test by reproducing xLAM's unsound labels, seen as a gain on test that
`test-clean` doesn't show.

**Gap recovered**:
How much of the distance from the raw student to the teacher a trained student closes, in strict
exact on the same eval: (student − raw student) / (teacher − raw student).

**Share of A's gain**:
How much of method A's improvement a label-free method reaches, in strict exact on the same eval:
(method − raw student) / (A − raw student).
