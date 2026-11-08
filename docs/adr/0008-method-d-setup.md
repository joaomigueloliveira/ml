# 0008. Method D: on-policy distillation from the raw student, by GPU-hour budget

Date: 2026-11-08 · Status: accepted

## Context

Method D is the second label-free method (ADR 0007). The student samples its own answers, and the
teacher grades every token: the loss is the reverse KL, KL(student ‖ teacher), on the student's
samples ([Thinking Machines, 2025](https://thinkingmachines.ai/blog/on-policy-distillation/);
[GKD](https://arxiv.org/abs/2306.13649)).

Phase 2 left B at 83.4 on `test-clean`, above its 2B teacher, but below it on BFCL (79.1 vs 82.7),
where it gets argument values wrong (`results/methods.md`). TRL 1.17, which the project pins, ships
`trl.DistillationTrainer`: always on-policy, generalized JSD with `beta=1.0` for reverse KL, LoRA
through PEFT, student sampling in vLLM colocated with training, and the loss computed chunk by
chunk so neither model's full logits are ever held at once.

## Decision

| Setting | Value |
|---|---|
| Trainer | TRL's `DistillationTrainer`, in a new `fc-opd` CLI (#53) |
| Loss | reverse KL (`beta=1.0`), exact over the vocabulary at every answer position |
| Start | the raw 0.8B with a fresh LoRA, r16/α32 all-linear (ADR 0006) |
| Teacher | Qwen3.5-2B, bf16, in the same process |
| Data | train-split prompts, never gold; a seed-0 subset, no repeats; one sample per prompt |
| Sampling | temperature 1.0, thinking off, the eval prompt, at most 1,024 tokens |
| Optimiser | lr 2e-4, cosine, 3% warmup, 16 rows per step (ADR 0006, untuned) |
| Length | a budget of 3.8 GPU-hours, the cost of one SFT epoch; rows follow from measured throughput |
| Sampling engine | vLLM colocated with sleep mode if it fits next to training and the teacher; HF `generate` otherwise |

- **GPU-hours start at 0** and every step counts the student's sampling, the teacher's forward pass
  and the update, so the teacher's cost is inside D's curve.
- **Selection** follows ADR 0006: about 6 checkpoints scored on validation with strict exact, the
  best wins, ties to the earlier, then test, `test-clean` and BFCL for that one only. One seed.
- **B→D**, continuing B's adapter with D's loss for the same budget, is a **pipeline**, reported as
  "B then D", not as method D. It becomes the next run if D doesn't beat B on validation; otherwise
  it stays a stretch goal. This amends ADR 0007, where D started from B.

## Considered options

- **Start from B, as Thinking Machines start from an SFT checkpoint.** Rejected as method D: A, B
  and C start from the raw student, and from B the curve can't show what D does on its own or per
  GPU-hour. Kept as the B→D pipeline.
- **Thinking Machines' sampled-token loss**, one teacher log-probability per token. Rejected: a
  hand-written sampling loop and update, where TRL computes the exact per-position KL in chunks.
- **Length in rows or epochs.** Rejected: a D step costs far more than an SFT step (sampling and a
  teacher forward pass), so equal rows would not be an equal cost. A budget makes the curves meet on
  the GPU-hour axis.
- **Half the learning rate.** Only argued for when refining B's adapter; from a fresh adapter the
  shared config holds.

## Consequences

- D's row count is known only after the smoke run (#53); the run itself is #54.
- The learning rate is untuned for a loss it wasn't chosen for. The README lists it.
- Temperature-1.0 samples from the raw student break the JSON more often than greedy decoding
  (parse_ok 84.5% on validation, greedy); those tokens are still graded by the teacher.
