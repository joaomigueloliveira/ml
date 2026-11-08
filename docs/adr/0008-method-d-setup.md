# 0008. Method D: on-policy distillation from the raw student, one epoch

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
| Data | all train-split prompts, never gold (a seed-0 subset only under the cap); one sample per prompt |
| Sampling | temperature 1.0, thinking off, the eval prompt, at most 1,024 tokens |
| Optimiser | lr 2e-4, cosine, 3% warmup, 16 rows per step (ADR 0006, untuned) |
| Length | one epoch over the train prompts, a checkpoint every 500 steps, as A and B; capped at about 12 GPU-hours |
| Sampling engine | vLLM colocated with sleep mode if it fits next to training and the teacher; HF `generate` otherwise |

- **GPU-hours start at 0** and every step counts the student's sampling, the teacher's forward pass
  and the update, so the teacher's cost is inside D's curve.
- **The cap.** If the smoke run (#53) puts an epoch above about 12 GPU-hours (one night), D trains
  on a seed-0 subset sized to 12 GPU-hours instead, with the cosine schedule over that subset. The cap
  is fixed before the run, from throughput, never from validation scores.
- **Selection** follows ADR 0006: every checkpoint scored on validation with strict exact after the
  run, the best wins, ties to the earlier, then test, `test-clean` and BFCL for that one only. One
  seed. The checkpoint nearest B's 3.1 GPU-hours gives the matched-cost comparison with B.
- **Monitoring during the run**, watch only: the loss, the parse rate of the student's rollouts
  and, at each checkpoint, a 200-row validation probe from the loaded sampling engine. It never stops
  the run or chooses the checkpoint.
- **B→D**, continuing B's adapter with D's loss for one epoch, is a **pipeline**, reported as
  "B then D", not as method D. It becomes the next run if D doesn't beat B on validation; otherwise
  it stays a stretch goal. This amends ADR 0007, where D started from B.

## Considered options

- **Start from B, as Thinking Machines start from an SFT checkpoint.** Rejected as method D: A, B
  and C start from the raw student, and from B the curve can't show what D does on its own or per
  GPU-hour. Kept as the B→D pipeline.
- **Thinking Machines' sampled-token loss**, one teacher log-probability per token. Rejected: a
  hand-written sampling loop and update, where TRL computes the exact per-position KL in chunks.
- **A GPU-hour budget (3.8, one SFT epoch's cost) instead of an epoch.** Rejected: the cosine
  schedule needs the total step count up front, so a budget means guessing it from throughput; and
  the per-checkpoint GPU-hours already give the matched-cost point. An epoch keeps D on A's and B's
  steps and rows.
- **Early stopping on validation** (stop at 1500 if no better than 1000). Rejected: it would have
  stopped A at 80.8, outside the interval of its final 84.7; adjacent checkpoints rarely differ
  significantly on 1,000 rows; and it needs scoring during training. A patience rule (two
  checkpoints without a new best) would not have stopped A, but still needs in-run scoring.
- **Half the learning rate.** Only argued for when refining B's adapter; from a fresh adapter the
  shared config holds.

## Consequences

- Whether D gets the full epoch or the 12 GPU-hour subset is known after the smoke run (#53); the
  run itself is #54, likely overnight.
- The learning rate is untuned for a loss it wasn't chosen for. The README lists it.
- Temperature-1.0 samples from the raw student break the JSON more often than greedy decoding
  (parse_ok 84.5% on validation, greedy); those tokens are still graded by the teacher.
