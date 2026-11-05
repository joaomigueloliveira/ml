# 0004. Qwen3.5-0.8B student, Qwen3.5-2B teacher

Date: 2026-11-05 · Status: accepted

## Context

Scouting run on 1,000 xLAM rows, greedy, thinking off (details in issue #4):

| Model | exact | eval time |
|---|---:|---:|
| Qwen3.5-0.8B | 49.3% | 49 s |
| Qwen3.5-2B | 69.4% | 95 s |
| Qwen3.5-4B | 71.2% | 398 s (memory-limited) |

On-policy distillation needs the teacher in memory next to the training student. A 4B teacher (~9 GB in
bf16) does not fit beside training on 12 GB; a 2B teacher (~4.5 GB) does.

## Decision

Student **Qwen3.5-0.8B**, teacher **Qwen3.5-2B**, both bf16 on the local GPU. All Qwen3.5 sizes share a
tokenizer, which logit and on-policy distillation require.

## Consequences

- A ~20-point gap to close, with no quantized teacher and no cloud GPU needed.
- The 4B adds about 2 points over the 2B; it returns only as a follow-up if results look teacher-limited.
