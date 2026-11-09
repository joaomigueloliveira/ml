# qwen-function-calling-distillation

Can a small Qwen3.5 model learn function calling from a bigger one, **without labels**? In practice
distillation is what you reach for when there are queries but no gold answers, and a model better
than the one you want to ship. This project trains Qwen3.5-0.8B from a Qwen3.5-2B teacher on one
consumer GPU, and compares the label-free methods with SFT on gold labels, the reference for what
labels buy (ADR 0007).

| Method | | Training uses labels |
|---|---|---|
| **A** | SFT on gold labels | yes: the reference |
| **B** | SFT on teacher outputs (sequence-level distillation) | no |
| **D** | On-policy distillation from the raw student: it writes, the teacher scores every token ([Thinking Machines, 2025](https://thinkingmachines.ai/blog/on-policy-distillation/); [GKD](https://arxiv.org/abs/2306.13649)) | no |
| **C** | Logit distillation on the teacher's top-k token probabilities | no |
| **filtered B** | B on the teacher outputs that match gold: what the teacher's mistakes cost | yes, as a filter |
| **E** | GRPO, reward checked against gold: does RL add anything on top of SFT? | yes, in the reward |

A, B, D and filtered B are done, and B then D (a pipeline) was tried and dropped; C and E are
next. Every result is
reported next to the untrained baselines and A, with two numbers on `test-clean`: the gap recovered
from the raw 0.8B towards the 2B, and the share of A's gain reached without labels.

## Task

[Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k):
given a user query and a set of tool definitions, output the JSON list of calls that answers it.
Scoring is deterministic: no LLM judge.

| Metric | Meaning |
|---|---|
| `parse_ok` | output is a valid JSON list of `{name, arguments}` calls, with nothing around it |
| `exact` | same calls as the gold answer (order-insensitive; `5 == 5.0`, `true != 1`) |
| `lenient_exact` | `exact`, but ignoring optional arguments passed at their declared default or `null`, on either side. Many xLAM defaults look like example values copied from API docs, so such calls are the same by the tool spec, not necessarily for the real API. `exact` stays the headline |
| `names` | the right functions were called |
| `arg_f1` | F1 over `(function, argument, value)` triples, as partial credit |

Each xLAM summary also breaks every metric down by call pattern (single / multiple / parallel),
distractor tools (0 / 1-2 / 3+) and argument source (every value written in the query, or not), with
row counts and 95% intervals. Buckets under 300 rows are flagged `too_small` (ADR 0001).

Splits: train (54,561), validation (1,000) and test (2,000), grouped so no query text appears in
two splits, with train filtered of duplicates, code-valued arguments, contradicting labels and gold
answers that also appear in validation or test. The data story, from the quality findings to the label audit and `test-clean`, is in the dataset card:
[`docs/DATASET.md`](docs/DATASET.md).

A second eval, on tools the student never trained on, uses the non-live single-turn part of
[BFCL](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) (1,000
rows), with specs converted to xLAM's format: [`docs/bfcl.md`](docs/bfcl.md).

## Results

Strict exact match with 95% intervals. Baselines (#10): [`results/baselines.md`](results/baselines.md).
Methods, with every checkpoint, McNemar tests and the teacher-output counts:
[`results/methods.md`](results/methods.md). Checkpoints are chosen on validation; test, test-clean
and BFCL are run once, for the chosen one.

| Model | Training uses labels | GPU-h | Test | Test-clean | BFCL (unseen tools) |
|---|---|---:|---:|---:|---:|
| Qwen3.5-0.8B (student) | | 0 | 46.2 [44.0, 48.4] | 54.8 [52.3, 57.2] | 64.5 [61.5, 67.4] |
| Qwen3.5-2B (teacher) | | 0 | 65.1 [63.0, 67.2] | 76.8 [74.7, 78.8] | 82.7 [80.2, 84.9] |
| Qwen3.5-4B (reference) | | 0 | 68.8 [66.7, 70.7] | 81.8 [79.8, 83.6] | 82.7 [80.2, 84.9] |
| **B**: 0.8B, SFT on teacher outputs | no | 3.12 | 71.2 [69.2, 73.1] | 83.4 [81.5, 85.1] | 79.1 [76.5, 81.5] |
| **D**: 0.8B, on-policy distillation | no | 13.54 | 70.5 [68.5, 72.5] | 83.3 [81.4, 85.0] | 82.5 [80.0, 84.7] |
| D at B's budget (step 500) | no | 2.22 | 69.0 [67.0, 71.0] | 81.0 [79.1, 82.9] | 78.7 [76.1, 81.1] |
| B then D (pipeline, step 400) | no | 4.77 | 69.8 [67.8, 71.8] | 81.7 [79.7, 83.5] | 78.9 [76.3, 81.3] |
| **Filtered B**: 0.8B, SFT on teacher outputs that match gold | yes, as a filter | 2.50 | 74.8 [72.8, 76.6] | 85.7 [83.9, 87.3] | 80.8 [78.2, 83.1] |
| **A**: 0.8B, SFT on gold labels | yes | 3.83 | 84.8 [83.1, 86.3] | 93.2 [91.9, 94.3] | 84.5 [82.1, 86.6] |

- **Without labels, B beats its own teacher on xLAM.** On test-clean it recovers 130% of the
  22.0-point gap from the 0.8B to the 2B (p < 0.0001 against both), and reaches 74% of what
  training on gold labels gains. #2's success criterion, at least 50% of the gap, is met. Part of
  it is format: 43 of the 159 rows B gets right and the 2B misses are 2B outputs that break off
  mid-JSON, which B never trained on (unparsed outputs are dropped). On the 1,563 rows where both
  outputs parse, B still leads, 83.9 to 79.8. It isn't memorisation: only 9 test-clean queries
  appear in train, and B's lead is larger on tool sets train never saw.
- **On tools it never trained on, it doesn't.** B gains 14.6 points on BFCL over the raw 0.8B but
  stays below the 2B (p = 0.003). Its format holds (parse_ok 98.8 vs 98.9); it loses on argument
  values, in every BFCL category. A ties the 2B there (p = 0.20).
- **On-policy distillation (D) ties B on xLAM, and matches the teacher on unseen tools.** On
  test-clean D and B are the same (83.3 vs 83.4; 60 and 62 rows only one of them gets right,
  p = 0.93): 129% of the gap to the 2B, 74% of A's gain. On BFCL D reaches 82.5, level with the 2B
  (p = 0.93) and above B (p = 0.0004), from argument values and function choice, not format. It
  took 4x B's GPU-hours: at B's budget D is level with B on BFCL too, and with one seed and two BFCL
  points the data can't tell on-policy training from more training.
- **Per GPU-hour, B is cheaper.** D's selected checkpoint took 13.5 GPU-hours to tie B's 3.1. At
  B's budget (step 500, 2.2 GPU-hours) D is behind on test-clean, 81.0 vs 83.4 (p = 0.006), and
  level on BFCL. Validation, on 1,000 rows, called that a tie (p = 0.50).
- **D's loss on top of B doesn't help.** B then D continues B's selected checkpoint with D's loss
  (lr 5e-5, a fourth of D's, stopped at step 1000 of the epoch, #58). No checkpoint beat B on
  validation; its best, step 400 (1.65 GPU-hours on top of B's), is behind B on test-clean (81.7 vs
  83.4, p = 0.01) and level on BFCL (78.9 vs 79.1, p = 0.91), below D there (p = 0.0001). Likely
  because the teacher that D's loss pulls towards is below B on xLAM, and on BFCL the gain D made
  from the raw student didn't appear from B in 1,000 steps. One learning rate, one seed: the run
  wasn't resumed.
- **Labels still buy a lot.** A beats B and D on test and test-clean (p < 0.0001); the teacher's
  outputs agree with gold on only 67.6% of the train rows B learns from. On BFCL, A, D and the 2B
  are level.
- **Training only on the teacher outputs that match gold gains about 2 points.** Filtered B
  keeps only the 37,696 teacher outputs that lenient-match gold (#60). It beats B on test-clean
  (85.7 vs 83.4, p = 0.007; 140% of the gap, 80% of A's gain), from fewer rows and fewer GPU-hours
  (2.50 vs 3.12), and ties it on BFCL (80.8 vs 79.1, p = 0.18). A is still 7.5 points ahead on
  test-clean and 3.7 on BFCL (p < 0.0001, p = 0.001): what's left likely comes from the 17,000
  dropped rows, probably the harder ones, and gold's own wording. Filtered B also differs from B in
  its schedule length (2,356 steps vs 3,277), and each is one seed.

B's and filtered B's GPU-hours include the 1.43 h the teacher took to answer the train split; D's
include its own sampling and the teacher's grading, inside every step; B then D's include B's.
Next: logit distillation, C (#59).

Training cost (#36): one epoch of LoRA SFT over the full train split takes about 3.9 GPU-hours on the
RTX 3060 (0.072 GPU-hours per 1,000 rows, 4.6 GiB peak), after length grouping removed most of the
padding. The settings and the sweeps behind them: [`results/sft-cost.md`](results/sft-cost.md).

Training curves and eval runs, the baselines included, are logged to the public W&B project
[joaomigueloliveira30/qwen-function-calling-distillation](https://forge.coreweave.com/wandb/joaomigueloliveira30/qwen-function-calling-distillation).
`results/` stays the source of truth; W&B mirrors it.

## Limitations

Shortcuts taken on purpose, why, and what would make us revisit them.

- **The LoRA settings and learning rate are untuned.** Every SFT method shares one config (r=16,
  α=32, lr 2e-4), chosen as common defaults for the cost run. A sweep costs about 4 GPU-hours per
  point and would have to be repeated for each method to stay fair (ADR 0006). Revisit if a method's
  validation curve looks learning-rate-limited, e.g. its best checkpoint is an early one.
- **One seed per method.** Comparisons rest on paired McNemar tests over the same test rows, not on
  seed-to-seed variance (ADR 0006). Revisit if two methods land within a point or two on validation.
- **Method D's learning rate is untuned.** It reuses the SFT config (lr 2e-4) for a different loss,
  (ADR 0008). Revisit if its validation curve collapses or is still climbing at the end of the
  epoch. B then D tried one rate, 5e-5, for 1,000 of the epoch's 3,410 steps.
- **"Label-free" is about training only.** Checkpoints are chosen on labelled validation and scored on
  labelled test, as in practice, where a small labelled eval set is the norm (ADR 0007).
- **One small teacher.** The 2B is right on about 65% of train rows; a stronger teacher would likely
  lift every label-free method. The results are about this pair (ADR 0004).
- **Validation is not audited.** Checkpoints are chosen on raw validation labels, which carry xLAM's
  label noise (ADR 0005). Revisit if a method gains much more on raw test than on `test-clean`.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/), an NVIDIA GPU for inference/training, and access to the
gated dataset (accept the terms on the dataset page, then `uv run hf auth login`).

```bash
uv sync --extra inference         # base + vLLM
uv run fc-eval --model Qwen/Qwen3.5-0.8B --split validation
uv run fc-eval --model Qwen/Qwen3.5-0.8B --split test-clean   # official numbers: test and test-clean
uv run fc-eval --model Qwen/Qwen3.5-0.8B --split bfcl         # unseen tools
uv run fc-eval --rescore runs/eval-test-Qwen_Qwen3.5-0.8B       # re-score a saved run, no GPU
uv run fc-report                  # results/baselines.{json,md} from the test and BFCL runs
uv run --no-sync fc-report --methods   # results/methods.{json,md}: each method's checkpoints

uv sync --extra inference --extra train   # + TRL, PEFT, flash-linear-attention
uv run --no-sync fc-sft --n-rows 1000     # method A: LoRA SFT on gold labels -> runs/train-sft-gold-*
```

### Experiment tracking

Logging to [Weights & Biases](https://forge.coreweave.com/wandb) is off unless asked for, so tests,
CI and offline runs need no account. To turn it on, log in once with `uv run wandb login`, then pass
`--wandb`:

```bash
uv run fc-eval --model Qwen/Qwen3.5-0.8B --split validation --wandb   # one eval run per split
uv run fc-report --wandb          # the official baselines, one run per model, from results/
```

The official baseline runs (`summary.json` and `predictions.jsonl`) are backed up in a private
W&B project, because they hold gated xLAM rows:
`joaomigueloliveira30/qwen-function-calling-distillation-backup`, artifact `official-baseline-runs`.

`WANDB_MODE=offline` keeps the run local (`wandb sync` uploads it later), and `WANDB_PROJECT` /
`WANDB_ENTITY` point it at another project. Trainers log through the same helper,
`fc_distill.tracking`, with GPU-hours (the teacher's included) as the x-axis. Every eval run uses
the same flat keys (`test_clean/exact`, `bfcl/slice/simple/exact`, ...), so runs compare in one
table. W&B findings, the key schema and what trainers log: [`docs/wandb.md`](docs/wandb.md),
[`docs/wandb-dashboard.md`](docs/wandb-dashboard.md).

## Development

```bash
uv sync                           # base + dev tools, no GPU needed
uv run pre-commit install
uv run pytest

# re-run the dataset notebook (writes data/splits.json; needs the HF login above)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset.ipynb
```

Linting and formatting use ruff, type checking uses mypy (strict), and both run in pre-commit and CI.

## Layout

```
src/fc_distill/
  data.py       dataset loading (pinned revision), committed splits, prompt
  quality.py    detectors for label and schema problems
  splits.py     train / validation / test construction and train filters
  eda.py        statistics and plots for the dataset notebook
  scoring.py    strict, deterministic scorer
  evaluate.py   vLLM eval CLI (fc-eval)
  audit.py      test label audit and the test-clean subset (fc-audit)
  bfcl.py       BFCL unseen-tools subset: loading, spec conversion, scoring
  slices.py     per-slice scores: call pattern, distractor tools, argument source
  report.py     baselines report with intervals and McNemar tests (fc-report)
  methods.py    method results: checkpoint selection, official scores, teacher-output counts (fc-report --methods)
  sft.py        methods A and B: LoRA SFT on gold labels or teacher outputs, GPU-hour accounting (fc-sft)
  opd.py        method D: on-policy distillation with TRL's DistillationTrainer (fc-opd)
  tracking.py   W&B tracking shared by trainers and evals; GPU-hour accounting, provenance
  eval_tracking.py  how eval results are laid out in W&B (flat keys, slices table)
notebooks/      01_dataset.ipynb: the dataset story and the split
tests/          unit tests (CPU only)
data/           committed split ids and the audit labels
docs/           dataset card, ADRs, the label-audit rubric, the BFCL and W&B notes
results/        results worth keeping (label audit, baselines, methods)
```

## License

Apache-2.0. Qwen3.5 models are Apache-2.0; the xLAM dataset is CC-BY-4.0.
