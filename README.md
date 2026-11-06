# qwen-function-calling-distillation

Can a small Qwen3.5 model learn function calling from a bigger one? This project compares the three
kinds of knowledge distillation on the same task, the same held-out set and a single consumer GPU:

1. **SFT on teacher outputs** (sequence-level distillation)
2. **Logit distillation** on the teacher's top-k token probabilities
3. **On-policy distillation**, where the student writes and the teacher scores every token
   ([Thinking Machines, 2025](https://thinkingmachines.ai/blog/on-policy-distillation/);
   [GKD](https://arxiv.org/abs/2306.13649))

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

Baselines (#10), strict exact match with 95% intervals. Full tables, slices and McNemar tests:
[`results/baselines.md`](results/baselines.md).

| Model | Role | Test | Test-clean | BFCL (unseen tools) |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B | student | 46.4 [44.2, 48.5] | 55.1 [52.6, 57.5] | 64.8 [61.8, 67.7] |
| Qwen3.5-2B | teacher | 65.3 [63.2, 67.4] | 76.9 [74.8, 78.9] | 82.7 [80.2, 84.9] |
| Qwen3.5-4B | reference | 69.0 [67.0, 71.0] | 82.2 [80.3, 84.0] | 82.7 [80.2, 84.9] |

The gap to close is 21.8 points on test-clean. Training results: _in progress_.

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
```

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
notebooks/      01_dataset.ipynb: the dataset story and the split
tests/          unit tests (CPU only)
data/           committed split ids and the audit labels
docs/           dataset card, ADRs, the label-audit rubric and the BFCL notes
results/        results worth keeping (label audit, baselines)
```

## License

Apache-2.0. Qwen3.5 models are Apache-2.0; the xLAM dataset is CC-BY-4.0.
