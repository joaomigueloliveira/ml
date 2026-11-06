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
| `names` | the right functions were called |
| `arg_f1` | F1 over `(function, argument, value)` triples, as partial credit |

Splits: train (56,392), validation (1,000) and test (2,000), grouped so no query text appears in
two splits, with train filtered of duplicates, code-valued arguments and contradicting labels. How
and why: [`notebooks/01_dataset.ipynb`](notebooks/01_dataset.ipynb) and [`docs/adr/`](docs/adr/).

A second eval, on tools the student never trained on, uses the non-live single-turn part of
[BFCL](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) (1,000
rows), with specs converted to xLAM's format: [`docs/bfcl.md`](docs/bfcl.md).

## Results

_In progress._

## Quickstart

Requires [uv](https://docs.astral.sh/uv/), an NVIDIA GPU for inference/training, and access to the
gated dataset (accept the terms on the dataset page, then `uv run hf auth login`).

```bash
uv sync --extra inference         # base + vLLM
uv run fc-eval --model Qwen/Qwen3.5-0.8B --split validation
uv run fc-eval --model Qwen/Qwen3.5-0.8B --split test-clean   # official numbers: test and test-clean
uv run fc-eval --model Qwen/Qwen3.5-0.8B --split bfcl         # unseen tools
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
notebooks/      01_dataset.ipynb: the dataset story and the split
tests/          unit tests (CPU only)
data/           committed split ids and the audit labels
docs/           ADRs, the label-audit rubric and the BFCL notes
results/        results worth keeping (label audit)
```

## License

Apache-2.0. Qwen3.5 models are Apache-2.0; the xLAM dataset is CC-BY-4.0.
