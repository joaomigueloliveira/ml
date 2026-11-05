# xlam-distill

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

A fixed 1,000-row eval split ([`data/splits.json`](data/splits.json), seed 0) is held out from all
training.

## Results

_In progress._

## Quickstart

Requires [uv](https://docs.astral.sh/uv/), an NVIDIA GPU for inference/training, and access to the
gated dataset (accept the terms on the dataset page, then `uv run hf auth login`).

```bash
uv sync --extra inference         # base + vLLM
uv run xlam-eval --model Qwen/Qwen3.5-0.8B --n 500
```

## Development

```bash
uv sync                           # base + dev tools, no GPU needed
uv run pre-commit install
uv run pytest
```

Linting and formatting use ruff, type checking uses mypy (strict), and both run in pre-commit and CI.

## Layout

```
src/xlam_distill/
  data.py       dataset loading, fixed split, prompt
  scoring.py    strict, deterministic scorer
  evaluate.py   vLLM eval CLI (xlam-eval)
tests/          unit tests (CPU only)
data/           committed split file
```

## License

Apache-2.0. Qwen3.5 models are Apache-2.0; the xLAM dataset is CC-BY-4.0.
