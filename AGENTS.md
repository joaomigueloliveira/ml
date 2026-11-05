# AGENTS.md

Function-calling distillation on xLAM-60k: a small Qwen3.5 student learns from a larger Qwen3.5
teacher via SFT, logit distillation and on-policy distillation. See `README.md` for the goal and
metrics.

## Commands

```bash
uv sync                          # base + dev tools (CPU only)
uv sync --extra inference        # + vLLM (GPU)
uv run pytest                    # tests
uv run pre-commit run --all-files  # ruff, ruff-format, mypy --strict, uv-lock, file hygiene
uv run xlam-eval --model <hf-id> --n 500
```

Never `pip install`; add dependencies with `uv add` (or `uv add --optional <extra>` for GPU-only ones)
so `uv.lock` stays in sync.

## Code conventions

- Package code lives in `src/xlam_distill/`; tests in `tests/` mirror module names.
- Everything must pass `mypy --strict`. Untyped third-party modules are listed in `pyproject.toml`.
- Tests run on CPU with no network. Code that needs a GPU or a download keeps its heavy imports
  inside the function (see `evaluate.py`) so the rest of the package stays importable in CI.
- The eval split in `data/splits.json` is fixed. Never regenerate it or train on eval rows.
- Experiment outputs go to `runs/` (gitignored). Results worth keeping are summarised in `results/`.

## Workflow

- `main` is protected: every change lands through a pull request with green CI, squash-merged.
- One issue → one branch → one PR. Branch names: `<type>/<issue>-<slug>`, e.g. `feat/3-baseline-eval`.
- Commits and PR titles use Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`,
  `refactor:`, `ci:`). Link the issue in the PR body with `Closes #<n>`.
- PR bodies follow `.github/pull_request_template.md`.
- No AI attribution lines (`Co-Authored-By`, "Generated with …") in commits or PRs.

## Agent skills

### Issue tracker

Issues and specs live in GitHub Issues on `joaomigueloliveira/ml`, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root, created lazily. See `docs/agents/domain.md`.
