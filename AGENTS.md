# AGENTS.md

Function-calling distillation on xLAM-60k: a small Qwen3.5 student learns from a larger Qwen3.5
teacher via SFT, logit distillation and on-policy distillation. See `README.md` for the goal and
metrics.

Plan, phases and decisions: issue #2, the project's wayfinder map; the reasons behind the big ones:
`docs/adr/`. Read both before starting any roadmap issue.

## Objectives

Check every plan and ticket against these (agreed 2026-11-07, the map's Notes hold the detail):

1. The maintainer **learns** the concepts: SFT, distillation, on-policy training, RL, evals.
2. The output is a **clean git repo that tells the story** of the experiments and their conclusions.
3. **KISS.** Every phase must fill a blank in the README claim (issue #2's Destination); anything else
   is a stretch goal. Add rigour (new baselines, tests, tooling) only when a result needs it.

## Commands

```bash
uv sync                          # base + dev tools (CPU only)
uv sync --extra inference        # + vLLM (GPU)
uv run pytest                    # tests
uv run pre-commit run --all-files  # ruff, ruff-format, mypy --strict, uv-lock, file hygiene
uv run fc-eval --model <hf-id> --n 500
```

Never `pip install`; add dependencies with `uv add` (or `uv add --optional <extra>` for GPU-only ones)
so `uv.lock` stays in sync.

`uv sync` and plain `uv run` sync the environment to base + dev, which uninstalls vLLM. On the GPU
machine run evals with `uv run --extra inference`, and run everything else with `uv run --no-sync`
while a GPU run is in progress.

## Code conventions

- Package code lives in `src/fc_distill/`; tests in `tests/` mirror module names.
- Tests run on CPU with no network. GPU libraries are imported inside functions (ruff `TID253`).
- Splits are train / validation / test (ADR 0001). Choose checkpoints and settings on validation;
  the test split only produces official numbers. Test labels stay as published; `test-clean` is a
  subset (ADR 0002).
- Experiment outputs go to `runs/` (gitignored). Results worth keeping are summarised in `results/`.
  `runs` is a symlink to one directory shared by every worktree (`$FC_RUNS_DIR`, default
  `~/ml-runs`), so removing a worktree keeps its runs. `orca.yaml`'s setup hook creates it; in a
  worktree made another way, run that hook's commands before writing any runs.

## Workflow

- `main` is protected: every change lands through a pull request with green CI, squash-merged.
- One issue → one branch → one PR. Branch names: `<type>/<issue>-<slug>`, e.g. `feat/3-baseline-eval`.
- Commits and PR titles use Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`,
  `refactor:`, `ci:`). Link the issue in the PR body with `Closes #<n>`.
- PR bodies follow `.github/pull_request_template.md`.
- A PR that changes data, splits, scoring or reported numbers also gets an eval-risk review:
  `docs/agents/eval-review.md`.
- Commits and PR bodies carry no AI attribution; a `commit-msg` hook rejects it in commits.

## Agent skills

### Issue tracker

Issues and specs live in GitHub Issues on `joaomigueloliveira/qwen-function-calling-distillation`, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root, created lazily. See `docs/agents/domain.md`.
