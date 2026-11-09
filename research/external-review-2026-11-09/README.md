# External review, 2026-11-09

An independent, read-only review of the experiments and conclusions up to main @ 41815bd: the
claims audit, analyses of the saved predictions, threats to validity and ranked experiment ideas.
`REPORT.md` is the report (also posted on #2); every number in it cites the script that made it.

This branch is never merged. The tickets it led to: #68–#75, and additions to #59.

Run a script from the repo root, against the shared `runs/`, CPU only:

    uv run --no-sync python research/external-review-2026-11-09/s1_metrics.py

Outputs land in `out/`. `out/s4b_repair_examples.md` is left out: it quotes gated xLAM gold answers;
`s4b_repair_examples.py` regenerates it locally.
