# #69: does fc-eval score B then D correctly?

Never merged. Scripts run from the GPU worktree's root with its venv (`chain.sh`); outputs in
`runs/69-adapter-scoring/` (not committed). 2026-11-10, about 0.4 GPU-h, validation only.

## Question

B then D's in-training probe read 70.3 on all of validation at step 1000; fc-eval read 68.4.
D's probe matched fc-eval within 1 row at all 7 checkpoints. Is fc-eval wrong for an adapter
continued from another adapter?

## Method

The saved checkpoint scored three ways on all 1,000 validation rows, greedy, the eval prompt,
1,024 tokens:

- **adapter**: fc-eval's existing run (vLLM 0.33 with the LoRA adapter).
- **merged**: `merge_adapter.py` folds the adapter into a copy of the base weights
  (W += α/r · B·A, all 186 modules), then plain `fc-eval --model <copy>`: no adapter for vLLM to load.
- **hf**: `hf_answers.py`, HF `generate` with the adapter loaded through PEFT, fc-opd's own
  `greedy_answers`, as the probe produces them, but on the checkpoint read back from disk.

B's selected checkpoint (step 1500), an adapter trained from scratch, is the control.

## Results (`compare.py`)

| Checkpoint | adapter | merged | hf | probe in training |
|---|---:|---:|---:|---:|
| B then D, step 1000 | 684 | 684 | 682 | 703 |
| B, step 1500 | 708 | 711 | 708 | (no probe) |

Rows only one of the pair gets right:

| Pair | B then D | B |
|---|---|---|
| adapter vs merged | 2 / 2 (text differs on 31) | 0 / 3 (21) |
| adapter vs hf | 3 / 1 (31) | 1 / 1 (17) |
| merged vs hf | 3 / 1 (34) | 3 / 0 (20) |

On the probe's 200 rows: adapter 140, hf 139.

## Conclusions

1. **fc-eval's numbers for B then D stand.** Three engines agree within 2 rows on the saved
   checkpoint. The 19-row gap is between the probe and the *saved* checkpoint, not between HF and
   vLLM: the model in memory during training scored higher than the adapter it saved. The probe is
   watch-only (ADR 0008), so no reported number changes; #58 stays closed.
2. **vLLM applies the adapter fully.** Merged weights and the adapter differ by at most 3 rows, and
   both match HF as closely as each other. No sign of a partial load for either checkpoint.
3. **Why the in-training probe differs is not found.** Ruled out: LoRA dropout (0 in all three runs)
   and bf16 drift from TRL's merge/unmerge around each vLLM weight sync (`roundtrip.py`: one round
   trip changes 8.3% of base weights for B then D, but 8.6% for D at step 3000, whose probe matched).
   The remaining difference is the start: B's adapter loaded with `PeftModel.from_pretrained` and
   handed to TRL, against TRL wrapping a fresh `LoraConfig` for D. Matters only if a run is ever
   continued from an adapter again.
