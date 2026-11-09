# Referee report: function-calling distillation, 0.8B ← 2B (main @ 41815bd)

Independent review, 2026-11-09. Every number comes from a script in this folder (sN_*.py; outputs in
out/sN_*.md). **Measured** = computed from saved predictions or logs; **inferred** = interpretation.
Test and BFCL were used only to explain results already reported.

## 1. Executive summary

1. **How optional defaults are scored decides most of the close calls.** On test-clean, 123 rows are
   audited "sound" but tagged "default made explicit". Without them, D500 < B (p = 0.006) becomes
   p = 0.14 and B→D < B (p = 0.014) becomes p = 0.74. Under lenient_exact, D *beats* B on test-clean
   (67 vs 43, p = 0.028) [s1, s12]. "D ties B" is two opposite effects cancelling: D is better at
   function choice (names p = 0.006) and copying values; B is better at gold's default convention [s2].
2. **The teacher's "format" failures are missing calls.** Of the 2B's 58 unparsed test-clean outputs,
   45 parse after a closing-bracket repair; 43 are a first call followed by `}]}`. Repaired, the 2B
   goes 76.8 → 76.9 (+1 row) [s4, s4b].
3. **"B beats its teacher" holds on test-clean, not on validation.** Where both parse: test-clean
   83.9 vs 79.9 (p = 1e-6); validation 71.8 vs 70.2 (p = 0.12), lenient a tie [s3]. Inferred:
   dropping unparsed outputs is a label-free quality filter (34 of 2,137 dropped match gold [s4]),
   and SFT averages away the teacher's per-row quirks while D copies them.
4. **Validation cannot rank these methods.** Minimum detectable difference ≈ 2.4 points (80% power)
   [s8]. It ties D500 vs D3000 (p = 0.62) where test-clean (0.001), test (0.03) and BFCL (0.0002)
   separate them [s11]. Bootstrap: D's pick spread over steps 500/1500/3000/3410, each 18–31% [s8].
5. **D wins on BFCL by staying closer to the teacher.** D equals the 2B on 75.5% of BFCL rows, B
   71.7%. On rows the 2B gets right D keeps 765/827, B 737/827; on rows it gets wrong they tie. D
   reproduces 42.3% of the 2B's wrong answers on test-clean (B 34.5%); D fences ~2% of outputs like
   the 2B, B never [s3, s9, s10].
6. **Gold-quirk learning is large; ADR 0005's trigger is mis-specified.** On ambiguous test rows: A
   matches gold 56.4%, Bf 34.1%, B 22.3%, 2B 17.1% [s1]. The trigger compares point gains (38.6 vs
   38.4), so it never fires. Bf's lead over B shrinks val +5.8 → test +3.6 → test-clean +2.4.
7. **About half of A − Bf is the optional-argument convention.** Of 155 A-only test-clean rows, 84
   are default-only or extra-optional errors; the lenient gap is 5.1, not 7.5 [s5, s1]. The lenient
   filter kept 44% of rows whose gold values aren't in the query, 60% of multiple-call rows. On BFCL
   Bf chooses functions worse than B (names 36/14, p = 0.003).
8. **A label-free filter reaches most of Bf's precision.** Validity check + every value appears in
   the query: 33,918 rows at 86.8% lenient agreement with gold (B's data 71.9%) [s13].
9. **The B→D probe gap is systematic.** D's 200-row probe matches fc-eval ±1 at all 7 checkpoints;
   B→D's reads higher at 5 of 6 (+11 total), +19 on full validation at step 1000 [s7b].
10. **Multiple comparisons.** Holm over the README's 18 tests: D500 vs B test-clean → 0.056, B→D vs
    B → 0.11, no longer significant. Several "level" claims rest on ±2.3-point intervals [s8].

## 2. Claims audit

| # | Claim | Verdict | Reason | Evidence |
|---|---|---|---|---|
| 1 | B recovers 130% of the gap and beats its teacher on test-clean | holds on test-clean; weak on validation | Holds where both parse and under lenient; on validation p = 0.12, lenient tie | s3 |
| 2 | Part of B's lead is format (2B breaks off mid-JSON) | wrong as described | 43/45 repairable outputs stop after the first call: content errors | s4, s4b |
| 3 | Not memorisation; lead larger on unseen tool sets | holds (minor) | 0 exact normalised-query matches (README says 9); unseen tool sets n = 109: B 83.5 vs 2B 68.8 | s12 |
| 4 | B below the 2B on BFCL, losing on values | holds | 54/90, p = 0.003 (Holm 0.034) | s1, s2, s8 |
| 5 | D ties B on xLAM | weak | Strict tie; lenient D ahead p = 0.028; names D ahead p = 0.006 | s1, s2 |
| 6 | D beats B on BFCL from values and function choice, not format | mostly holds | 62/28; 5 of B's losing rows unparsed | s2, s9 |
| 7 | D level with the 2B on BFCL | holds (±2.4) | −0.2 [−2.4, +2.0] | s8 |
| 8 | Per GPU-hour B cheaper; D at B's budget behind on test-clean | weak | p = 0.14 without default rows; lenient 0.08; Holm 0.056 | s1, s8, s12 |
| 9 | Can't tell on-policy from more training | holds, cheaply testable | D500→D3000 +3.8 on BFCL (p = 0.0002); B's later checkpoints never run on BFCL | s11 |
| 10 | B→D doesn't help; behind B on test-clean | weak | p = 0.74 without default rows; lenient 47/45; 29/70 right→wrong are default-only, 31/70 now output the 2B's answer | s6, s12 |
| 11 | Filtered B gains ~2 points | holds | +2.5 [+0.9, +4.1], Holm 0.028; partly quirk alignment | s1, s8 |
| 12 | A − Bf from dropped harder rows and gold wording | partly | ~54% of A-only rows are optional-argument conventions; lenient gap 5.1 | s5 |
| 13 | A, D and the 2B level on BFCL | weak | A vs 2B +1.8 [−0.8, +4.4]; A omits 51 required args whose description mentions a default (~23 rows) | s8, s9 |
| 14 | ADR 0001: nothing chosen by looking at test | holds, caveat | "B cheaper per GPU-hour" concluded from test, against validation | s11 |
| 15 | ADR 0005: label noise shifts, doesn't reorder; trigger unfired | wrong | Quirk matching differs by method, so it can reorder; trigger insensitive | s1 |
| 16 | ADR 0002: auditing three baselines' errors avoids bias | holds | Trained models' errors on unaudited rows: 6–18 per model | s1 |
| 17 | ADR 0008: probe differs by "a few rows" | holds for D only | B→D +1 to +5 per 200 rows, +19 on full validation | s7b |
| 18 | ADR 0008: 0.15 threshold harmless | holds | 509–621/1,000 BFCL rows differ from raw; the flaw is the check rows (first 32 BFCL rows, simple, raw gets 28/32) | s11 |
| 19 | parse_ok = "nothing around it" | wrong (docs) | Parser strips fences and accepts a bare call object | s10 |
| 20 | Evals reproducible | holds | Same checkpoint twice flips 0–6 of 1,625 rows | s1 |

## 3. Analyses

### 3.1 The metric decides D vs B [s1, s12, s13]

| Test-clean, D vs B | Only D | Only B | p |
|---|---:|---:|---:|
| strict exact | 60 | 59 | 1.0 |
| lenient_exact | 67 | 43 | 0.028 |
| names | 21 | 6 | 0.006 |
| strict, model's own default args dropped (label-free) | 55 | 34 | 0.033 |
| strict, without the 123 "default made explicit" rows | 51 | 39 | 0.25 |
| validation, lenient | 34 | 34 | 1.0 |

Right out of the 123 "default made explicit" rows: A 97, Bf 58, B 52, D 41, D500 33, B→D 29, 2B 25.
Inferred: SFT learns the teacher's per-tool majority behaviour; reverse KL on the student's samples
reproduces the teacher's row-level behaviour. Number-vs-string type errors on test-clean: B 3, D 10,
B→D 15, 2B 15 [s2]. Dropping default-valued arguments isn't free: A falls 93.2 → 88.0 [s13].

### 3.2 D vs B by slice and error type [s2]

No slice difference significant on validation (all p > 0.4) or test-clean (smallest: parallel, 6/16,
p = 0.053). Errors of the losing model on test-clean discordant rows:

| Error | D wrong, B right (59) | B wrong, D right (60) |
|---|---:|---:|
| Default-only | 26 | 6 |
| Extra optional argument | 9 | 3 |
| Content, gold value in query | 2 | 12 |
| Missing call | 2 | 12 |
| String format | 0 | 6 |
| Number vs string | 5 | 0 |

Validation: same pattern weaker (default-only 9 vs 3, missing call 5 vs 5). Lengths similar (B 141,
D 145, 2B 146 chars); call count matches gold B 969, D 975 of 1,000.

### 3.3 Agreement with the teacher [s3]

Share equal to the 2B's output, then share of the 2B's parsed wrong answers reproduced:

| Model | Validation (of 290) | Test-clean (of 319) | BFCL (of 162) |
|---|---|---|---|
| A | 68.8, 10.7% | 75.7, 5.3% | 61.7, 14.2% |
| B | 76.5, 41.0% | 80.3, 34.5% | 71.7, 39.5% |
| Filtered B | 70.5, 20.3% | 77.5, 21.6% | 66.7, 21.6% |
| D | 77.3, 43.8% | 83.1, 42.3% | 75.5, 36.4% |
| B→D (400) | 78.5, 46.2% | 82.5, 43.9% | 70.7, 35.2% |

B's agreement keeps rising after step 1500 (validation 76.5 → 79.7 at 3277) while exact is flat
[s6]. Inferred: B's later checkpoints may do better on BFCL; never measured (idea 2).

### 3.4 Why D wins on BFCL but not on xLAM [s2, s9]

| BFCL category | Only D | Only B | p |
|---|---:|---:|---:|
| simple | 23 | 10 | 0.035 |
| multiple | 16 | 3 | 0.004 |
| parallel | 12 | 4 | |
| parallel_multiple | 11 | 11 | |

B's errors on the 62 D-only rows: 8 missing required args (D has none on all of BFCL), 10 values in
the query not copied, 6 values needing derivation (unit conversion, 5.0 vs 0.05, magnitudes), 6
missing calls, 5 unparsed. 48 of the 62 the 2B also gets right. Inferred: on unseen tools B falls
back on xLAM habits; D keeps more of the teacher's general behaviour. Side finding: A omits 51
required BFCL args, 50 with a default in the description (an xLAM gold habit, ~23 rows).

### 3.5 What the filtered-B filter removed [s5]

| Train slice | B keeps | Bf keeps |
|---|---:|---:|
| Single call | 99.2% | 77.1% |
| Multiple calls | 93.2% | 60.2% |
| Parallel | 94.0% | 66.4% |
| Gold values in query | 96.4% | 79.8% |
| Gold values not in query | 95.3% | 44.0% |

Dropped by error: content not in query 2,497; extra optional 2,510; content in query 1,881; string
format 1,657; missing call 1,197. Kept 2,262 default-only mismatches. Bf beats B on every validation
slice; on test-clean most on parallel (26/10, p = 0.011) and values not in query (44/21, p = 0.006).

A − Bf: 10.0 test, 7.4 test-clean, 5.1 test-clean lenient → gold quirks ≈ 2.6, default convention
≈ 2.3, remaining ≈ 5. Bf's errors on 155 A-only rows: default-only 48, extra optional 36, content
not in query 15, in query 14, missing call 13. Coverage proxy (filter kept for those functions):
A-only 58%, both right 74.5%, both wrong 54% — tracks difficulty as much as coverage; unproven.

### 3.6 B → B→D [s6]

Validation step 400: 159 outputs changed; right→wrong 30 (12 default-only), wrong→right 26; 45 now
equal the 2B's (25 before). Test-clean: right→wrong 70 (29 default-only, 16 extra optional, 31 now
the 2B's answer), wrong→right 43; lenient 47 vs 45. Step 1000: 22 of 50 right→wrong default-only.
The loss pulls B back to the teacher's argument conventions.

### 3.7 D's dynamics, the step-1000 dip, the probe [s6, s7, s7b]

Logs healthy: loss 0.047 → 0.020, rollout parse 0.972 → 0.991, length 56–62 tokens, truncation
≤ 0.45%, entropy up slightly at 1001–1250. The dip is missing calls: 52 at step 1000 vs 16 at 1500;
76 rows right→wrong 500→1000, 42 recover by 1500; agreement with the 2B rises through it. Inferred:
transient greedy failure to continue to later calls near peak lr (~1.9e-4). The 200-row probe was
too small to see it; the full probe matched fc-eval exactly (657 = 657).

B→D probe minus fc-eval: +5, +1, +2, +2, 0, +1 per 200 rows; +19 full at step 1000. D: never > ±1.
Same LoRA targets; vLLM maps the packed modules. Only code difference: the start adapter loaded via
PeftModel.from_pretrained without peft_config. Unexplained.

### 3.8 Selection and power [s8, s11]

Optimism of the selected checkpoint (split-half, at 1,000 rows): D 0.8, B 0.6, Bf 0.5, A 0.4. Small,
but selection unstable (B picks 1500 67% of the time). A "smallest checkpoint tied with the best"
rule would pick D500 (p = 0.62 vs 3000) and B1000 (p = 0.46).

## 4. Gaps and threats to validity

**High**
- H1. Metric dependence: the xLAM ordering of D, B, D500, B→D flips with default scoring; report
  lenient next to strict for every pairwise claim.
- H2. Validation underpowered (~2.4 points) and its label noise not neutral; ADR 0005's trigger
  should fire; Phase 4's "best on validation" needs a pre-set tie rule.
- H3. One seed: Bf > B, D500 < B, B→D < B, D > B on BFCL could all be overturned.
- H4. "Beats its teacher" confounded: an xLAM-fine-tuned 0.8B vs a zero-shot 2B. Needs a 2B
  self-trained on its own outputs.

**Medium**
- M1. "B cheaper per GPU-hour" weak: test-only, Holm-nonsignificant, convention-driven. B is charged
  1.43 h for 54,561 rows but used ~24,000 by step 1500 (~0.63 h).
- M2. B→D probe discrepancy could hide a loading/eval issue for continued adapters.
- M3. Bf's gain partly quirk alignment; worse function choice on BFCL (p = 0.003; missing calls 36
  vs 19).
- M4. "Level" = not significantly different, on ±2.3-point intervals.

**Low**
- test-clean from the test run vs separate runs: ≤ 0.1 points, ≤ 3 discordant rows.
- parse_ok doc vs code.
- Adapter check uses the first 32 BFCL rows (simple, easy); use a stratified random 32.
- Single labeller, non-independent audit review (disclosed).
- B − 2B: +2.8 validation vs +6.1 test (z ≈ 2), unexplained.

**C (#59).** Likely lands within ~2 points of B and D on validation. Pre-register BFCL, teacher
agreement and the default breakdown as explanatory readouts. Cheapest: top-k (k ≈ 20) logprobs from
vLLM on B's 52,424 outputs, so C − B isolates soft vs hard targets. Expect C to inherit the
teacher's default-writing uncertainty.

**E.** ~19% of gold unsound; A shows how much gold rewards quirks. Use a lenient or arg_f1 reward.
Choose the start explicitly (from A: RL on top of SFT; from B: labels as reward vs targets). Fix a
GPU-hour budget (e.g. 4 h) and subset up front: 4–8 samples at D's 15 s/step can't cover an epoch.

## 5. Experiments, ranked by value for cost

1. Report lenient and the default breakdown for every comparison — no GPU (s1, s12). Changes
   claims 5, 8, 10.
2. BFCL on every B and D checkpoint, explanation only — ~0.3 GPU-h. If B's BFCL rises with steps,
   D's edge is teacher imitation, not on-policy.
3. Label-free grounded filtered B (validity + values in query) — ~2.5 GPU-h, 33,918 rows at 86.8%.
   Could become the label-free headline. Watch the values-not-in-query slice.
4. Two more seeds of B and of Bf or B→D — ~4–8 GPU-h.
5. Check the B→D probe gap — ~0.2 GPU-h: HF generate on full validation and merged weights in vLLM.
6. Audit validation, or select on lenient with a pre-set cheapest-within-noise rule — no GPU,
   ~600 rows of labelling.
7. (stretch) Self-train the 2B on its own parsed outputs — ~6–8 GPU-h.
8. (stretch) Student self-consistency, k = 5 at T = 0.7 — ~0.1 GPU-h.
9. (stretch) 4B teacher pass for B — ~8 GPU-h.
10. Don't spend on constrained JSON decoding: repair recovers 1/45; students parse ≥ 99.5%.

## 6. Suggested changes

**Roadmap (#2):** ticket for lenient + default breakdown (1); BFCL curve before C (2); grounded
label-free filtered B as a method (3); fire ADR 0005's trigger, redefined on ambiguous rows, then
audit validation or select on lenient with a pre-set tie rule; #59 records the expected-tie risk and
readouts; E with lenient reward, fixed budget, explicit start; reopen #58 only if (5) finds an eval
discrepancy.

**README:** fix the "format" sentence (missing calls; dropping unparsed outputs as a label-free
filter, 34/2,137 match gold); "D ties B" → strict tie, D better at choosing and completing calls, B
better at xLAM's optional-argument convention; soften "B is cheaper per GPU-hour"; B→D "behind under
strict, level under lenient, moves B towards the teacher's conventions"; BFCL "not significantly
different" and A's omitted-default habit; Limitations: quirk rates and the ~2.4-point validation
limit; fix the parse_ok definition.
