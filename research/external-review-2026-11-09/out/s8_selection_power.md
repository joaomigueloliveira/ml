A: full-validation exact by step {500: 78.7, 1000: 81.0, 1500: 80.8, 2000: 84.2, 2500: 83.7, 3000: 84.4, 3411: 84.7}
   bootstrap selection frequency: {500: 0, 1000: 0, 1500: 0, 2000: 24, 2500: 1, 3000: 10, 3411: 66} (%)
   split-half optimism of the selected checkpoint (500 rows): 0.62 pts; scaled to 1,000 rows ~0.44; mean regret vs the half-2 best 0.59 pts
B: full-validation exact by step {500: 62.5, 1000: 70.0, 1500: 70.8, 2000: 69.8, 2500: 70.0, 3000: 69.9, 3277: 70.1}
   bootstrap selection frequency: {500: 0, 1000: 18, 1500: 67, 2000: 3, 2500: 5, 3000: 1, 3277: 6} (%)
   split-half optimism of the selected checkpoint (500 rows): 0.83 pts; scaled to 1,000 rows ~0.58; mean regret vs the half-2 best 0.91 pts
Bf: full-validation exact by step {500: 73.1, 1000: 76.6, 1500: 75.0, 2000: 76.3, 2356: 75.9}
   bootstrap selection frequency: {500: 0, 1000: 66, 1500: 0, 2000: 31, 2356: 3} (%)
   split-half optimism of the selected checkpoint (500 rows): 0.67 pts; scaled to 1,000 rows ~0.47; mean regret vs the half-2 best 0.66 pts
D: full-validation exact by step {500: 70.0, 1000: 65.7, 1500: 70.3, 2000: 69.1, 2500: 69.9, 3000: 70.6, 3410: 70.6}
   bootstrap selection frequency: {500: 18, 1000: 0, 1500: 25, 2000: 0, 2500: 2, 3000: 31, 3410: 24} (%)
   split-half optimism of the selected checkpoint (500 rows): 1.16 pts; scaled to 1,000 rows ~0.82; mean regret vs the half-2 best 1.17 pts
BD: full-validation exact by step {100: 67.7, 200: 69.0, 300: 68.9, 400: 70.4, 500: 69.1, 1000: 68.4}
   bootstrap selection frequency: {100: 0, 200: 2, 300: 0, 400: 97, 500: 0, 1000: 0} (%)
   split-half optimism of the selected checkpoint (500 rows): -0.01 pts; scaled to 1,000 rows ~-0.01; mean regret vs the half-2 best 0.08 pts

## Minimum detectable paired difference (80% power, alpha 0.05)
validation, B vs D (76 discordant): MDE ~2.4 pts
validation, B vs D500 (106): MDE ~2.9 pts
test-clean, B vs D (119): MDE ~1.9 pts
test-clean, B vs D500 (184): MDE ~2.3 pts
BFCL, B vs D (90): MDE ~2.7 pts

## Headline tests (official runs as saved), Holm over 18
test | only first / only second | p | Holm p | diff [95% CI] pts
B vs 2B, test-clean | 158/53 | 2.55e-13 | 3.83e-12 | +6.5 [+4.7, +8.2]
B vs 2B, BFCL | 54/90 | 0.00339 | 0.0339 | -3.6 [-5.9, -1.3]
A vs 2B, BFCL (tie) | 96/78 | 0.197 | 1 | +1.8 [-0.8, +4.4]
D vs B, test-clean (tie) | 60/59 | 1 | 1 | +0.1 [-1.3, +1.4]
D vs 2B, BFCL (tie) | 60/62 | 0.928 | 1 | -0.2 [-2.4, +2.0]
D vs B, BFCL | 62/28 | 0.000438 | 0.00569 | +3.4 [+1.6, +5.2]
D500 vs B, test-clean | 73/111 | 0.00622 | 0.056 | -2.3 [-4.0, -0.7]
D500 vs B, BFCL (tie) | 61/65 | 0.789 | 1 | -0.4 [-2.6, +1.8]
BD vs B, test-clean | 43/70 | 0.0141 | 0.113 | -1.7 [-2.9, -0.4]
BD vs B, BFCL (tie) | 40/42 | 0.912 | 1 | -0.2 [-2.0, +1.6]
BD vs D, BFCL | 24/60 | 0.000107 | 0.0015 | -3.6 [-5.4, -1.8]
A vs B, test-clean | 201/39 | 1.76e-27 | 3e-26 | +10.0 [+8.2, +11.8]
A vs D, test-clean | 199/38 | 1.61e-27 | 2.9e-26 | +9.9 [+8.1, +11.7]
Bf vs B, test-clean | 109/68 | 0.00254 | 0.028 | +2.5 [+0.9, +4.1]
Bf vs B, BFCL (tie) | 79/62 | 0.178 | 1 | +1.7 [-0.6, +4.0]
A vs Bf, test-clean | 155/34 | 1.17e-19 | 1.87e-18 | +7.4 [+5.8, +9.1]
A vs Bf, BFCL | 82/45 | 0.00131 | 0.0157 | +3.7 [+1.5, +5.9]
A vs D, BFCL (tie) | 80/60 | 0.108 | 0.756 | +2.0 [-0.3, +4.3]
