
## D: probe (HF) vs fc-eval (vLLM) on the same 200 rows; fc-eval on all 1,000
step 500 | probe 144/200 | fc-eval 143/200 | diff +1 | fc-eval full 700/1000
step 1000 | probe 142/200 | fc-eval 142/200 | diff +0 | fc-eval full 657/1000
step 1500 | probe 150/200 | fc-eval 151/200 | diff -1 | fc-eval full 703/1000
step 2000 | probe 141/200 | fc-eval 141/200 | diff +0 | fc-eval full 691/1000
step 2500 | probe 144/200 | fc-eval 144/200 | diff +0 | fc-eval full 699/1000
step 3000 | probe 146/200 | fc-eval 147/200 | diff -1 | fc-eval full 706/1000
step 3410 | probe 147/200 | fc-eval 147/200 | diff +0 | fc-eval full 706/1000

## BD: probe (HF) vs fc-eval (vLLM) on the same 200 rows; fc-eval on all 1,000
step 100 | probe 142/200 | fc-eval 137/200 | diff +5 | fc-eval full 677/1000
step 200 | probe 141/200 | fc-eval 140/200 | diff +1 | fc-eval full 690/1000
step 300 | probe 142/200 | fc-eval 140/200 | diff +2 | fc-eval full 689/1000
step 400 | probe 145/200 | fc-eval 143/200 | diff +2 | fc-eval full 704/1000
step 500 | probe 141/200 | fc-eval 141/200 | diff +0 | fc-eval full 691/1000
step 1000 | probe 141/200 | fc-eval 140/200 | diff +1 | fc-eval full 684/1000

B step 1500 fc-eval on the probe rows: 143/200; full 708/1000
