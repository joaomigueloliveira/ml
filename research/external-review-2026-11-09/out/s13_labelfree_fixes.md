## (a) strict exact after dropping the model's own default-valued optional arguments

### validation
0.8B: 46.6 -> 46.6
2B: 68.0 -> 67.5
A: 84.7 -> 79.5
B: 70.8 -> 68.9
Bf: 76.6 -> 74.3
D: 70.6 -> 69.3
D500: 70.0 -> 68.7
BD: 70.4 -> 69.6
  canonicalised D vs B: 33/29 p=0.704
  canonicalised BD vs B: 27/20 p=0.382
  canonicalised D500 vs B: 43/45 p=0.915
  canonicalised B vs 2B: 47/33 p=0.146
  canonicalised Bf vs B: 73/19 p=1.23e-08

### test-clean
0.8B: 54.8 -> 54.8
2B: 76.8 -> 76.7
A: 93.2 -> 88.0
B: 83.3 -> 81.2
Bf: 85.8 -> 84.2
D: 83.3 -> 82.5
D500: 80.9 -> 79.8
BD: 81.6 -> 80.9
  canonicalised D vs B: 55/34 p=0.0334
  canonicalised BD vs B: 36/40 p=0.731
  canonicalised D500 vs B: 60/83 p=0.0654
  canonicalised B vs 2B: 118/45 p=9.74e-09
  canonicalised Bf vs B: 95/45 p=2.88e-05

## (b) label-free filters on the teacher's 52424 parsed train outputs (B's data)
filter | rows kept | share lenient-matching gold
parse only (B) | 52424 | 71.9
validity | 50919 | 73.4
grounded values | 34634 | 85.7
validity and grounded | 33918 | 86.8
lenient match (filtered B, uses labels) | 37696 | 100 by construction (94.0 strict)
