## test-clean rows audited sound with category 'default made explicit': 123
0.8B: exact 21/123, lenient 75/123
2B: exact 25/123, lenient 77/123
4B: exact 45/123, lenient 87/123
A: exact 97/123, lenient 105/123
B: exact 52/123, lenient 81/123
Bf: exact 58/123, lenient 88/123
D: exact 41/123, lenient 80/123
D500: exact 33/123, lenient 67/123
BD: exact 29/123, lenient 72/123
D vs B, test-clean without those rows (n=1502): 51/39 p=0.246
BD vs B, test-clean without those rows (n=1502): 38/42 p=0.738
D500 vs B, test-clean without those rows (n=1502): 65/84 p=0.14

## test-clean by whether the exact tool set appears in train
tool set seen n=1516: B 83.2, D 83.7, 2B 77.4; B vs 2B 139/50 p=6.85e-11
tool set unseen n=109: B 83.5, D 78.0, 2B 68.8; B vs 2B 19/3 p=0.000855
test-clean queries whose normalised text appears in train: 0
