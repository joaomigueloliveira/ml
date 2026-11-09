
## validation: exact before / after bracket repair (rows repaired to parse; of those now exact)
0.8B | 46.6 -> 51.7 | unparsed 155, repaired 76, now right 51
2B | 68.0 -> 68.1 | unparsed 30, repaired 18, now right 1
A | 84.7 -> 84.7 | unparsed 4, repaired 1, now right 0
B | 70.8 -> 71.0 | unparsed 4, repaired 3, now right 2
Bf | 76.6 -> 76.8 | unparsed 10, repaired 3, now right 2
D | 70.6 -> 70.7 | unparsed 4, repaired 2, now right 1
BD | 70.4 -> 70.4 | unparsed 5, repaired 1, now right 0
  after repair B vs 2B: only B 63 / only 2B 34, p=0.00423
  after repair D vs 2B: only D 59 / only 2B 33, p=0.00878
  after repair Bf vs 2B: only Bf 121 / only 2B 34, p=1.21e-12

## test-clean: exact before / after bracket repair (rows repaired to parse; of those now exact)
0.8B | 54.8 -> 59.8 | unparsed 220, repaired 109, now right 82
2B | 76.8 -> 76.9 | unparsed 58, repaired 45, now right 1
4B | 81.8 -> 81.8 | unparsed 66, repaired 1, now right 1
A | 93.2 -> 93.2 | unparsed 3, repaired 0, now right 0
B | 83.3 -> 83.3 | unparsed 8, repaired 0, now right 0
Bf | 85.8 -> 85.8 | unparsed 3, repaired 0, now right 0
D | 83.3 -> 83.3 | unparsed 3, repaired 1, now right 0
BD | 81.6 -> 81.7 | unparsed 6, repaired 4, now right 1
  after repair B vs 2B: only B 157 / only 2B 53, p=3.84e-13
  after repair D vs 2B: only D 137 / only 2B 32, p=1.12e-16
  after repair Bf vs 2B: only Bf 203 / only 2B 58, p=4.92e-20
  gap recovered vs repaired raw models: B 129% (repaired 2B 76.9)
  gap recovered vs repaired raw models: D 129% (repaired 2B 76.9)
  gap recovered vs repaired raw models: Bf 140% (repaired 2B 76.9)

## bfcl: exact before / after bracket repair (rows repaired to parse; of those now exact)
0.8B | 64.5 -> 67.4 | unparsed 101, repaired 40, now right 29
2B | 82.7 -> 82.7 | unparsed 11, repaired 2, now right 0
4B | 82.7 -> 82.9 | unparsed 62, repaired 2, now right 2
A | 84.5 -> 84.6 | unparsed 8, repaired 2, now right 1
B | 79.1 -> 79.5 | unparsed 12, repaired 5, now right 4
Bf | 80.8 -> 81.3 | unparsed 16, repaired 7, now right 5
D | 82.5 -> 82.6 | unparsed 7, repaired 2, now right 1
BD | 78.9 -> 79.2 | unparsed 11, repaired 5, now right 3
  after repair B vs 2B: only B 55 / only 2B 87, p=0.00904
  after repair D vs 2B: only D 60 / only 2B 61, p=1
  after repair Bf vs 2B: only Bf 77 / only 2B 91, p=0.316

## train: 2B unparsed 2137, repaired to parse 1210, of which lenient-match gold 34
