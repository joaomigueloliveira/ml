
## validation: agreement with the 2B (identical parsed calls)
teacher right 680, teacher wrong but parsed 290, teacher unparsed 30
model | agree with 2B, all rows | agree on teacher-wrong-parsed rows (copies the error) | right on teacher-wrong-parsed rows | wrong on teacher-right rows
0.8B | 50.7 | 75/290 (25.9) | 31 | 248
A | 68.8 | 31/290 (10.7) | 164 | 23
B | 76.5 | 119/290 (41.0) | 49 | 34
Bf | 70.5 | 59/290 (20.3) | 102 | 34
D | 77.3 | 127/290 (43.8) | 45 | 34
D500 | 71.1 | 87/290 (30.0) | 63 | 56
BD | 78.5 | 134/290 (46.2) | 37 | 29

## test-clean: agreement with the 2B (identical parsed calls)
teacher right 1248, teacher wrong but parsed 319, teacher unparsed 58
model | agree with 2B, all rows | agree on teacher-wrong-parsed rows (copies the error) | right on teacher-wrong-parsed rows | wrong on teacher-right rows
0.8B | 55.9 | 94/319 (29.5) | 58 | 434
A | 75.7 | 17/319 (5.3) | 251 | 35
B | 80.3 | 110/319 (34.5) | 115 | 53
Bf | 77.5 | 69/319 (21.6) | 159 | 58
D | 83.1 | 135/319 (42.3) | 94 | 32
D500 | 78.2 | 99/319 (31.0) | 104 | 77
BD | 82.5 | 140/319 (43.9) | 81 | 47

## bfcl: agreement with the 2B (identical parsed calls)
teacher right 827, teacher wrong but parsed 162, teacher unparsed 11
model | agree with 2B, all rows | agree on teacher-wrong-parsed rows (copies the error) | right on teacher-wrong-parsed rows | wrong on teacher-right rows
0.8B | 55.5 | 38/162 (23.5) | 34 | 222
A | 61.7 | 23/162 (14.2) | 86 | 78
B | 71.7 | 64/162 (39.5) | 46 | 90
Bf | 66.7 | 35/162 (21.6) | 69 | 96
D | 75.5 | 59/162 (36.4) | 52 | 62
D500 | 70.4 | 49/162 (30.2) | 49 | 96
BD | 70.7 | 57/162 (35.2) | 46 | 92

## validation: B vs 2B where both parse (n=968): B 71.8, 2B 70.2, only B 49 / only 2B 34, p=0.124
  lenient: B 74.1, 2B 73.9, 36/34, p=0.905
  2B's error on rows B right, 2B wrong (both parse): [('optional default added/dropped', 16), ('missing call', 7), ('extra optional arg', 5), ('string format/case only', 4), ('content (gold not in query)', 3), ('extra required arg', 3), ('content (gold in query)', 3), ('wrong function', 2)]
  B's error on rows 2B right, B wrong (both parse): [('content (gold in query)', 7), ('missing call', 6), ('string format/case only', 3), ('wrong function', 3), ('content (gold not in query)', 3), ('number (gold not in query)', 3), ('extra required arg', 2), ('extra optional arg', 2)]
  2B unparsed 30: median chars 99, >2000 chars: 1; B right on 13, D right on 15

## test-clean: B vs 2B where both parse (n=1561): B 83.9, 2B 79.9, only B 115 / only 2B 52, p=1.21e-06
  lenient: B 87.6, 2B 85.2, 96/58, p=0.00275
  2B's error on rows B right, 2B wrong (both parse): [('optional default added/dropped', 32), ('extra optional arg', 19), ('missing call', 17), ('type (str vs number/bool)', 12), ('extra required arg', 6), ('content (gold not in query)', 5), ('extra unknown arg', 5), ('wrong function', 4)]
  B's error on rows 2B right, B wrong (both parse): [('content (gold in query)', 10), ('content (gold not in query)', 7), ('optional default added/dropped', 6), ('string format/case only', 6), ('missing optional arg', 5), ('extra optional arg', 5), ('missing call', 4), ('missing required arg', 3)]
  2B unparsed 58: median chars 71, >2000 chars: 0; B right on 43, D right on 44

## train: 2B unparsed 2137 of 54561; median chars 87; >3000 chars 29
