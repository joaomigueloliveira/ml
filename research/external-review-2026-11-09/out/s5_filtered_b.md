train 54561; B keeps 52424; Bf keeps 37696

## Teacher's train outputs by class (B's training rows), and what Bf keeps
right | 35434 | in B 35434 | in Bf 35434
extra optional arg | 2510 | in B 2510 | in Bf 0
content (gold not in query) | 2497 | in B 2497 | in Bf 0
optional default added/dropped | 2262 | in B 2262 | in Bf 2262
unparsed | 2137 | in B 0 | in Bf 0
content (gold in query) | 1881 | in B 1881 | in Bf 0
string format/case only | 1657 | in B 1657 | in Bf 0
missing call | 1197 | in B 1197 | in Bf 0
extra required arg | 1180 | in B 1180 | in Bf 0
number (gold not in query) | 1034 | in B 1034 | in Bf 0
missing optional arg | 645 | in B 645 | in Bf 0
type (str vs number/bool) | 610 | in B 610 | in Bf 0
extra unknown arg | 450 | in B 450 | in Bf 0
wrong function | 367 | in B 367 | in Bf 0
missing required arg | 366 | in B 366 | in Bf 0
extra call | 158 | in B 158 | in Bf 0
number (gold in query) | 155 | in B 155 | in Bf 0
unknown function | 21 | in B 21 | in Bf 0

## Train rows by slice: share kept by B and by Bf
call_pattern=single | n=24892 | B keeps 99.2 | Bf keeps 77.1 | share of B's rows 47.1 | share of Bf's rows 50.9
call_pattern=multiple | n=19284 | B keeps 93.2 | Bf keeps 60.2 | share of B's rows 34.3 | share of Bf's rows 30.8
call_pattern=parallel | n=10385 | B keeps 94.0 | Bf keeps 66.4 | share of B's rows 18.6 | share of Bf's rows 18.3
distractors=0 | n=22112 | B keeps 95.1 | Bf keeps 66.2 | share of B's rows 40.1 | share of Bf's rows 38.8
distractors=1-2 | n=21122 | B keeps 95.8 | Bf keeps 67.9 | share of B's rows 38.6 | share of Bf's rows 38.1
distractors=3+ | n=11327 | B keeps 98.5 | Bf keeps 76.9 | share of B's rows 21.3 | share of Bf's rows 23.1
argument_source=from_query | n=38245 | B keeps 96.4 | Bf keeps 79.8 | share of B's rows 70.3 | share of Bf's rows 80.9
argument_source=not_in_query | n=16316 | B keeps 95.3 | Bf keeps 44.0 | share of B's rows 29.7 | share of Bf's rows 19.1

## validation: Bf vs B and A vs Bf by slice
call_pattern=single n=492 | B 80.3 Bf 85.0 A 89.2 | B/Bf only 11/34 p=0.000824 | A/Bf only 29/8 p=0.000753
call_pattern=multiple n=324 | B 63.0 Bf 69.8 A 79.3 | B/Bf only 11/33 p=0.00126 | A/Bf only 40/9 p=9.26e-06
call_pattern=parallel n=184 | B 59.2 Bf 66.3 A 82.1 | B/Bf only 5/18 p=0.0106 | A/Bf only 33/4 p=1.08e-06
distractors=0 n=400 | B 64.2 Bf 72.8 A 82.5 | B/Bf only 9/43 p=2.04e-06 | A/Bf only 48/9 p=1.52e-07
distractors=1-2 n=379 | B 71.8 Bf 76.8 A 83.6 | B/Bf only 8/27 p=0.00188 | A/Bf only 36/10 p=0.000156
distractors=3+ n=221 | B 81.0 Bf 83.3 A 90.5 | B/Bf only 10/15 p=0.424 | A/Bf only 18/2 p=0.000402
argument_source=from_query n=707 | B 82.5 Bf 88.5 A 93.8 | B/Bf only 14/57 p=2.67e-07 | A/Bf only 50/13 p=3.02e-06
argument_source=not_in_query n=293 | B 42.7 Bf 47.8 A 62.8 | B/Bf only 13/28 p=0.0275 | A/Bf only 52/8 p=5.21e-09
  B's error where Bf right, B wrong: [('string format/case only', 17), ('optional default added/dropped', 10), ('content (gold in query)', 10), ('extra required arg', 9), ('missing call', 7), ('number (gold not in query)', 5), ('extra optional arg', 5), ('extra unknown arg', 4)]
  Bf's error where A right, Bf wrong (102): [('optional default added/dropped', 22), ('content (gold not in query)', 17), ('missing call', 12), ('content (gold in query)', 9), ('extra optional arg', 9), ('string format/case only', 8), ('extra required arg', 6), ('unparsed', 5), ('number (gold not in query)', 5), ('wrong function', 4)]

## test-clean: Bf vs B and A vs Bf by slice
call_pattern=single n=801 | B 86.3 Bf 87.6 A 94.0 | B/Bf only 25/36 p=0.2 | A/Bf only 66/15 p=8.63e-09
call_pattern=multiple n=503 | B 79.3 Bf 82.1 A 92.0 | B/Bf only 33/47 p=0.146 | A/Bf only 64/14 p=8.58e-09
call_pattern=parallel n=321 | B 81.9 Bf 86.9 A 93.1 | B/Bf only 10/26 p=0.0113 | A/Bf only 25/5 p=0.000325
distractors=0 n=624 | B 80.8 Bf 83.7 A 92.0 | B/Bf only 27/45 p=0.0444 | A/Bf only 64/12 p=1e-09
distractors=1-2 n=607 | B 84.0 Bf 86.2 A 93.6 | B/Bf only 32/45 p=0.171 | A/Bf only 61/16 p=2.42e-07
distractors=3+ n=394 | B 86.0 Bf 88.6 A 94.7 | B/Bf only 9/19 p=0.0872 | A/Bf only 30/6 p=6.96e-05
argument_source=from_query n=1261 | B 88.4 Bf 89.8 A 96.8 | B/Bf only 47/65 p=0.108 | A/Bf only 99/11 p=7.37e-19
argument_source=not_in_query n=364 | B 65.4 Bf 71.7 A 80.8 | B/Bf only 21/44 p=0.00592 | A/Bf only 56/23 p=0.000264
  B's error where Bf right, B wrong: [('missing call', 17), ('content (gold in query)', 15), ('extra optional arg', 15), ('optional default added/dropped', 14), ('extra required arg', 9), ('extra unknown arg', 8), ('string format/case only', 7), ('number (gold not in query)', 7)]
  Bf's error where A right, Bf wrong (155): [('optional default added/dropped', 48), ('extra optional arg', 36), ('content (gold not in query)', 15), ('content (gold in query)', 14), ('missing call', 13), ('missing optional arg', 9), ('extra required arg', 7), ('wrong function', 4), ('number (gold not in query)', 3), ('string format/case only', 3)]

## validation: Bf's training coverage of the functions gold calls (train rows calling the same function names)
A right, Bf wrong | n=102 | mean share of those train rows Bf keeps 48.9 | median train rows 46 | no train row 0
both right | n=745 | mean share of those train rows Bf keeps 74.2 | median train rows 74 | no train row 0
both wrong | n=132 | mean share of those train rows Bf keeps 50.4 | median train rows 42 | no train row 0
Bf right, A wrong | n=21 | mean share of those train rows Bf keeps 57.7 | median train rows 75 | no train row 0

## test-clean: Bf's training coverage of the functions gold calls (train rows calling the same function names)
A right, Bf wrong | n=155 | mean share of those train rows Bf keeps 58.3 | median train rows 38 | no train row 0
both right | n=1360 | mean share of those train rows Bf keeps 74.5 | median train rows 69 | no train row 0
both wrong | n=76 | mean share of those train rows Bf keeps 54.3 | median train rows 43 | no train row 0
Bf right, A wrong | n=34 | mean share of those train rows Bf keeps 62.7 | median train rows 87 | no train row 0
