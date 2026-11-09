
## Error classes, validation (rows; coarse then fine)
model | arg key | default-only | function | unparsed | value
0.8B | 79 | 32 | 142 | 155 | 126
2B | 86 | 35 | 31 | 30 | 138
A | 32 | 1 | 10 | 4 | 106
B | 78 | 25 | 37 | 4 | 148
Bf | 52 | 23 | 28 | 10 | 121
D | 86 | 27 | 26 | 4 | 151
D500 | 86 | 26 | 27 | 18 | 143
BD | 79 | 33 | 29 | 5 | 150

fine | 0.8B | 2B | A | B | Bf | D | D500 | BD
content (gold not in query) | 38 | 48 | 40 | 57 | 52 | 57 | 55 | 60
extra optional arg | 38 | 35 | 8 | 31 | 26 | 40 | 42 | 37
missing call | 124 | 22 | 3 | 22 | 20 | 19 | 10 | 18
string format/case only | 25 | 35 | 19 | 32 | 19 | 33 | 34 | 35
unparsed | 155 | 30 | 4 | 4 | 10 | 4 | 18 | 5
optional default added/dropped | 32 | 35 | 1 | 25 | 23 | 27 | 26 | 33
content (gold in query) | 32 | 21 | 20 | 28 | 21 | 27 | 24 | 24
number (gold not in query) | 20 | 21 | 22 | 23 | 23 | 25 | 24 | 21
extra required arg | 18 | 22 | 6 | 19 | 13 | 24 | 19 | 24
missing optional arg | 10 | 10 | 14 | 14 | 9 | 7 | 13 | 10
wrong function | 13 | 5 | 4 | 10 | 8 | 5 | 11 | 7
type (str vs number/bool) | 9 | 11 | 4 | 6 | 4 | 8 | 5 | 9
missing required arg | 7 | 11 | 3 | 7 | 3 | 7 | 9 | 3
extra unknown arg | 6 | 8 | 1 | 7 | 1 | 8 | 3 | 5
extra call | 4 | 4 | 3 | 5 | 0 | 2 | 6 | 4
number (gold in query) | 2 | 2 | 1 | 2 | 2 | 1 | 1 | 1
unknown function | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0

## Error classes, test-clean (rows; coarse then fine)
model | arg key | default-only | function | unparsed | value
0.8B | 115 | 74 | 245 | 220 | 81
2B | 123 | 83 | 46 | 58 | 67
A | 37 | 24 | 7 | 3 | 39
B | 96 | 58 | 30 | 8 | 80
Bf | 84 | 62 | 22 | 3 | 60
D | 108 | 81 | 20 | 3 | 59
D500 | 117 | 72 | 30 | 7 | 84
BD | 108 | 83 | 25 | 6 | 77

fine | 0.8B | 2B | A | B | Bf | D | D500 | BD
optional default added/dropped | 74 | 83 | 24 | 58 | 62 | 81 | 72 | 83
extra optional arg | 70 | 67 | 12 | 48 | 52 | 59 | 70 | 65
missing call | 231 | 36 | 4 | 27 | 18 | 16 | 15 | 21
unparsed | 220 | 58 | 3 | 8 | 3 | 3 | 7 | 6
extra required arg | 22 | 30 | 5 | 20 | 14 | 27 | 27 | 29
content (gold not in query) | 18 | 16 | 11 | 21 | 25 | 16 | 28 | 21
content (gold in query) | 25 | 10 | 12 | 23 | 16 | 10 | 23 | 14
number (gold not in query) | 12 | 17 | 10 | 17 | 12 | 17 | 22 | 16
missing optional arg | 11 | 9 | 13 | 14 | 14 | 7 | 16 | 6
type (str vs number/bool) | 19 | 15 | 1 | 3 | 0 | 10 | 4 | 15
extra unknown arg | 7 | 15 | 0 | 9 | 0 | 11 | 0 | 8
string format/case only | 3 | 5 | 3 | 12 | 5 | 4 | 2 | 7
wrong function | 11 | 7 | 1 | 2 | 4 | 2 | 10 | 3
missing required arg | 5 | 2 | 7 | 5 | 4 | 4 | 4 | 0
number (gold in query) | 4 | 4 | 2 | 4 | 2 | 2 | 5 | 4
extra call | 3 | 2 | 2 | 1 | 0 | 2 | 5 | 1
unknown function | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0

## Error classes, bfcl (rows; coarse then fine)
model | arg key | function | unparsed | value
0.8B | 13 | 113 | 101 | 128
2B | 5 | 34 | 11 | 123
A | 23 | 22 | 8 | 102
B | 12 | 29 | 12 | 156
Bf | 16 | 47 | 16 | 113
D | 3 | 23 | 7 | 142
D500 | 5 | 36 | 25 | 147
BD | 5 | 32 | 11 | 163

fine | 0.8B | 2B | A | B | Bf | D | D500 | BD
missing call | 102 | 21 | 15 | 19 | 36 | 15 | 10 | 25
number (gold not in query) | 21 | 34 | 14 | 43 | 15 | 37 | 38 | 39
optional arg given a non-default value | 27 | 24 | 5 | 28 | 29 | 35 | 33 | 44
unparsed | 101 | 11 | 8 | 12 | 16 | 7 | 25 | 11
content (gold in query) | 28 | 17 | 22 | 24 | 16 | 17 | 26 | 10
string format/case only | 9 | 16 | 24 | 23 | 19 | 20 | 19 | 25
list/dict value | 18 | 16 | 14 | 18 | 10 | 15 | 16 | 20
content (gold not in query) | 13 | 9 | 19 | 14 | 21 | 12 | 10 | 22
missing required arg | 8 | 3 | 23 | 10 | 15 | 0 | 4 | 0
extra call | 5 | 10 | 4 | 8 | 4 | 4 | 19 | 4
number (gold in query) | 9 | 5 | 4 | 6 | 3 | 6 | 4 | 3
wrong function | 6 | 3 | 3 | 2 | 7 | 4 | 7 | 3
extra unknown arg | 5 | 2 | 0 | 2 | 1 | 3 | 1 | 5
type (str vs number/bool) | 3 | 2 | 0 | 0 | 0 | 0 | 1 | 0

## validation: rows only B right (39): D's error | rows only D right (37): B's error
content (gold in query) | D wrong (B right): 5 | B wrong (D right): 7
optional default added/dropped | D wrong (B right): 9 | B wrong (D right): 3
missing call | D wrong (B right): 5 | B wrong (D right): 5
type (str vs number/bool) | D wrong (B right): 4 | B wrong (D right): 3
content (gold not in query) | D wrong (B right): 2 | B wrong (D right): 5
string format/case only | D wrong (B right): 2 | B wrong (D right): 3
extra unknown arg | D wrong (B right): 4 | B wrong (D right): 0
extra optional arg | D wrong (B right): 4 | B wrong (D right): 0
missing optional arg | D wrong (B right): 0 | B wrong (D right): 4
unparsed | D wrong (B right): 1 | B wrong (D right): 2
wrong function | D wrong (B right): 1 | B wrong (D right): 2
missing required arg | D wrong (B right): 0 | B wrong (D right): 2
extra required arg | D wrong (B right): 2 | B wrong (D right): 0
number (gold in query) | D wrong (B right): 0 | B wrong (D right): 1

## test-clean: rows only B right (59): D's error | rows only D right (60): B's error
optional default added/dropped | D wrong (B right): 26 | B wrong (D right): 6
content (gold in query) | D wrong (B right): 2 | B wrong (D right): 12
missing call | D wrong (B right): 2 | B wrong (D right): 12
extra optional arg | D wrong (B right): 9 | B wrong (D right): 3
extra required arg | D wrong (B right): 6 | B wrong (D right): 1
content (gold not in query) | D wrong (B right): 2 | B wrong (D right): 5
number (gold not in query) | D wrong (B right): 3 | B wrong (D right): 3
string format/case only | D wrong (B right): 0 | B wrong (D right): 6
type (str vs number/bool) | D wrong (B right): 5 | B wrong (D right): 0
missing required arg | D wrong (B right): 1 | B wrong (D right): 2
unparsed | D wrong (B right): 0 | B wrong (D right): 3
extra unknown arg | D wrong (B right): 2 | B wrong (D right): 1
missing optional arg | D wrong (B right): 0 | B wrong (D right): 3
number (gold in query) | D wrong (B right): 0 | B wrong (D right): 2
extra call | D wrong (B right): 1 | B wrong (D right): 0
wrong function | D wrong (B right): 0 | B wrong (D right): 1

## bfcl: rows only B right (28): D's error | rows only D right (62): B's error
content (gold in query) | D wrong (B right): 4 | B wrong (D right): 10
optional arg given a non-default value | D wrong (B right): 9 | B wrong (D right): 5
missing call | D wrong (B right): 3 | B wrong (D right): 6
content (gold not in query) | D wrong (B right): 3 | B wrong (D right): 6
missing required arg | D wrong (B right): 0 | B wrong (D right): 8
string format/case only | D wrong (B right): 2 | B wrong (D right): 6
list/dict value | D wrong (B right): 2 | B wrong (D right): 5
number (gold not in query) | D wrong (B right): 1 | B wrong (D right): 6
unparsed | D wrong (B right): 0 | B wrong (D right): 5
number (gold in query) | D wrong (B right): 1 | B wrong (D right): 2
extra unknown arg | D wrong (B right): 1 | B wrong (D right): 1
extra call | D wrong (B right): 0 | B wrong (D right): 2
wrong function | D wrong (B right): 2 | B wrong (D right): 0

## validation: B vs D by slice (exact; only B / only D, p) and 2B
call_pattern=single | n=492 | B 80.3 | D 79.9 | 2B 78.9 | 15/13 p=0.851
call_pattern=multiple | n=324 | B 63.0 | D 62.7 | 2B 56.5 | 15/14 p=1
call_pattern=parallel | n=184 | B 59.2 | D 59.8 | 2B 59.2 | 9/10 p=1
distractors=0 | n=400 | B 64.2 | D 64.8 | 2B 62.7 | 15/17 p=0.86
distractors=1-2 | n=379 | B 71.8 | D 71.8 | 2B 67.3 | 15/15 p=1
distractors=3+ | n=221 | B 81.0 | D 79.2 | 2B 78.7 | 9/5 p=0.424
argument_source=from_query | n=707 | B 82.5 | D 82.7 | 2B 79.6 | 24/26 p=0.888
argument_source=not_in_query | n=293 | B 42.7 | D 41.3 | 2B 39.9 | 15/11 p=0.557

## test-clean: B vs D by slice (exact; only B / only D, p) and 2B
call_pattern=single | n=801 | B 86.3 | D 86.3 | 2B 83.6 | 24/24 p=1
call_pattern=multiple | n=503 | B 79.3 | D 77.5 | 2B 69.2 | 29/20 p=0.253
call_pattern=parallel | n=321 | B 81.9 | D 85.0 | 2B 71.7 | 6/16 p=0.0525
distractors=0 | n=624 | B 80.8 | D 81.2 | 2B 73.7 | 22/25 p=0.771
distractors=1-2 | n=607 | B 84.0 | D 82.9 | 2B 76.3 | 30/23 p=0.41
distractors=3+ | n=394 | B 86.0 | D 87.3 | 2B 82.5 | 7/12 p=0.359
argument_source=from_query | n=1261 | B 88.4 | D 87.9 | 2B 81.1 | 40/33 p=0.483
argument_source=not_in_query | n=364 | B 65.4 | D 67.6 | 2B 61.8 | 19/27 p=0.302

## BFCL: B vs D by category
simple n=400 | 0.8B 81.5 | 2B 87.2 | A 88.2 | B 82.5 | Bf 86.8 | D 85.8 | D500 83.8 | BD 81.2 | B/D only 10/23 p=0.0351
multiple n=200 | 0.8B 77.0 | 2B 84.5 | A 88.0 | B 82.0 | Bf 86.0 | D 88.5 | D500 88.0 | BD 83.5 | B/D only 3/16 p=0.00443
parallel n=200 | 0.8B 34.5 | 2B 77.5 | A 83.5 | B 76.0 | Bf 76.0 | D 80.0 | D500 76.5 | BD 79.0 | B/D only 4/12 p=0.0768
parallel_multiple n=200 | 0.8B 48.0 | 2B 77.0 | A 74.5 | B 72.5 | Bf 68.5 | D 72.5 | D500 61.5 | BD 69.5 | B/D only 11/11 p=1

## Output length (chars) and calls per row vs gold
validation 0.8B: mean chars 153, max 5031 | calls vs gold: {'same': 714, 'fewer': 125, 'unparsed': 155, 'more': 6}
validation 2B: mean chars 146, max 2607 | calls vs gold: {'same': 944, 'unparsed': 30, 'more': 4, 'fewer': 22}
validation A: mean chars 137, max 1310 | calls vs gold: {'same': 990, 'unparsed': 4, 'fewer': 3, 'more': 3}
validation B: mean chars 141, max 3267 | calls vs gold: {'same': 969, 'fewer': 22, 'unparsed': 4, 'more': 5}
validation Bf: mean chars 142, max 1885 | calls vs gold: {'same': 970, 'unparsed': 10, 'fewer': 20}
validation D: mean chars 145, max 5042 | calls vs gold: {'same': 975, 'fewer': 19, 'more': 2, 'unparsed': 4}
validation D500: mean chars 172, max 3565 | calls vs gold: {'same': 965, 'fewer': 10, 'unparsed': 18, 'more': 7}
validation BD: mean chars 142, max 1816 | calls vs gold: {'same': 973, 'fewer': 18, 'unparsed': 5, 'more': 4}
bfcl 0.8B: mean chars 189, max 3016 | calls vs gold: {'same': 791, 'unparsed': 101, 'more': 6, 'fewer': 102}
bfcl 2B: mean chars 215, max 3216 | calls vs gold: {'same': 958, 'more': 10, 'unparsed': 11, 'fewer': 21}
bfcl A: mean chars 187, max 851 | calls vs gold: {'same': 973, 'unparsed': 8, 'more': 4, 'fewer': 15}
bfcl B: mean chars 198, max 4008 | calls vs gold: {'same': 961, 'unparsed': 12, 'more': 8, 'fewer': 19}
bfcl Bf: mean chars 186, max 760 | calls vs gold: {'same': 944, 'unparsed': 16, 'fewer': 36, 'more': 4}
bfcl D: mean chars 204, max 1510 | calls vs gold: {'same': 974, 'unparsed': 7, 'more': 4, 'fewer': 15}
bfcl D500: mean chars 238, max 4193 | calls vs gold: {'same': 946, 'unparsed': 25, 'more': 19, 'fewer': 10}
bfcl BD: mean chars 194, max 2561 | calls vs gold: {'same': 960, 'unparsed': 11, 'more': 4, 'fewer': 25}
