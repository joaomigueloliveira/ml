
## validation: B -> B->D
step | outputs changed | right->wrong | wrong->right | of changed rows: new output = 2B's | old (B) output = 2B's | lenient r->w / w->r
100 | 157 | 45 | 14 | 41 | 31 | 29/14
200 | 157 | 37 | 19 | 41 | 30 | 19/19
300 | 167 | 41 | 22 | 50 | 29 | 24/23
400 | 159 | 30 | 26 | 45 | 25 | 23/27
  step 400 B->D's error on right->wrong rows: [('optional default added/dropped', 12), ('extra optional arg', 3), ('type (str vs number/bool)', 3), ('extra required arg', 3), ('string format/case only', 2), ('content (gold in query)', 2)]
  step 400 B's error on wrong->right rows: [('missing call', 4), ('content (gold in query)', 3), ('string format/case only', 3), ('content (gold not in query)', 3), ('missing optional arg', 3), ('number (gold not in query)', 2)]
  step 400 right->wrong rows where B->D now outputs the 2B's answer: 13 of 30
500 | 175 | 40 | 23 | 46 | 36 | 25/25
1000 | 191 | 50 | 26 | 54 | 40 | 33/26
  step 1000 B->D's error on right->wrong rows: [('optional default added/dropped', 22), ('type (str vs number/bool)', 7), ('extra optional arg', 7), ('content (gold in query)', 3), ('extra required arg', 2), ('missing optional arg', 2)]
  step 1000 B's error on wrong->right rows: [('content (gold not in query)', 4), ('string format/case only', 4), ('missing call', 4), ('content (gold in query)', 4), ('number (gold not in query)', 2), ('unparsed', 2)]
  step 1000 right->wrong rows where B->D now outputs the 2B's answer: 19 of 50

## test-clean: B -> B->D
step | outputs changed | right->wrong | wrong->right | of changed rows: new output = 2B's | old (B) output = 2B's | lenient r->w / w->r
400 | 198 | 70 | 43 | 68 | 32 | 47/45
  step 400 B->D's error on right->wrong rows: [('optional default added/dropped', 29), ('extra optional arg', 16), ('type (str vs number/bool)', 9), ('extra required arg', 7), ('content (gold in query)', 3), ('content (gold not in query)', 2)]
  step 400 B's error on wrong->right rows: [('content (gold in query)', 11), ('optional default added/dropped', 5), ('missing optional arg', 5), ('missing call', 5), ('string format/case only', 4), ('extra optional arg', 3)]
  step 400 right->wrong rows where B->D now outputs the 2B's answer: 31 of 70

## D over training, validation: agreement with the 2B and exact
D 500 | exact 70.0 | lenient 72.6 | agree with 2B 71.1 | classes {'value': 143, 'right': 700, 'function': 27, 'unparsed': 18, 'default-only': 26, 'arg key': 86}
D 1000 | exact 65.7 | lenient 69.9 | agree with 2B 73.0 | classes {'value': 151, 'right': 657, 'function': 58, 'unparsed': 12, 'default-only': 42, 'arg key': 80}
D 1500 | exact 70.3 | lenient 73.9 | agree with 2B 75.1 | classes {'value': 151, 'right': 703, 'unparsed': 3, 'default-only': 36, 'function': 26, 'arg key': 81}
D 2000 | exact 69.1 | lenient 72.9 | agree with 2B 76.5 | classes {'value': 161, 'right': 691, 'function': 24, 'unparsed': 3, 'default-only': 38, 'arg key': 83}
D 2500 | exact 69.9 | lenient 73.0 | agree with 2B 76.6 | classes {'value': 148, 'right': 699, 'function': 34, 'default-only': 31, 'arg key': 84, 'unparsed': 4}
D 3000 | exact 70.6 | lenient 73.3 | agree with 2B 77.3 | classes {'value': 151, 'right': 706, 'function': 26, 'default-only': 27, 'arg key': 86, 'unparsed': 4}
D 3410 | exact 70.6 | lenient 73.7 | agree with 2B 77.6 | classes {'value': 149, 'right': 706, 'function': 27, 'default-only': 31, 'arg key': 83, 'unparsed': 4}
B 500 | exact 62.5 | agree with 2B 65.7
B 1000 | exact 70.0 | agree with 2B 72.4
B 1500 | exact 70.8 | agree with 2B 76.5
B 2000 | exact 69.8 | agree with 2B 78.2
B 2500 | exact 70.0 | agree with 2B 78.8
B 3000 | exact 69.9 | agree with 2B 79.8
B 3277 | exact 70.1 | agree with 2B 79.7

## D step 500 -> 1000: right->wrong 76, wrong->right 33; recovered by 1500: 42
  step-1000 error on those rows: [('missing call', 26), ('optional default added/dropped', 12), ('content (gold in query)', 7), ('extra required arg', 6), ('unparsed', 5), ('string format/case only', 5), ('type (str vs number/bool)', 4), ('extra optional arg', 4)]
  all step-1000 errors: [('content (gold not in query)', 54), ('missing call', 52), ('optional default added/dropped', 42), ('extra optional arg', 39), ('string format/case only', 30), ('number (gold not in query)', 28), ('content (gold in query)', 28), ('extra required arg', 23)]
  all step-1500 errors: [('content (gold not in query)', 60), ('extra optional arg', 38), ('optional default added/dropped', 36), ('string format/case only', 34), ('number (gold not in query)', 28), ('content (gold in query)', 25), ('extra required arg', 25), ('missing call', 16)]
