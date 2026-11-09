"""s4b: what the 2B's repaired outputs look like next to gold (a few, truncated)."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import examples, preds
from s4_repair import repair

EX = examples()
from collections import Counter

from taxonomy import classify_xlam

p = preds("2B", "test-clean")
c = Counter()
k = 0
for i, r in p.items():
    if r["parse_ok"]:
        continue
    f = repair(r["prediction"])
    cl = classify_xlam(f, EX[i])
    c[cl[1]] += 1
    if k < 6 and cl[0] != "unparsed":
        k += 1
        print(cl, "| repaired:", f[:110], "| gold:", str(r["gold"])[:110])
print(c)
