"""Row-by-row comparison of predictions.jsonl files on the same rows: strict exact for each,
and for each pair, rows whose text differs and rows only one gets right (#69).

usage: compare.py name=path.jsonl [name=path.jsonl ...]
"""

import itertools
import json
import sys

runs = {}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    with open(path) as f:
        runs[name] = {r["id"]: r for r in map(json.loads, f)}
ids = sorted(set.intersection(*(set(r) for r in runs.values())))
print(f"{len(ids)} common rows")
for name, r in runs.items():
    print(f"  {name:12} exact {sum(bool(r[i]['exact']) for i in ids)}")
for (na, a), (nb, b) in itertools.combinations(runs.items(), 2):
    text = sum(a[i]["prediction"].strip() != b[i]["prediction"].strip() for i in ids)
    only_a = sum(bool(a[i]["exact"]) and not b[i]["exact"] for i in ids)
    only_b = sum(bool(b[i]["exact"]) and not a[i]["exact"] for i in ids)
    print(
        f"  {na} vs {nb}: text differs {text}, only {na} right {only_a}, only {nb} right {only_b}"
    )
