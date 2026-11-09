"""s10: outputs accepted by parse_calls although they are not a bare JSON list (code fences, a bare
dict, think tags), per model and split. README says parse_ok means 'nothing around it'."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import preds

for split in ["validation", "test-clean", "bfcl"]:
    for m in ["0.8B", "2B", "A", "B", "Bf", "D", "BD"]:
        p = preds(m, split)
        fence = sum(r["parse_ok"] and "```" in r["prediction"] for r in p.values())
        bare = 0
        for r in p.values():
            if r["parse_ok"]:
                try:
                    v = json.loads(r["prediction"].strip())
                    bare += isinstance(v, dict)
                except Exception:
                    pass
        print(f"{split} {m}: parsed with a code fence {fence}, parsed as a bare dict {bare}")
