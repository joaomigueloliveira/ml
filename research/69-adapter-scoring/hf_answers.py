"""Greedy answers on all of validation with HF generate and the adapter loaded through PEFT,
as fc-opd's probe produces them, scored with strict exact (#69).

usage: hf_answers.py <checkpoint-dir> <out.jsonl>
"""

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from fc_distill.data import load_split
from fc_distill.opd import greedy_answers
from fc_distill.scoring import score

ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
base = json.loads((ckpt / "adapter_config.json").read_text())["base_model_name_or_path"]
rows = sorted(load_split("validation"), key=lambda ex: ex["id"])
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to("cuda")
model = PeftModel.from_pretrained(model, str(ckpt))

start = time.perf_counter()
texts = greedy_answers(model, tokenizer, rows, max_tokens=1024)
print(f"{len(rows)} rows in {time.perf_counter() - start:.0f} s")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for ex, text in zip(rows, texts, strict=True):
        exact = score(text, ex["answers"], ex["tools"])["exact"]
        f.write(json.dumps({"id": ex["id"], "prediction": text, "exact": exact}) + "\n")
print(
    f"strict exact {sum(score(t, ex['answers'], ex['tools'])['exact'] for t, ex in zip(texts, rows))}"
)
