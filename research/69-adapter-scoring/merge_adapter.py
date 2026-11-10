"""Write a copy of the base model with a LoRA checkpoint folded into its weights:
W += (alpha / r) * B @ A, for every adapted module. fc-eval then scores the copy as a plain
model, with no adapter for vLLM to load (#69).

usage: merge_adapter.py <checkpoint-dir> <out-dir>
"""

import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
config = json.loads((ckpt / "adapter_config.json").read_text())
assert not config["use_rslora"] and not config["use_dora"]
scale = config["lora_alpha"] / config["r"]
base_dir = Path(snapshot_download(config["base_model_name_or_path"]))

lora = load_file(str(ckpt / "adapter_model.safetensors"))
out.mkdir(parents=True, exist_ok=True)
for f in base_dir.iterdir():
    if not f.name.endswith(".safetensors"):
        shutil.copy(f.resolve(), out / f.name)

merged = 0
for f in base_dir.glob("*.safetensors"):
    weights = load_file(str(f.resolve()))
    for key in [k for k in lora if ".lora_A." in k]:
        module = key.removeprefix("base_model.model.model.").split(".lora_A.")[0]
        target = f"model.language_model.{module}.weight"
        if target not in weights:
            continue
        a = lora[key].float()
        b = lora[key.replace(".lora_A.", ".lora_B.")].float()
        w = weights[target]
        weights[target] = (w.float() + scale * (b @ a)).to(w.dtype)
        merged += 1
    save_file(weights, str(out / f.name), metadata={"format": "pt"})

n_modules = sum(".lora_A." in k for k in lora)
assert merged == n_modules, f"merged {merged} of {n_modules} modules"
print(f"merged {merged} modules (scale {scale}) into {out}")
