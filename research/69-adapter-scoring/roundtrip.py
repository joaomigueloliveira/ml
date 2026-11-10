"""Rounding drift that one bf16 merge-then-unmerge leaves in the base weights, as TRL does around
each weight sync to the colocated vLLM engine: W' = bf16(bf16(W + d) - d), d = scale * B @ A (#69).

usage: roundtrip.py name=<checkpoint-dir> [...]
"""

import glob
import json
import sys
from pathlib import Path

from safetensors.torch import load_file

base = load_file(
    glob.glob(
        str(
            Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B"
            "/snapshots/*/model.safetensors-00001-of-00001.safetensors"
        )
    )[0]
)
for arg in sys.argv[1:]:
    name, ckpt = arg.split("=", 1)
    config = json.loads((Path(ckpt) / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    lora = load_file(str(Path(ckpt) / "adapter_model.safetensors"))
    changed = total = 0
    drift = delta = weight = 0.0
    for key in [k for k in lora if ".lora_A." in k]:
        module = key.removeprefix("base_model.model.model.").split(".lora_A.")[0]
        w = base[f"model.language_model.{module}.weight"]
        d = (scale * (lora[key.replace(".lora_A.", ".lora_B.")].float() @ lora[key].float())).to(
            w.dtype
        )
        w2 = (w + d) - d
        changed += (w2 != w).sum().item()
        total += w.numel()
        drift += (w2.float() - w.float()).norm().item() ** 2
        delta += d.float().norm().item() ** 2
        weight += w.float().norm().item() ** 2
    print(
        f"{name:10} |delta|/|W| {delta**0.5 / weight**0.5:.4f}  one round trip changes "
        f"{changed / total:.1%} of base weights, |drift|/|delta| {drift**0.5 / delta**0.5:.4f}"
    )
