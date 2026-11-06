"""Greedy eval of a model with vLLM on an xLAM split (validation, test, test-clean) or on BFCL's
non-live single-turn subset (unseen tools, fc_distill.bfcl).

uv run fc-eval --model Qwen/Qwen3.5-0.8B --n 500
uv run fc-eval --model Qwen/Qwen3.5-0.8B --split bfcl
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, get_args

from fc_distill import bfcl
from fc_distill.data import EvalSplit, Message, load_split, to_messages
from fc_distill.scoring import score, summarize

SPLITS = (*get_args(EvalSplit), "bfcl")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--split", choices=SPLITS, default="validation")
    ap.add_argument("--n", type=int, default=None, help="first n examples (default: all)")
    ap.add_argument("--out", type=Path, default=None, help="default: runs/eval-<model>")
    ap.add_argument("--max-model-len", type=int, default=4096)
    # The longest validation/test gold answer is 521 tokens (notebooks/01_dataset.ipynb, section 4).
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    ap.add_argument("--quantization", default=None)
    ap.add_argument(
        "--max-num-seqs", type=int, default=None, help="cap concurrent sequences (big models)"
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # FlashInfer's sampler JIT-compiles CUDA kernels and needs nvcc, which a driver-only setup
    # lacks. Greedy decoding doesn't need it, so fall back to vLLM's built-in sampler.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # Imported here so the rest of the package (and its tests) don't need the GPU extra.
    from vllm import LLM, SamplingParams

    if args.split == "bfcl":
        bfcl_examples = bfcl.load()[: args.n]
        prompts: list[list[Message]] = [bfcl.to_messages(ex) for ex in bfcl_examples]
    else:
        examples = load_split(args.split)[: args.n]
        prompts = [to_messages(ex) for ex in examples]
    out: Path = args.out or Path("runs") / f"eval-{args.split}-{args.model.replace('/', '_')}"
    out.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        quantization=args.quantization,
        **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}),
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    start = time.perf_counter()
    outputs = llm.chat(
        prompts,
        params,
        chat_template_kwargs={"enable_thinking": False},
    )
    elapsed = time.perf_counter() - start

    texts = [o.outputs[0].text for o in outputs]
    if args.split == "bfcl":
        metrics = score_bfcl(bfcl_examples, texts, out)
    else:
        scores = []
        with (out / "predictions.jsonl").open("w") as f:
            for ex, text in zip(examples, texts, strict=True):
                s = score(text, ex["answers"])
                scores.append(s)
                record = {"id": ex["id"], "prediction": text, "gold": ex["answers"], **s}
                f.write(json.dumps(record) + "\n")
        metrics = dict(summarize(scores))

    summary = metrics | {
        "model": args.model,
        "split": args.split,
        "seconds": round(elapsed, 1),
        "output_tokens": sum(len(o.outputs[0].token_ids) for o in outputs),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def score_bfcl(examples: list[bfcl.BfclExample], texts: list[str], out: Path) -> dict[str, Any]:
    """Score BFCL predictions, write predictions.jsonl, and summarise per slice and category,
    with the rows whose gold calls a function name that also appears in xLAM train."""
    scores = []
    with (out / "predictions.jsonl").open("w") as f:
        for ex, text in zip(examples, texts, strict=True):
            s = bfcl.score(text, ex["possible"])
            scores.append(s)
            record = {"id": ex["id"], "category": ex["category"], "prediction": text}
            f.write(json.dumps(record | {"gold": ex["possible"], **s}) + "\n")
    train_names = {t["name"] for ex in load_split("train") for t in ex["tools"]}
    seen = bfcl.seen_in_train(examples, train_names)
    return bfcl.summarize(examples, scores) | {"rows_calling_a_train_function_name": len(seen)}


if __name__ == "__main__":
    main()
