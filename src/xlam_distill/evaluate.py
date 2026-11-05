"""Greedy eval of a model on the held-out xLAM split with vLLM.

uv run xlam-eval --model Qwen/Qwen3.5-0.8B --n 500
"""

import argparse
import json
import time
from pathlib import Path

from xlam_distill.data import load_split, to_messages
from xlam_distill.scoring import score, summarize


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--n", type=int, default=None, help="first n eval examples (default: all)")
    ap.add_argument("--out", type=Path, default=None, help="default: runs/eval-<model>")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    ap.add_argument("--quantization", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # Imported here so the rest of the package (and its tests) don't need the GPU extra.
    from vllm import LLM, SamplingParams

    examples = load_split("eval")[: args.n]
    out: Path = args.out or Path("runs") / ("eval-" + args.model.replace("/", "_"))
    out.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        quantization=args.quantization,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    start = time.perf_counter()
    outputs = llm.chat(
        [to_messages(ex) for ex in examples],
        params,
        chat_template_kwargs={"enable_thinking": False},
    )
    elapsed = time.perf_counter() - start

    scores = []
    with (out / "predictions.jsonl").open("w") as f:
        for ex, o in zip(examples, outputs, strict=True):
            text = o.outputs[0].text
            s = score(text, ex["answers"])
            scores.append(s)
            record = {"id": ex["id"], "prediction": text, "gold": ex["answers"], **s}
            f.write(json.dumps(record) + "\n")

    summary = summarize(scores) | {
        "model": args.model,
        "seconds": round(elapsed, 1),
        "output_tokens": sum(len(o.outputs[0].token_ids) for o in outputs),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
