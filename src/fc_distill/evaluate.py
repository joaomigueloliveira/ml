"""Greedy eval of a model with vLLM on an xLAM split (validation, test, test-clean) or on BFCL's
non-live single-turn subset (unseen tools, fc_distill.bfcl).

uv run fc-eval --model Qwen/Qwen3.5-0.8B --n 500
uv run fc-eval --model Qwen/Qwen3.5-0.8B --split bfcl
uv run fc-eval --model Qwen/Qwen3.5-2B --split train --max-model-len 5120   # teacher outputs
uv run fc-eval --rescore runs/eval-test-Qwen_Qwen3.5-0.8B   # re-score saved predictions, no GPU
uv run fc-eval --adapter runs/<train run>/checkpoint-500     # a LoRA checkpoint on its base model

With --adapter, the summary records the checkpoint's step and the training run's GPU-hours at it
(read from the run's W&B rows), and the run fails if the adapter doesn't change the base model's
first outputs, which is how a silently unloaded adapter shows up.

--wandb also logs the run to Weights & Biases, as one eval run with the split's scores under
<split>/... keys (fc_distill.eval_tracking). summary.json stays the source of truth. Official
baselines are logged from results/ with `fc-report --wandb` instead.

A run refuses to start from a working tree with uncommitted changes to tracked files, so the
commit in its provenance identifies the code that produced it (--allow-dirty overrides).
"""

import argparse
import json
import os
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, get_args

from fc_distill import bfcl
from fc_distill.checkpoints import (
    STEP,
    Checkpoint,
    gpu_hours_at_step,
    read_checkpoint,
    training_history,
    vllm_adapter,
)
from fc_distill.data import (
    CHAT_TEMPLATE_KWARGS,
    EvalSplit,
    Example,
    Message,
    load_split,
    to_messages,
)
from fc_distill.eval_tracking import log_evals
from fc_distill.scoring import METRICS, score, summarize
from fc_distill.slices import slice_scores
from fc_distill.tracking import GPU_HOURS, data_provenance, git_state, hub_revision, start_run

# "train" is for teacher outputs (method B, ADR 0006): the run's predictions.jsonl is what fc-sft
# --targets teacher trains on. Its scores against gold are a by-product, never a result.
SPLITS = (*get_args(EvalSplit), "bfcl", "train")
# Rows generated again without the adapter, to check it changed something.
ADAPTER_CHECK_ROWS = 32
# A trained checkpoint changes about 70% of them; batch nondeterminism alone changes a few, so
# "not all identical" would pass an adapter that never loaded.
ADAPTER_MIN_CHANGED = 0.25


class Suite(NamedTuple):
    """How one eval split is loaded, prompted and scored."""

    load: Callable[[], Sequence[Any]]
    messages: Callable[[Any], list[Message]]
    score: Callable[[Sequence[Any], list[str], Path], dict[str, Any]]


def suite(split: str) -> Suite:
    if split == "bfcl":
        return Suite(bfcl.load, bfcl.to_messages, score_bfcl)
    return Suite(lambda: load_split(split), to_messages, score_xlam)  # type: ignore[arg-type]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", help="HF model id or local path")
    ap.add_argument("--revision", default=None, help="model revision (default: latest on main)")
    ap.add_argument(
        "--adapter",
        type=Path,
        default=None,
        metavar="CHECKPOINT_DIR",
        help="a LoRA checkpoint saved by fc-sft; --model defaults to its base model",
    )
    ap.add_argument(
        "--rescore",
        type=Path,
        metavar="RUN_DIR",
        help="re-score a saved run's predictions.jsonl and rewrite its summary (no model needed)",
    )
    ap.add_argument("--wandb", action="store_true", help="log the run to Weights & Biases")
    ap.add_argument(
        "--wandb-group", default=None, help="W&B group: 'baseline', or the training method"
    )
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
    ap.add_argument(
        "--allow-dirty", action="store_true", help="run even with uncommitted changes (recorded)"
    )
    args = ap.parse_args()
    if args.adapter is not None:
        base = read_checkpoint(args.adapter).base
        if args.model not in (None, base):
            ap.error(f"the adapter was trained on {base}, not {args.model}")
        args.model = base
    if (args.model is None) == (args.rescore is None):
        ap.error("give exactly one of --model (or --adapter) or --rescore")
    if args.model and not args.allow_dirty and git_state()["dirty"]:
        ap.error("uncommitted changes to tracked files; commit them or pass --allow-dirty")
    return args


def main() -> None:
    args = parse_args()
    if args.rescore:
        rescore(args.rescore)
        if args.wandb:
            log_to_wandb(args.rescore, args.wandb_group)
        return
    # FlashInfer's sampler JIT-compiles CUDA kernels and needs nvcc, which a driver-only setup
    # lacks. Greedy decoding doesn't need it, so fall back to vLLM's built-in sampler.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # Imported here so the rest of the package (and its tests) don't need the GPU extra.
    from vllm import LLM, SamplingParams

    s = suite(args.split)
    rows = list(s.load())[: args.n]
    ckpt = None if args.adapter is None else read_checkpoint(args.adapter)
    out: Path = args.out or Path("runs") / run_name(args.split, args.model, ckpt)
    out.mkdir(parents=True, exist_ok=True)
    lora = None
    if ckpt is not None:
        from vllm.lora.request import LoRARequest

        path = vllm_adapter(ckpt, out / "vllm-adapter")
        lora = LoRARequest(f"{ckpt.train_run}-{ckpt.step}", 1, str(path))

    llm = LLM(
        model=args.model,
        revision=args.revision,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        quantization=args.quantization,
        **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}),
        limit_mm_per_prompt={"image": 0, "video": 0},
        **({"enable_lora": True, "max_lora_rank": ckpt.rank, "max_loras": 1} if ckpt else {}),
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    start = time.perf_counter()
    prompts = [s.messages(r) for r in rows]
    outputs = llm.chat(
        prompts, params, chat_template_kwargs=CHAT_TEMPLATE_KWARGS, lora_request=lora
    )
    elapsed = time.perf_counter() - start
    texts = [o.outputs[0].text for o in outputs]
    if lora is not None:
        check = prompts[:ADAPTER_CHECK_ROWS]
        base = llm.chat(check, params, chat_template_kwargs=CHAT_TEMPLATE_KWARGS)
        changed = sum(o.outputs[0].text != t for o, t in zip(base, texts, strict=False))
        if changed < ADAPTER_MIN_CHANGED * len(check):
            raise RuntimeError(
                f"{args.adapter} changed {changed} of the first {len(check)} outputs: not loaded?"
            )

    metrics = s.score(rows, texts, out)
    run: dict[str, Any] = {
        "model": args.model,
        "split": args.split,
        "seconds": round(elapsed, 1),
        "output_tokens": sum(len(o.outputs[0].token_ids) for o in outputs),
        "provenance": provenance(args),
    }
    if ckpt is not None:
        run["adapter"] = adapter_summary(ckpt)
    write_summary(out, metrics, run)
    if args.wandb:
        log_to_wandb(out, args.wandb_group)


# Summary keys that describe the run rather than score it: the W&B config, where runs can be
# filtered by them.
CONFIG_KEYS = ("model", "split", "provenance", "rescored_by", "adapter")


def log_to_wandb(run_dir: Path, group: str | None) -> None:
    """Log a finished run's summary.json to W&B as an eval run named after its directory."""
    summary = json.loads((run_dir / "summary.json").read_text())
    tracker = start_run(
        enabled=True,
        job_type="eval",
        group=group,
        run_dir=run_dir,
        name=run_dir.name,
        tags=[f"split:{summary['split']}"],
    )
    tracker.config({k: summary[k] for k in CONFIG_KEYS if k in summary})
    log_evals(
        tracker, {summary["split"]: {k: v for k, v in summary.items() if k not in CONFIG_KEYS}}
    )
    if "adapter" in summary:
        # The x-axes of a validation-vs-GPU-hours curve across a training run's checkpoints.
        adapter = summary["adapter"]
        tracker.summary({GPU_HOURS: adapter["gpu_hours"], STEP: adapter["step"]})
    tracker.finish()


def run_name(split: str, model: str, ckpt: Checkpoint | None) -> str:
    """The eval run's directory under runs/: one per model, or per checkpoint of a training run."""
    if ckpt is None:
        return f"eval-{split}-{model.replace('/', '_')}"
    return f"eval-{split}-{ckpt.train_run}-checkpoint-{ckpt.step}"


def adapter_summary(ckpt: Checkpoint) -> dict[str, Any]:
    """Which checkpoint was scored, and the training GPU-hours it cost (ADR 0006)."""
    logged_step, hours = gpu_hours_at_step(training_history(ckpt.path.parent), ckpt.step)
    return {
        "path": str(ckpt.path),
        "train_run": ckpt.train_run,
        "step": ckpt.step,
        "rank": ckpt.rank,
        "gpu_hours": hours,
        "gpu_hours_logged_at_step": logged_step,
    }


def provenance(args: argparse.Namespace) -> dict[str, Any]:
    """What produced this run: code commit, exact model weights, data and prompt versions, and
    decoding settings."""
    import vllm

    return (
        git_state()
        | {"model_revision": hub_revision(args.model, args.revision)}
        | data_provenance()
        | {
            "bfcl_revision": bfcl.REVISION if args.split == "bfcl" else None,
            "vllm": vllm.__version__,
            "decoding": "greedy, thinking off",
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "gpu_mem": args.gpu_mem,
            "quantization": args.quantization,
        }
    )


def write_summary(out: Path, metrics: dict[str, Any], run: dict[str, Any]) -> None:
    summary = metrics | run
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def read_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A saved run's summary.json and predictions.jsonl records."""
    summary = json.loads((run_dir / "summary.json").read_text())
    with (run_dir / "predictions.jsonl").open() as f:
        records = [json.loads(line) for line in f]
    return summary, records


def rescore(run_dir: Path) -> None:
    """Score a saved run again with the current scorer: rewrites predictions.jsonl and
    summary.json. The run's own provenance is kept, and `rescored_by` records the scorer's code."""
    old, records = read_run(run_dir)
    s = suite(old["split"])
    by_id = {r["id"]: r for r in s.load()}
    rows = [by_id[r["id"]] for r in records]
    metrics = s.score(rows, [r["prediction"] for r in records], run_dir)
    keep = ("model", "split", "seconds", "output_tokens", "provenance")
    run = {k: old[k] for k in keep if k in old}
    rescored = git_state() | {"at": datetime.now(UTC).isoformat(timespec="seconds")}
    write_summary(run_dir, metrics, run | {"rescored_by": rescored})


def score_xlam(examples: Sequence[Example], texts: list[str], out: Path) -> dict[str, Any]:
    """Score xLAM predictions, write predictions.jsonl, and summarise overall and per slice."""
    scores = []
    with (out / "predictions.jsonl").open("w") as f:
        for ex, text in zip(examples, texts, strict=True):
            s = score(text, ex["answers"], ex["tools"])
            scores.append(s)
            record = {"id": ex["id"], "prediction": text, "gold": ex["answers"], **s}
            f.write(json.dumps(record) + "\n")
    return summarize(scores) | {"slices": slice_scores(examples, scores, METRICS)}


def score_bfcl(examples: Sequence[bfcl.BfclRow], texts: list[str], out: Path) -> dict[str, Any]:
    """Score BFCL predictions, write predictions.jsonl, and summarise per slice and category,
    with the rows whose gold calls a function name that some xLAM train answer calls."""
    scores = []
    with (out / "predictions.jsonl").open("w") as f:
        for ex, text in zip(examples, texts, strict=True):
            s = bfcl.score(text, ex["possible"])
            scores.append(s)
            record = {"id": ex["id"], "category": ex["category"], "prediction": text}
            f.write(json.dumps(record | {"gold": ex["possible"], **s}) + "\n")
    called = {c["name"] for ex in load_split("train") for c in ex["answers"]}
    seen = bfcl.seen_in_train(examples, called)
    return bfcl.summarize(examples, scores) | {"rows_calling_a_train_function_name": len(seen)}


if __name__ == "__main__":
    main()
