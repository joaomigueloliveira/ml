"""Method D: on-policy distillation of the student from the teacher (fc-opd), ADR 0008.

The student samples its own answers to train prompts and the teacher grades every token: the loss
is the reverse KL, KL(student || teacher), over the vocabulary at each answer position. TRL's
DistillationTrainer does the sampling, the teacher's forward pass and the chunked loss.

uv run --no-sync fc-opd --n-rows 256 --save-steps 8      # smoke run: memory, engine, throughput
uv run --no-sync fc-opd --save-steps 500 --wandb          # method D: one epoch over train (#54)
uv run --no-sync fc-opd --init-adapter runs/<B run>/checkpoint-1500 --save-steps 500 --wandb
                                                           # the B→D pipeline (#58)

Starts from the raw student with a fresh LoRA, or with --init-adapter from a trained adapter (B
then D, a pipeline, not method D), and the ADR 0006 optimiser settings; trains on
train prompts only, never gold. GPU-hours are the optimizer steps, which include the student's
sampling and the teacher's forward pass. While it runs it logs the share of rollouts that parse,
and at every checkpoint a greedy probe on 200 validation rows: watch-only, never used to stop the
run or choose the checkpoint (ADR 0008). Needs the train and inference extras.
"""

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

from fc_distill.checkpoints import PEFT_PREFIX, TEXT_PREFIX, VL_PREFIX
from fc_distill.data import CHAT_TEMPLATE_KWARGS, Example, Message, load_split, to_messages
from fc_distill.scoring import parse_calls, score
from fc_distill.sft import (
    LORA_TARGETS,
    SCHEDULE,
    STUDENT,
    WARMUP_RATIO,
    ComputeLogger,
    cost_summary,
)
from fc_distill.tracking import (
    GPU_HOURS,
    STUDENT_GPU_HOURS,
    GpuClock,
    git_state,
    hub_revision,
    start_run,
    train_provenance,
)

TEACHER = "Qwen/Qwen3.5-2B"
# Longest train prompt (3,550 tokens) plus the 1,024-token answer cap, for the vLLM engine.
VLLM_MAX_MODEL_LEN = 5120

# The validation probe: a health check during training, never a result (ADR 0008).
PROBE_ROWS = 200


class InitAdapter(TypedDict):
    """Where a run that continues an adapter starts: a run-name tag and the adapter's LoRA shape."""

    tag: str
    r: int
    alpha: int


class OpdRecord(TypedDict):
    """One training row: the eval's prompt and nothing else. D is label-free (ADR 0007)."""

    prompt: list[Message]


def opd_record(example: Example) -> OpdRecord:
    return {"prompt": to_messages(example)}


def rollout_stats(texts: Sequence[str]) -> dict[str, float]:
    """Health of the student's samples since the last log, for W&B: the share that parse as calls
    by the eval's parser, so a fenced answer counts. (TRL logs their lengths itself.)"""
    parsed = sum(parse_calls(t) is not None for t in texts)
    return {"rollout/parse_ok": parsed / len(texts)}


def probe_rows(validation: Sequence[Example], n: int = PROBE_ROWS, seed: int = 0) -> list[Example]:
    """The fixed validation subset the probe answers at every checkpoint, whatever order the
    split comes in, so probe scores are comparable along the run."""
    by_id = sorted(validation, key=lambda ex: ex["id"])
    return sorted(random.Random(seed).sample(by_id, n), key=lambda ex: ex["id"])


def probe_exact(outputs: Sequence[str], rows: Sequence[Example]) -> float:
    """Strict exact over the probe rows, each output scored against its own row's gold."""
    scores = [
        score(out, ex["answers"], ex["tools"])["exact"]
        for out, ex in zip(outputs, rows, strict=True)
    ]
    return sum(scores) / len(scores)


def vllm_vl_name(name: str) -> str:
    """A text-only HF parameter name as vLLM's vision-language Qwen3.5 class loads it: the same
    rename `checkpoints.vllm_adapter` applies to adapters, here for the full weights TRL pushes
    to its colocated engine. `lm_head` and names already prefixed pass through."""
    text, vl = TEXT_PREFIX.removeprefix(PEFT_PREFIX), VL_PREFIX.removeprefix(PEFT_PREFIX)
    if name.startswith(text) and not name.startswith(vl):
        return vl + name.removeprefix(text)
    return name


def init_adapter(checkpoint: Path, model: str) -> InitAdapter:
    """Read a trained adapter's checkpoint (runs/train-<run>/checkpoint-<step>) to continue it with
    D's loss (ADR 0008's B→D pipeline). Its rank and alpha are kept; one trained on another base
    model is refused."""
    config = json.loads((checkpoint / "adapter_config.json").read_text())
    base = config["base_model_name_or_path"]
    if base != model:
        raise ValueError(f"{checkpoint} was trained on {base}, not {model}")
    run = checkpoint.parent.name.removeprefix("train-")
    step = checkpoint.name.removeprefix("checkpoint-")
    return {"tag": f"from-{run}-{step}", "r": config["r"], "alpha": config["lora_alpha"]}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default=STUDENT, help="student: HF model id or local path")
    ap.add_argument("--revision", default=None, help="student revision (default: latest on main)")
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--teacher-revision", default=None)
    ap.add_argument("--n-rows", type=int, default=None, help="random train prompts (default: all)")
    ap.add_argument(
        "--seed", type=int, default=0, help="row sample, LoRA init, data order, sampling"
    )
    ap.add_argument("--batch-size", type=int, default=4, help="rows per micro-batch")
    ap.add_argument("--grad-accum", type=int, default=4, help="micro-batches per optimizer step")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16, help="fresh LoRA only")
    ap.add_argument("--lora-alpha", type=int, default=32, help="fresh LoRA only")
    ap.add_argument(
        "--init-adapter",
        type=Path,
        default=None,
        help="continue this adapter checkpoint instead of a fresh LoRA (B then D, ADR 0008); "
        "its rank and alpha are kept",
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-completion-length", type=int, default=1024, help="the eval's cap")
    ap.add_argument(
        "--vllm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sample with vLLM colocated in this process (sleep mode); --no-vllm: HF generate",
    )
    ap.add_argument("--vllm-gpu-memory", type=float, default=0.3, help="vLLM's share of the GPU")
    ap.add_argument(
        "--probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"greedy probe on {PROBE_ROWS} validation rows at each checkpoint (watch only)",
    )
    ap.add_argument(
        "--probe-full-at",
        type=int,
        default=1000,
        help="at this step the probe answers all of validation instead (watch only)",
    )
    ap.add_argument(
        "--longest",
        action="store_true",
        help="with --n-rows: the longest train prompts instead of random ones (memory check)",
    )
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None, help="default: runs/<run name>")
    ap.add_argument("--wandb", action="store_true", help="log the run to Weights & Biases")
    ap.add_argument(
        "--allow-dirty", action="store_true", help="run even with uncommitted changes (recorded)"
    )
    args = ap.parse_args()
    if args.probe and args.probe_full_at % args.save_steps:
        ap.error("--probe-full-at must be a multiple of --save-steps: the probe runs at saves")
    if not args.allow_dirty and git_state()["dirty"]:
        ap.error("uncommitted changes to tracked files; commit them or pass --allow-dirty")
    return args


def greedy_answers(
    model: Any, tokenizer: Any, rows: Sequence[Example], max_tokens: int, batch: int = 16
) -> list[str]:
    """The student's greedy answers with HF generate, for the probe. Not the colocated vLLM
    engine: it samples at the training temperature, and asleep between steps it holds no weights
    (ADR 0008). HF and vLLM greedy outputs can differ on a few rows."""
    import torch

    tokenizer.padding_side = "left"
    outputs: list[str] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            prompts = [
                tokenizer.apply_chat_template(
                    to_messages(ex),
                    tokenize=False,
                    add_generation_prompt=True,
                    **CHAT_TEMPLATE_KWARGS,
                )
                for ex in rows[i : i + batch]
            ]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            out = model.generate(**enc, do_sample=False, max_new_tokens=max_tokens, use_cache=True)
            outputs += tokenizer.batch_decode(
                out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
    if was_training:
        model.train()
    return outputs


def main() -> None:
    args = parse_args()
    # Imported here so the rest of the package (and its tests) don't need the GPU extras.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DistillationConfig, DistillationTrainer

    rows = load_split("train")
    if args.n_rows is not None and args.longest:
        rows = sorted(rows, key=lambda ex: len(json.dumps(to_messages(ex))))[-args.n_rows :]
    elif args.n_rows is not None:
        rows = random.Random(args.seed).sample(rows, args.n_rows)
    records = [opd_record(ex) for ex in rows]
    validation = load_split("validation")
    probe = probe_rows(validation) if args.probe else []
    start = init_adapter(args.init_adapter, args.model) if args.init_adapter else None
    lora_r, lora_alpha = (start["r"], start["alpha"]) if start else (args.lora_r, args.lora_alpha)
    variant = (
        (f"{start['tag']}-" if start else "")
        + f"r{lora_r}-b{args.batch_size}x{args.grad_accum}-n{len(records)}"
        + ("-longest" if args.longest else "")
        + ("" if args.vllm else "-hf")
    )
    name = f"train-opd-{variant}-s{args.seed}"
    out: Path = args.out or Path("runs") / name
    effective_batch = args.batch_size * args.grad_accum
    engine = "vllm-colocate" if args.vllm else "hf-generate"
    # Pinned, not just recorded: both models load at the commit the run reports.
    student_revision = hub_revision(args.model, args.revision)
    teacher_revision = hub_revision(args.teacher, args.teacher_revision)

    tracker = start_run(enabled=args.wandb, job_type="train", group="opd", run_dir=out, name=name)
    tracker.config(
        {
            "exp": {
                "method": "opd",
                "variant": variant,
                "student": args.model,
                "teacher": args.teacher,
                "seed": args.seed,
                "init_adapter": str(args.init_adapter) if args.init_adapter else None,
                "lora": {
                    "r": lora_r,
                    "alpha": lora_alpha,
                    "dropout": 0.0,
                    "target_modules": LORA_TARGETS,
                },
                "optim": {
                    "lr": args.lr,
                    "schedule": SCHEDULE,
                    "warmup_ratio": WARMUP_RATIO,
                    "effective_batch": effective_batch,
                    "micro_batch": args.batch_size,
                },
                "loss": {"divergence": "reverse_kl", "beta": 1.0},
                "sampling": {
                    "engine": engine,
                    "temperature": args.temperature,
                    "max_completion_length": args.max_completion_length,
                },
                "data": {"source": "prompts", "split": "train", "n_rows": len(records)},
                "probe": {"rows": len(probe), "full_at_step": args.probe_full_at},
            },
            "provenance": train_provenance(
                base=args.model,
                base_revision=student_revision,
                teacher=args.teacher,
                teacher_revision=teacher_revision,
            ),
        }
    )

    # Text-only causal LMs for both: the checkpoints are vision-language, the towers stay unloaded.
    student = AutoModelForCausalLM.from_pretrained(
        args.model, revision=student_revision, dtype=torch.bfloat16
    )
    if args.init_adapter:
        # A trainable PeftModel, so TRL takes it as is and adds no adapter of its own.
        student = PeftModel.from_pretrained(student, str(args.init_adapter), is_trainable=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, revision=teacher_revision, dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=student_revision)
    clock = GpuClock()
    # Device memory in use, as nvidia-smi sees it, the most seen at a logging step: after the
    # step, with vLLM asleep, so a floor on the peak during sampling (an nvidia-smi log during the
    # run gives that). PyTorch's own peak counters also count vLLM's sleep-mode pool and report
    # more than the card has.
    device_peak = [0]

    def device_used_bytes() -> int:
        free, total = torch.cuda.mem_get_info()
        device_peak[0] = max(device_peak[0], total - free)
        return device_peak[0]

    compute = type("ComputeCallback", (ComputeLogger, TrainerCallback), {})(
        tracker, clock, device_used_bytes
    )

    class ProbeCallback(TrainerCallback):  # type: ignore[misc]
        """At each checkpoint: greedy answers on the probe rows, strict exact, to W&B. Runs after
        the step's timing closes, so it isn't counted in GPU-hours."""

        def on_save(self, hf_args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if not probe:
                return
            full = state.global_step == args.probe_full_at
            asked = validation if full else probe
            answers = dict(
                zip(
                    (ex["id"] for ex in asked),
                    greedy_answers(kwargs["model"], tokenizer, asked, args.max_completion_length),
                    strict=True,
                )
            )
            # The 200-row curve gets a point at every save, the full step included.
            scores = {
                "probe/validation_exact": probe_exact([answers[ex["id"]] for ex in probe], probe)
            }
            if full:
                scores["probe/validation_exact_full"] = probe_exact(list(answers.values()), asked)
            print(f"probe, step {state.global_step}: {json.dumps(scores)}")
            tracker.log(
                scores
                | {"train/global_step": state.global_step, GPU_HOURS: clock.metrics()[GPU_HOURS]}
            )

    class OpdTrainer(DistillationTrainer):  # type: ignore[misc]
        """DistillationTrainer plus the rollout health signals (ADR 0008)."""

        def __init__(self, *args_: Any, **kwargs: Any) -> None:
            super().__init__(*args_, **kwargs)
            self._rollout_texts: list[str] = []
            engine_ = getattr(self, "vllm_generation", None)
            if engine_ is not None:
                trl_rename = engine_._fix_param_name_to_vllm
                engine_._fix_param_name_to_vllm = lambda name, **kw: vllm_vl_name(
                    trl_rename(name, **kw)
                )

        def _generate_and_score_completions(self, inputs: Any) -> Any:
            output = super()._generate_and_score_completions(inputs)
            self._rollout_texts += self.processing_class.batch_decode(
                output["completion_ids"], skip_special_tokens=True
            )
            return output

        def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
            if self._rollout_texts and "loss" in logs:
                logs = logs | rollout_stats(self._rollout_texts)
                self._rollout_texts = []
            super().log(logs, start_time)

    trainer = OpdTrainer(
        model=student,
        teacher_model=teacher,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list([dict(r) for r in records]),
        peft_config=None
        if args.init_adapter
        else LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        ),
        args=DistillationConfig(
            output_dir=str(out),
            run_name=name,
            report_to="wandb" if args.wandb else "none",
            seed=args.seed,
            num_train_epochs=1,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type=SCHEDULE,
            warmup_steps=WARMUP_RATIO,  # a float below 1 is a ratio of the total steps
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            beta=1.0,  # reverse KL (ADR 0008)
            temperature=args.temperature,
            max_completion_length=args.max_completion_length,
            chat_template_kwargs=dict(CHAT_TEMPLATE_KWARGS),
            use_vllm=args.vllm,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=args.vllm_gpu_memory,
            vllm_enable_sleep_mode=True,
            vllm_max_model_length=VLLM_MAX_MODEL_LEN,
            log_completions=True,
            num_completions_to_print=2,
        ),
        callbacks=[compute, ProbeCallback()],
    )
    trainer.train()
    trainer.save_model(str(out / "adapter"))

    cost = cost_summary(
        rows=len(records),
        tokens=trainer.state.num_input_tokens_seen,
        seconds=clock.metrics()[STUDENT_GPU_HOURS] * 3600,
        peak_bytes=device_peak[0],
    )
    final = trainer.state.log_history[-1]
    report = {
        "name": name,
        "engine": engine,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "final_train_loss": final.get("train_loss"),
        "train_runtime_s": final.get("train_runtime"),
        "cost": cost,
        "total_gpu_hours": clock.metrics()[GPU_HOURS],
    }
    (out / "cost.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    tracker.summary({f"cost/{k}": v for k, v in cost.items()})
    tracker.finish()


if __name__ == "__main__":
    main()
