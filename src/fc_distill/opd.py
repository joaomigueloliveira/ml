"""Method D: on-policy distillation of the student from the teacher (fc-opd), ADR 0008.

The student samples its own answers to train prompts and the teacher grades every token: the loss
is the reverse KL, KL(student || teacher), over the vocabulary at each answer position. TRL's
DistillationTrainer does the sampling, the teacher's forward pass and the chunked loss.

uv run --no-sync fc-opd --n-rows 256 --save-steps 8      # smoke run: memory, engine, throughput
uv run --no-sync fc-opd --save-steps 500 --wandb          # method D: one epoch over train (#54)

Starts from the raw student with a fresh LoRA and the ADR 0006 optimiser settings; trains on
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


class OpdRecord(TypedDict):
    """One training row: the eval's prompt and nothing else. D is label-free (ADR 0007)."""

    prompt: list[Message]


def opd_record(example: Example) -> OpdRecord:
    return {"prompt": to_messages(example)}


def rollout_stats(texts: Sequence[str], lengths: Sequence[int]) -> dict[str, float]:
    """Health of one batch of the student's samples, for W&B: the share that parse as calls (by
    the eval's parser, so a fenced answer counts) and the mean completion length in tokens."""
    parsed = sum(parse_calls(t) is not None for t in texts)
    return {
        "rollout/parse_ok": parsed / len(texts),
        "rollout/completion_tokens": sum(lengths) / len(lengths),
    }


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
    """A text-only HF parameter name as vLLM's vision-language Qwen3.5 class loads it. TRL starts
    vLLM from the checkpoint, whose class is the VL one, and pushes the weights of our text-only
    student: `model.layers.0...` must arrive as `model.language_model.layers.0...`. `lm_head` and
    names already prefixed pass through."""
    if name.startswith("model.") and not name.startswith("model.language_model."):
        return "model.language_model." + name.removeprefix("model.")
    return name


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
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
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
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None, help="default: runs/<run name>")
    ap.add_argument("--wandb", action="store_true", help="log the run to Weights & Biases")
    ap.add_argument(
        "--allow-dirty", action="store_true", help="run even with uncommitted changes (recorded)"
    )
    args = ap.parse_args()
    if not args.allow_dirty and git_state()["dirty"]:
        ap.error("uncommitted changes to tracked files; commit them or pass --allow-dirty")
    return args


def greedy_answers(
    model: Any, tokenizer: Any, rows: Sequence[Example], batch: int = 16
) -> list[str]:
    """The student's greedy answers with HF generate, for the probe: independent of the sampling
    engine, whose temperature is the training one."""
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
            out = model.generate(**enc, do_sample=False, max_new_tokens=1024, use_cache=True)
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
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DistillationConfig, DistillationTrainer

    rows = load_split("train")
    if args.n_rows is not None:
        rows = random.Random(args.seed).sample(rows, args.n_rows)
    records = [opd_record(ex) for ex in rows]
    probe = probe_rows(load_split("validation")) if args.probe else []
    variant = f"r{args.lora_r}-b{args.batch_size}x{args.grad_accum}-n{len(records)}" + (
        "" if args.vllm else "-hf"
    )
    name = f"train-opd-{variant}-s{args.seed}"
    out: Path = args.out or Path("runs") / name
    effective_batch = args.batch_size * args.grad_accum

    tracker = start_run(enabled=args.wandb, job_type="train", group="opd", run_dir=out, name=name)
    tracker.config(
        {
            "exp": {
                "method": "opd",
                "variant": variant,
                "student": args.model,
                "teacher": args.teacher,
                "seed": args.seed,
                "lora": {
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
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
                    "engine": "vllm-colocate" if args.vllm else "hf-generate",
                    "temperature": args.temperature,
                    "max_completion_length": args.max_completion_length,
                },
                "data": {"source": "prompts", "split": "train", "n_rows": len(records)},
                "probe_rows": len(probe),
            },
            "provenance": train_provenance(
                base=args.model,
                base_revision=hub_revision(args.model, args.revision),
                teacher=args.teacher,
                teacher_revision=hub_revision(args.teacher, args.teacher_revision),
            ),
        }
    )

    # Text-only causal LMs for both: the checkpoints are vision-language, the towers stay unloaded.
    student = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, revision=args.teacher_revision, dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    clock = GpuClock()
    compute = type("ComputeCallback", (ComputeLogger, TrainerCallback), {})(
        tracker, clock, torch.cuda.max_memory_reserved
    )

    class ProbeCallback(TrainerCallback):  # type: ignore[misc]
        """At each checkpoint: greedy answers on the probe rows, strict exact, to W&B. Runs after
        the step's timing closes, so it isn't counted in GPU-hours."""

        def on_save(self, args_: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if not probe:
                return
            exact = probe_exact(greedy_answers(kwargs["model"], tokenizer, probe), probe)
            print(f"probe, step {state.global_step}: exact {exact:.3f} on {len(probe)} rows")
            tracker.log(
                {
                    "probe/validation_exact": exact,
                    "train/global_step": state.global_step,
                    GPU_HOURS: clock.metrics()[GPU_HOURS],
                }
            )

    class OpdTrainer(DistillationTrainer):  # type: ignore[misc]
        """DistillationTrainer plus the rollout health signals (ADR 0008)."""

        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, **kw)
            self._rollouts: tuple[list[str], list[int]] = ([], [])
            if getattr(self, "vllm_generation", None) is not None:
                fix = self.vllm_generation._fix_param_name_to_vllm
                self.vllm_generation._fix_param_name_to_vllm = lambda n, **k: vllm_vl_name(
                    fix(n, **k)
                )

        def _generate_and_score_completions(self, inputs: Any) -> Any:
            output = super()._generate_and_score_completions(inputs)
            texts, lengths = self._rollouts
            texts += self.processing_class.batch_decode(
                output["completion_ids"], skip_special_tokens=True
            )
            lengths += output["completion_mask"].sum(dim=1).tolist()
            return output

        def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
            texts, lengths = self._rollouts
            if texts and "loss" in logs:
                logs = logs | rollout_stats(texts, lengths)
                self._rollouts = ([], [])
            super().log(logs, start_time)

    trainer = OpdTrainer(
        model=student,
        teacher_model=teacher,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list([dict(r) for r in records]),
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
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
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    trainer.save_model(str(out / "adapter"))

    cost = cost_summary(
        rows=len(records),
        tokens=trainer.state.num_input_tokens_seen,
        seconds=clock.metrics()[STUDENT_GPU_HOURS] * 3600,
        peak_bytes=torch.cuda.max_memory_reserved(),
    )
    final = trainer.state.log_history[-1]
    report = {
        "name": name,
        "engine": "vllm-colocate" if args.vllm else "hf-generate",
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
