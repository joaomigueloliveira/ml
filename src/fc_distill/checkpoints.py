"""LoRA checkpoints of a training run, as fc-eval scores them on validation (ADR 0006).

HF's Trainer saves one PEFT adapter per `--save-steps` under runs/<train run>/checkpoint-<step>/.
"""

import json
import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from fc_distill.tracking import ENTITY, GPU_HOURS, PROJECT, RUN_ID_FILE

# HF's W&B callback steps every training row on it; ComputeLogger's rows carry it too.
STEP = "train/global_step"


class Checkpoint(NamedTuple):
    """One saved adapter: the base model it applies to, its LoRA rank, and where in which
    training run it was saved."""

    path: Path
    base: str
    rank: int
    train_run: str
    step: int


def read_checkpoint(path: Path) -> Checkpoint:
    adapter = json.loads((path / "adapter_config.json").read_text())
    state = json.loads((path / "trainer_state.json").read_text())
    return Checkpoint(
        path=path,
        base=adapter["base_model_name_or_path"],
        rank=adapter["r"],
        train_run=path.parent.name,
        step=state["global_step"],
    )


def gpu_hours_at_step(history: Iterable[Mapping[str, Any]], step: int) -> tuple[int, float]:
    """The training run's `compute/gpu_hours` when it saved the checkpoint at `step`, from its
    logged rows, with the step that row was logged at. Rows come every `--logging-steps`, so a
    final step between two of them takes the last row before it."""
    logged = [
        (r[STEP], r[GPU_HOURS])
        for r in history
        if r.get(STEP) is not None and r.get(GPU_HOURS) is not None and r[STEP] <= step
    ]
    if not logged:
        raise ValueError(f"no GPU-hours logged at or before step {step}")
    return max(logged)


def training_history(train_run: Path) -> list[dict[str, Any]]:
    """The training run's logged step and GPU-hours rows, read back from W&B (network)."""
    import wandb

    run_id = (train_run / RUN_ID_FILE).read_text().strip()
    entity = os.environ.get("WANDB_ENTITY", ENTITY)
    project = os.environ.get("WANDB_PROJECT", PROJECT)
    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    return list(run.scan_history(keys=[STEP, GPU_HOURS]))


# PEFT names the adapter's weights after the text-only model fc-sft trains
# (`model.layers.N...`). vLLM serves the vision-language checkpoint, whose text layers sit under
# `model.language_model.` and map to `language_model.model.`: unrenamed, the weights match no
# module and vLLM serves the base model as if no adapter were loaded.
_PEFT_TEXT = "base_model.model.model."
_PEFT_VL = "base_model.model.model.language_model."


def vllm_adapter(ckpt: Checkpoint, out: Path) -> Path:
    """A copy of the checkpoint's adapter with its weights renamed for vLLM's Qwen3.5."""
    from safetensors.torch import load_file, save_file

    weights = load_file(str(ckpt.path / "adapter_model.safetensors"))
    renamed = {
        k.replace(_PEFT_TEXT, _PEFT_VL, 1) if k.startswith(_PEFT_TEXT) else k: v
        for k, v in weights.items()
    }
    out.mkdir(parents=True, exist_ok=True)
    save_file(renamed, str(out / "adapter_model.safetensors"))
    shutil.copy(ckpt.path / "adapter_config.json", out / "adapter_config.json")
    return out
