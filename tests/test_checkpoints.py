import json
from pathlib import Path

import pytest

from fc_distill.checkpoints import gpu_hours_at_step, read_checkpoint


def _write_checkpoint(
    train_run: Path, step: int, base: str = "Qwen/Qwen3.5-0.8B", r: int = 16
) -> Path:
    """A checkpoint as HF's Trainer saves a PEFT model: adapter config and trainer state."""
    ckpt = train_run / f"checkpoint-{step}"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base, "r": r, "lora_alpha": 2 * r})
    )
    (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": 3411}))
    return ckpt


def test_checkpoint_names_its_base_model_rank_training_run_and_step(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "train-sft-gold-r16-b4x4-n54561-gl-s0", 1500, r=16)

    c = read_checkpoint(ckpt)

    assert (c.base, c.rank, c.train_run, c.step) == (
        "Qwen/Qwen3.5-0.8B",
        16,
        "train-sft-gold-r16-b4x4-n54561-gl-s0",
        1500,
    )


# The training run's W&B rows: ComputeLogger logs GPU-hours every 5 optimizer steps.
_HISTORY = [
    {"train/global_step": 495, "compute/gpu_hours": 0.55},
    {"train/global_step": 500, "compute/gpu_hours": 0.56},
    {"train/global_step": 505, "compute/gpu_hours": 0.57},
    {"train/global_step": 3410, "compute/gpu_hours": 3.80},
]


def test_gpu_hours_at_a_checkpoint_come_from_the_row_logged_at_its_step() -> None:
    assert gpu_hours_at_step(_HISTORY, 500) == (500, 0.56)


def test_a_final_step_between_logging_steps_takes_the_last_row_before_it() -> None:
    # The last step (3,411) isn't a multiple of 5: the 3,410 row is one step short.
    assert gpu_hours_at_step(_HISTORY, 3411) == (3410, 3.80)


def test_a_step_before_the_first_row_has_no_gpu_hours() -> None:
    with pytest.raises(ValueError, match="no GPU-hours"):
        gpu_hours_at_step(_HISTORY, 400)
