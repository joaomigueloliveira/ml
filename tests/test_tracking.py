import sys
from pathlib import Path

import pytest

from fc_distill import tracking


def test_logging_off_never_imports_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)  # any `import wandb` now raises
    tracker = tracking.start_run(enabled=False, job_type="eval", group="g", run_dir=tmp_path)
    tracker.config({"a": 1})
    tracker.log({"val/exact": 0.5})
    tracker.summary({"exact": 0.5})
    tracker.table("slices", ["slice"], [["single"]])
    tracker.finish()
    assert list(tmp_path.iterdir()) == []


def test_wandb_mode_disabled_overrides_an_enabled_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    tracker = tracking.start_run(
        enabled=True, job_type="train", group="g", run_dir=tmp_path, name="n", tags=("t",)
    )
    tracker.config({"provenance": {"dirty": False}})
    tracker.log({"val/exact": 0.5, "compute/gpu_hours": 0.1})
    tracker.summary({"exact": 0.5})
    tracker.table("slices", ["slice", "exact"], [["single", 0.5]])
    tracker.finish()
    assert list(tmp_path.iterdir()) == []


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_gpu_clock_adds_teacher_compute_to_the_x_axis() -> None:
    time = FakeTime()
    clock = tracking.GpuClock(now=time)
    with clock.measure("student"):
        time.now += 2 * 3600
    with clock.measure("teacher"):
        time.now += 0.5 * 3600
    time.now += 3600  # untimed work (checkpoint saves) is not counted
    clock.add_teacher_hours(0.25)  # this run's share of an offline teacher job
    assert clock.metrics() == {
        "compute/student_gpu_hours": 2.0,
        "compute/teacher_gpu_hours": 0.75,
        "compute/gpu_hours": 2.75,
    }


def test_gpu_clock_resumes_from_saved_state() -> None:
    time = FakeTime()
    clock = tracking.GpuClock(now=time)
    with clock.measure("student"):
        time.now += 3600
    resumed = tracking.GpuClock.from_state(clock.state(), now=time)
    with resumed.measure("student"):
        time.now += 3600
    assert resumed.metrics()["compute/gpu_hours"] == 2.0


def test_train_provenance_names_code_data_and_both_models() -> None:
    prov = tracking.train_provenance(
        base="Qwen/Qwen3.5-0.8B",
        base_revision="s1",
        teacher="Qwen/Qwen3.5-2B",
        teacher_revision="t1",
    )
    assert isinstance(prov["dirty"], bool)
    assert len(prov["commit"]) == 40
    assert (prov["base_model"], prov["base_revision"]) == ("Qwen/Qwen3.5-0.8B", "s1")
    assert (prov["teacher_model"], prov["teacher_revision"]) == ("Qwen/Qwen3.5-2B", "t1")
    assert {"xlam_revision", "splits_sha256", "prompt_sha256"} <= prov.keys()


def test_train_provenance_without_a_teacher() -> None:
    prov = tracking.train_provenance(base="Qwen/Qwen3.5-0.8B", base_revision="s1")
    assert prov["teacher_model"] is None
    assert prov["teacher_revision"] is None
