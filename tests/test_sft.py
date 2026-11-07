from types import SimpleNamespace

from fc_distill.data import to_messages
from fc_distill.scoring import score
from fc_distill.sft import ComputeLogger, cost_summary, sft_record
from fc_distill.tracking import GpuClock
from tests.factories import make_example
from tests.fakes import FakeTracker


def test_record_prompts_like_the_eval_and_completes_with_the_gold_calls() -> None:
    ex = make_example(1, answers=[{"name": "f", "arguments": {"x": 5}}])

    record = sft_record(ex)

    assert record["prompt"] == to_messages(ex)
    [completion] = record["completion"]
    assert completion["role"] == "assistant"
    assert completion["content"] == '[{"name": "f", "arguments": {"x": 5}}]'
    assert score(completion["content"], ex["answers"])["exact"]


def test_record_turns_thinking_off() -> None:
    assert sft_record(make_example(1))["chat_template_kwargs"] == {"enable_thinking": False}


def test_cost_summary_reports_the_issue_36_numbers() -> None:
    cost = cost_summary(rows=2000, tokens=1_800_000, seconds=3600.0, peak_bytes=6 * 2**30)

    assert cost == {
        "rows": 2000,
        "tokens": 1_800_000,
        "gpu_hours": 1.0,
        "tokens_per_s": 500.0,
        "gpu_hours_per_1k_rows": 0.5,
        "peak_gpu_mem_gb": 6.0,
    }


def test_compute_logger_logs_step_time_throughput_and_memory_on_logging_steps() -> None:
    ticks = iter([0.0, 1800.0, 2000.0, 3800.0])  # two optimizer steps of 30 min, 200 s apart
    tracker = FakeTracker()
    logger = ComputeLogger(tracker, GpuClock(now=lambda: next(ticks)), peak_bytes=lambda: 3 * 2**30)

    for step, tokens, should_log in [(1, 900_000, False), (2, 1_800_000, True)]:
        state = SimpleNamespace(global_step=step, num_input_tokens_seen=tokens)
        control = SimpleNamespace(should_log=should_log)
        logger.on_step_begin(None, state, control)
        logger.on_step_end(None, state, control)

    assert tracker.history == [
        {
            "compute/student_gpu_hours": 1.0,
            "compute/teacher_gpu_hours": 0.0,
            "compute/gpu_hours": 1.0,
            "compute/tokens_per_s": 500.0,
            "compute/peak_gpu_mem_gb": 3.0,
            # HF's W&B callback steps every metric on train/global_step: without it, the row would
            # plot at the previous logging step.
            "train/global_step": 2,
        }
    ]
