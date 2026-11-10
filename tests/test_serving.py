from typing import Any

import pytest

from fc_distill.serving import read_memory, render, report

LOG = (
    "(EngineCore pid=1) INFO [model_runner.py:428] Model loading took 1.43 GiB memory and 2.3 s\n"
    "(EngineCore pid=1) INFO [kv_cache_utils.py:1147] GPU KV cache size: 362,496 tokens, "
    "Maximum concurrency for 4,096 tokens per request: 88.50x\n"
)


def test_read_memory_takes_the_weights_and_kv_cache_from_vllms_log() -> None:
    assert read_memory(LOG) == {"weights_gib": 1.43, "kv_cache_tokens": 362496}


def test_read_memory_leaves_out_what_the_log_doesnt_say() -> None:
    assert read_memory("nothing here") == {"weights_gib": None, "kv_cache_tokens": None}


def _run(
    model: str, cap: int, seconds: float, latency: tuple[float, ...] = (0.4, 0.5, 0.6, 0.5, 0.9)
) -> dict[str, Any]:
    return {
        "model": model,
        "model_revision": "abc",
        "max_num_seqs": cap,
        "requests": 1000,
        "seconds": seconds,
        "input_tokens": 500_000,
        "output_tokens": 50_000,
        "latency_seconds": list(latency),
        "weights_gib": 1.43,
        "kv_cache_tokens": 362496,
        "vllm": "0.33.0",
        "git": {"commit": "f00", "dirty": False},
    }


def test_report_compares_each_model_at_its_cheapest_setting() -> None:
    runs = [
        _run("Qwen/Qwen3.5-0.8B", 16, 40.0),
        _run("Qwen/Qwen3.5-0.8B", 256, 20.0),
        _run("Qwen/Qwen3.5-2B", 16, 90.0),
        _run("Qwen/Qwen3.5-2B", 256, 50.0),
    ]
    data = report(runs, student="Qwen/Qwen3.5-0.8B", teacher="Qwen/Qwen3.5-2B")
    # 1000 requests in 20 s: 20 GPU-seconds per 1,000 requests; the 2B's best is 50.
    student = data["models"]["Qwen/Qwen3.5-0.8B"]
    assert (student["best"]["max_num_seqs"], student["best"]["gpu_seconds_per_1k"]) == (256, 20.0)
    assert student["best"]["requests_per_s"] == 50.0
    assert student["best"]["latency_p50"] == 0.5
    # Five latencies: p90 is the largest.
    assert student["best"]["latency_p90"] == 0.9
    assert data["speedup"] == pytest.approx(2.5)
    md = render(data)
    assert "2.5x cheaper" in md
    assert "| Qwen/Qwen3.5-2B | 16 |" in md


def test_report_gives_the_ratio_one_request_at_a_time_and_per_output_token() -> None:
    runs = [
        _run("Qwen/Qwen3.5-0.8B", 256, 20.0, latency=(0.2, 0.3)),
        _run("Qwen/Qwen3.5-2B", 256, 36.0, latency=(0.5, 0.75)),
    ]
    runs[1]["output_tokens"] = 54_000
    data = report(runs)
    # Batched: 36 / 20 = 1.8. One at a time over the same prompts: 1.25 / 0.5.
    assert data["speedup"] == 1.8
    assert data["latency_speedup"] == 2.5
    # Per output token: 2,500 tokens/s against 1,500.
    assert data["per_token_speedup"] == pytest.approx(1.67)
    md = render(data)
    assert "answers 2.5x faster one request at a time" in md
    assert "serves 1.8x cheaper with every prompt batched" in md


def test_report_without_the_teacher_gives_no_ratio() -> None:
    data = report([_run("Qwen/Qwen3.5-0.8B", 256, 20.0)])
    assert (data["speedup"], data["latency_speedup"]) == (None, None)
    assert "cheaper" not in render(data)
