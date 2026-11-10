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


def _run(model: str, cap: int, seconds: float) -> dict[str, Any]:
    return {
        "model": model,
        "max_num_seqs": cap,
        "requests": 1000,
        "seconds": seconds,
        "output_tokens": 50_000,
        "latency_seconds": [0.4, 0.5, 0.6, 0.5, 0.9],
        "weights_gib": 1.43,
        "kv_cache_tokens": 362496,
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
    assert data["speedup"] == pytest.approx(2.5)
    assert data["clears_2x"] is True
    md = render(data)
    assert "2.5x cheaper" in md
    assert "| Qwen/Qwen3.5-2B | 16 |" in md
