"""Method C's teacher pass (ADR 0010): the 2B's top-k log-probs at every answer position of B's
training rows, scored by vLLM on the very token ids fc-sft trains on.

uv run --extra inference --extra train fc-teacher-topk --teacher-run runs/eval-train-Qwen_Qwen3.5-2B
uv run --no-sync fc-teacher-topk --teacher-run runs/eval-train-Qwen_Qwen3.5-2B --n-rows 200  # smoke

The rows are method B's (every teacher output that parses, rewritten in gold's format) and are
tokenized by TRL's own dataset preparation, as fc-sft's SFTTrainer does it, so the stored
positions are the positions the student's loss covers. vLLM's `prompt_logprobs` scores each
position of the given ids; the answer positions (those with a label) keep their top k and the
written token's log-prob and rank. Writes topk.npz and summary.json to runs/<name>/. The file is
derived from the gated xLAM prompts: back it up only to the private W&B project.

Needs both extras: TRL tokenizes, vLLM scores.
"""

import argparse
import json
import os
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from fc_distill.data import load_split
from fc_distill.evaluate import GPU_MEM, MAX_MODEL_LEN
from fc_distill.sft import STUDENT, SftRecord, load_teacher_run, teacher_rows
from fc_distill.tracking import data_provenance, git_state, hub_revision

K = 20
IGNORE = -100  # TRL's label on positions the loss skips
# fc-sft's --max-length: no train row is longer, and TRL truncates to it.
MAX_LENGTH = 4096


def answer_positions(labels: Sequence[int]) -> list[int]:
    """The positions the loss covers: those with a label (TRL masks the prompt with -100)."""
    return [i for i, label in enumerate(labels) if label != IGNORE]


class Position(NamedTuple):
    """The teacher's top k at one position, most likely first, and the written token's own
    log-prob and rank (1 = the teacher's top token)."""

    ids: list[int]
    logprobs: list[float]
    token_logprob: float
    token_rank: int


def top_k_entries(
    *, token_ids: Sequence[int], logprobs: Sequence[float], ranks: Sequence[int], token: int, k: int
) -> Position:
    """One position of vLLM's prompt log-probs: the k entries ranked 1 to k, and the written
    token's own log-prob and rank (vLLM adds it as an extra entry when it ranks below k)."""
    by_id = {tid: (rank, lp) for tid, lp, rank in zip(token_ids, logprobs, ranks, strict=True)}
    if token not in by_id:
        raise ValueError(f"written token {token} has no log-prob at this position")
    token_rank, token_logprob = by_id[token]
    top = sorted((rank, tid, lp) for tid, (rank, lp) in by_id.items() if rank <= k)[:k]
    return Position(
        ids=[tid for _, tid, _ in top],
        logprobs=[lp for _, _, lp in top],
        token_logprob=token_logprob,
        token_rank=token_rank,
    )


Ints = npt.NDArray[np.int32]
Floats = npt.NDArray[np.float32]


class RowTopk(NamedTuple):
    """One training row's answer positions: position in the token ids, written token, the
    teacher's top-k ids and log-probs ([n, k]), and the written token's log-prob and rank."""

    row_id: int
    positions: Ints
    tokens: Ints
    ids: Ints
    logprobs: Floats
    token_logprob: Floats
    token_rank: Ints


_FIELDS = ("positions", "tokens", "ids", "logprobs", "token_logprob", "token_rank")


def save_topk(path: Path, rows: Sequence[RowTopk]) -> None:
    """All rows in one .npz: each field concatenated over rows, with offsets into it."""
    lengths = [len(r.positions) for r in rows]
    np.savez(
        path,
        row_ids=np.array([r.row_id for r in rows], dtype=np.int64),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        **{f: np.concatenate([getattr(r, f) for r in rows]) for f in _FIELDS},
    )


def load_topk(path: Path) -> dict[int, RowTopk]:
    """`save_topk`'s rows, by row id."""
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    offsets = data["offsets"]
    return {
        int(row_id): RowTopk(int(row_id), *(data[f][offsets[i] : offsets[i + 1]] for f in _FIELDS))
        for i, row_id in enumerate(data["row_ids"])
    }


def coverage(rows: Sequence[RowTopk]) -> dict[str, float]:
    """How much the stored top k says: the share of answer positions whose written token is the
    teacher's top-1, and the probability mass the top k hold (what the tail entry stands in for)."""
    mass = np.concatenate([np.exp(r.logprobs.astype(np.float64)).sum(axis=1) for r in rows])
    ranks = np.concatenate([r.token_rank for r in rows])
    return {
        "rows": len(rows),
        "positions": int(mass.size),
        "written_is_top1": float((ranks == 1).mean()),
        "written_in_top_k": float((ranks <= rows[0].ids.shape[1]).mean()),
        "mass_median": float(np.median(mass)),
        "mass_p01": float(np.quantile(mass, 0.01)),
        "mass_min": float(mass.min()),
    }


def training_tokens(
    records: Sequence[tuple[int, SftRecord]], tokenizer: Any
) -> dict[int, tuple[list[int], list[int]]]:
    """Each row's input ids and labels exactly as fc-sft's SFTTrainer builds them: TRL's own
    dataset preparation (chat template, completion mask, truncation), keyed by row id."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    dataset = Dataset.from_list([dict(r) | {"row_id": i} for i, r in records])
    # `_prepare_dataset` reads only these three attributes of the trainer (TRL 1.17.1): the
    # template override (None unless assistant_only_loss), the loss mask and the tokenizer.
    trainer = argparse.Namespace(
        chat_template=None, completion_only_loss=True, _tokenizer=tokenizer
    )
    config = SFTConfig(
        output_dir="unused", max_length=MAX_LENGTH, completion_only_loss=True, report_to="none"
    )
    prepared = SFTTrainer._prepare_dataset(
        trainer, dataset, tokenizer, config, False, None, "train"
    )
    return {r["row_id"]: (r["input_ids"], r["labels"]) for r in prepared}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--teacher-run", type=Path, required=True, help="the fc-eval train run B used")
    ap.add_argument("--k", type=int, default=K, help="log-probs kept per position")
    ap.add_argument("--n-rows", type=int, default=None, help="random rows, seed 0 (default: all)")
    ap.add_argument("--chunk", type=int, default=2048, help="rows per vLLM call (host memory)")
    ap.add_argument("--gpu-mem", type=float, default=GPU_MEM, help="vLLM gpu_memory_utilization")
    ap.add_argument("--out", type=Path, default=None, help="default: runs/<run name>")
    ap.add_argument(
        "--allow-dirty", action="store_true", help="run even with uncommitted changes (recorded)"
    )
    args = ap.parse_args()
    if not args.allow_dirty and git_state()["dirty"]:
        ap.error("uncommitted changes to tracked files; commit them or pass --allow-dirty")
    return args


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # no nvcc here (see fc-eval)
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    teacher = load_teacher_run(args.teacher_run)
    rows = [ex for ex in load_split("train") if ex["id"] in teacher.outputs]
    kept, counts = teacher_rows(rows, teacher.outputs, rule="none")
    if args.n_rows is not None:
        kept = random.Random(0).sample(kept, args.n_rows)
    # fc-sft's tokenizer: the student's, the same file as the teacher's (ADR 0010).
    tokens = training_tokens(kept, AutoTokenizer.from_pretrained(STUDENT))
    order = [row_id for row_id, _ in kept if row_id in tokens]
    print("rows:", json.dumps(counts | {"scored": len(order)}))

    name = f"topk-{args.k}-{Path(args.teacher_run).name}" + (
        "" if args.n_rows is None else f"-n{args.n_rows}"
    )
    out: Path = args.out or Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=teacher.model,
        revision=teacher.revision,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=args.gpu_mem,
        max_logprobs=args.k,
        # Every prompt position is scored; a cached prefix would skip some of them.
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    params = SamplingParams(
        max_tokens=1, temperature=0.0, prompt_logprobs=args.k, detokenize=False, flat_logprobs=True
    )

    results: list[RowTopk] = []
    mismatched, seconds = 0, 0.0
    for start in range(0, len(order), args.chunk):
        chunk = order[start : start + args.chunk]
        prompts = [TokensPrompt(prompt_token_ids=tokens[i][0]) for i in chunk]
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        seconds += time.perf_counter() - t0
        for row_id, o in zip(chunk, outputs, strict=True):
            input_ids, labels = tokens[row_id]
            if list(o.prompt_token_ids or []) != input_ids:
                mismatched += 1
                continue
            results.append(score_row(row_id, input_ids, labels, o.prompt_logprobs, args.k))
        print(f"{start + len(chunk)} / {len(order)} rows, {seconds / 3600:.2f} GPU-h")
    if mismatched:
        raise RuntimeError(f"{mismatched} rows' token ids changed inside vLLM")

    save_topk(out / "topk.npz", results)
    summary = {
        "teacher_run": str(args.teacher_run),
        "model": teacher.model,
        "k": args.k,
        "rows": counts | {"scored": len(results), "token_mismatches": mismatched},
        "coverage": coverage(results),
        "seconds": round(seconds, 1),
        "gpu_hours": seconds / 3600,
        "file_bytes": (out / "topk.npz").stat().st_size,
        "provenance": git_state()
        | {"model_revision": hub_revision(teacher.model, teacher.revision)}
        | data_provenance()
        | {
            "vllm": vllm.__version__,
            "scoring": "prompt_logprobs on fc-sft's token ids, raw log-probs (T = 1)",
            "tokenizer": STUDENT,
            "max_length": MAX_LENGTH,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def score_row(
    row_id: int, input_ids: Sequence[int], labels: Sequence[int], logprobs: Any, k: int
) -> RowTopk:
    """One row's answer positions from vLLM's flat prompt log-probs."""
    picked = []
    for p in answer_positions(labels):
        a, b = logprobs.start_indices[p], logprobs.end_indices[p]
        picked.append(
            top_k_entries(
                token_ids=logprobs.token_ids[a:b],
                logprobs=logprobs.logprobs[a:b],
                ranks=logprobs.ranks[a:b],
                token=input_ids[p],
                k=k,
            )
        )
    return _row_topk(row_id, answer_positions(labels), input_ids, picked, k)


def _row_topk(
    row_id: int,
    positions: Sequence[int],
    input_ids: Sequence[int],
    picked: Sequence[Position],
    k: int,
) -> RowTopk:
    if any(len(p.ids) != k for p in picked):
        raise ValueError(f"row {row_id}: a position has fewer than {k} entries")
    return RowTopk(
        row_id=row_id,
        positions=np.array(positions, dtype=np.int32),
        tokens=np.array([input_ids[p] for p in positions], dtype=np.int32),
        ids=np.array([p.ids for p in picked], dtype=np.int32).reshape(-1, k),
        logprobs=np.array([p.logprobs for p in picked], dtype=np.float32).reshape(-1, k),
        token_logprob=np.array([p.token_logprob for p in picked], dtype=np.float32),
        token_rank=np.array([p.token_rank for p in picked], dtype=np.int32),
    )
