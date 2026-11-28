"""Benchmark the WSP worker's scorer: the old per-row loop against the vectorised one (2026-11-21).

Run ``wsp_cb092cedad97`` (a 4,257,928-token x 4096 fp16 practice set) spent about three minutes in
every observation tool. This script builds a synthetic practice memmap of the same shape per token
(``--tokens`` x ``--d-model`` fp16, rows of ~127 tokens as in that bundle), then times:

- ``old``: ``tests/support/wsp_old_scorer.OldScorer`` (the scorer before the fix), all rows;
- ``new-cold``: the new scorer, page cache dropped for the file first (``POSIX_FADV_DONTNEED``),
  so every page comes from disk, as on a worker whose 6 GiB cannot hold a 35 GB bundle;
- ``new-warm``: a fresh scorer (empty cache) with the file in the page cache: the CPU cost;
- ``new-cached``: the same head again (the cache hit a repeated tool call gets);
- ``tool-old`` / ``tool-new``: ``practice_metrics`` (the bar on the train split, then the split's
  rows: the two scorings one tool makes) through each scorer, warm.

Emulate a pod's CPU limit with ``taskset``: ``taskset -c 0,1 python ... --workers 2``. Run from
``backend/`` with ``PYTHONPATH=src:.``. Prints one JSON object. The data file is kept between runs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from tests.support.wsp_old_scorer import OldScorer

from wsp_worker import observation_tools as ot
from wsp_worker.scoring import Scorer

MEAN_ROW_TOKENS = 4_257_928 / 33_550  # the incident bundle's tokens per row


def build(path: Path, tokens: int, d_model: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lengths = np.maximum(1, rng.exponential(MEAN_ROW_TOKENS, size=int(tokens / 100)).astype(int))
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    offsets = offsets[offsets <= tokens]
    offsets[-1] = tokens
    expected = tokens * d_model * 2
    if not path.exists() or path.stat().st_size != expected:
        mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(tokens, d_model))
        shift = rng.normal(size=d_model).astype(np.float32) * 3
        for start in range(0, tokens, 8192):
            n = min(8192, tokens - start)
            block = rng.standard_normal((n, d_model), dtype=np.float32) * 2 + shift
            mm[start : start + n] = block.astype(np.float16)
        mm.flush()
        del mm
    return offsets.astype(np.int64)


def drop_cache(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def head(d_model: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 1)
    std = rng.uniform(1.5, 2.5, size=d_model)
    std[5] = 0.0
    return {
        "weight": rng.normal(size=d_model) * 0.05,
        "bias": np.array([0.1]),
        "mean": rng.normal(size=d_model) * 3,
        "std": std,
        "attention_query": rng.normal(size=d_model) * 0.02,
    }


def timed(fn: Any) -> tuple[float, Any]:
    t = time.perf_counter()
    out = fn()
    return time.perf_counter() - t, out


def view_for(scorer: Any, rows: int) -> ot.PracticeView:
    validation = np.zeros(rows, dtype=bool)
    validation[::5] = True
    labels = (np.arange(rows) % 3 == 0).astype(np.int8)
    return ot.PracticeView(scorer, labels, validation, np.zeros(rows, bool), target_fpr=0.01)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=500_000)
    p.add_argument("--d-model", type=int, default=4096)
    p.add_argument("--rule", default="max")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--skip-old", action="store_true")
    args = p.parse_args()
    offsets = build(args.data, args.tokens, args.d_model, 0)
    rows = int(offsets.size - 1)
    function = {"kind": "linear_probe/v1", "linear_probe": {"rule": args.rule, "params": {}}}
    h = head(args.d_model, 0)
    every = np.arange(rows)

    def memmap() -> np.ndarray:
        return np.memmap(args.data, dtype=np.float16, mode="r", shape=(args.tokens, args.d_model))

    out: dict[str, Any] = {
        "tokens": args.tokens,
        "d_model": args.d_model,
        "rows": rows,
        "rule": args.rule,
        "workers": args.workers,
        "affinity": len(os.sched_getaffinity(0)),
    }
    drop_cache(args.data)
    cold = Scorer(function, memmap(), offsets, workers=args.workers)
    out["new_cold_s"], new_scores = timed(lambda: cold.score(h, every))
    warm = Scorer(function, memmap(), offsets, workers=args.workers)
    out["new_warm_s"], _ = timed(lambda: warm.score(h, every))
    out["new_cached_s"], _ = timed(lambda: warm.score({k: v.copy() for k, v in h.items()}, every))
    fresh = Scorer(function, memmap(), offsets, workers=args.workers)
    ctx_new = ot.Context(view_for(fresh, rows), lambda: dict(h), dict(h), seed=1)
    out["tool_new_s"], _ = timed(lambda: ot.practice_metrics(ctx_new, split="validation"))
    out["tool_new_cached_s"], _ = timed(lambda: ot.practice_metrics(ctx_new, split="validation"))
    if not args.skip_old:
        old = OldScorer(function, memmap(), offsets)
        out["old_s"], old_scores = timed(lambda: old.score(h, every))
        out["max_abs_diff"] = float(np.abs(old_scores - new_scores).max())
        ctx_old = ot.Context(view_for(old, rows), lambda: dict(h), dict(h), seed=1)
        out["tool_old_s"], _ = timed(lambda: ot.practice_metrics(ctx_old, split="validation"))
    scale = 4_257_928 / args.tokens
    out["scaled_to_incident"] = {
        k: round(v * scale, 2) for k, v in out.items() if k.endswith("_s") and "cached" not in k
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
