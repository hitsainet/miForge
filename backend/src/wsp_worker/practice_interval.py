"""Is a practice AUROC difference real? The verdict's paired bootstrap, on the practice rows
(plan A1, 2026-11-22; D128 round 9).

Run ``wsp_54b6c1e5f0c9`` kept an edit for +0.0002 AUROC on practice validation, inside the noise
([-0.00012, +0.00053]), and it was significantly worse held out. The editor had no way to know:
the tools gave point figures only, and the verdict counts a difference only when its interval
excludes zero. This is that interval: stratified resamples, the same counts for both heads,
percentiles 2.5 and 97.5 (``wsp.verdict``'s, restated because the worker imports nothing from
``wsp``; pinned equal by ``test_practice_interval.py``). Seeded from the bundle, so a replay
answers bit for bit. On D128 (4,982 rows) 2,000 resamples take about 0.5 s.

It guards against reading noise as a gain once. It does not correct for many looks at the same
rows, which the judging section says in words.

Standard library and numpy only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

RESAMPLES = 2000
#: Resamples held in memory at once.
BATCH = 100
CI_LOW, CI_HIGH = 2.5, 97.5


def _counts(rng: np.random.Generator, pos: np.ndarray, neg: np.ndarray, n: int) -> np.ndarray:
    draws = np.concatenate(
        [
            pos[rng.integers(0, pos.size, size=(BATCH, pos.size))],
            neg[rng.integers(0, neg.size, size=(BATCH, neg.size))],
        ],
        axis=1,
    )
    flat = draws + (np.arange(BATCH)[:, None] * n)
    counts = np.bincount(flat.reshape(-1), minlength=BATCH * n)
    out: np.ndarray = counts.reshape(BATCH, n).astype(np.float64)
    return out


class _Ranked:
    """One head's order and tie groups, for the weighted AUROC of any count matrix."""

    def __init__(self, scores: np.ndarray, labels: np.ndarray) -> None:
        s = np.asarray(scores, dtype=np.float64)
        self.order = np.argsort(s, kind="stable")
        ordered = s[self.order]
        first = np.concatenate([[True], ordered[1:] != ordered[:-1]])
        self.starts = np.flatnonzero(first)
        self.group = np.cumsum(first) - 1
        self.pos = labels[self.order]

    def auroc(self, counts: np.ndarray) -> np.ndarray:
        w = counts[:, self.order]
        wp, wn = w * self.pos, w * ~self.pos
        by_group = np.add.reduceat(wn, self.starts, axis=1)
        below = np.cumsum(by_group, axis=1) - by_group
        score = (wp * (below[:, self.group] + 0.5 * by_group[:, self.group])).sum(axis=1)
        out: np.ndarray = score / (wp.sum(axis=1) * wn.sum(axis=1))
        return out


def against_parent(
    after: np.ndarray, before: np.ndarray, labels: np.ndarray, seed: int
) -> dict[str, Any] | None:
    """``after``'s AUROC minus ``before``'s on the same rows, with its interval; ``None`` without
    both classes."""
    y = np.asarray(labels).astype(bool)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    if pos.size == 0 or neg.size == 0:
        return None
    now, then = _Ranked(after, y), _Ranked(before, y)
    whole = np.ones((1, y.size), dtype=np.float64)
    delta = float(now.auroc(whole)[0] - then.auroc(whole)[0])
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(RESAMPLES // BATCH):
        counts = _counts(rng, pos, neg, y.size)
        deltas.append(now.auroc(counts) - then.auroc(counts))
    low, high = (float(x) for x in np.percentile(np.concatenate(deltas), [CI_LOW, CI_HIGH]))
    return {
        "delta": delta,
        "low": low,
        "high": high,
        "confidence": (CI_HIGH - CI_LOW) / 100,
        "resamples": RESAMPLES,
        "within_noise": bool(low <= 0.0 <= high),
    }
