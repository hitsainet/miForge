"""The worker's own ``linear_probe/v1`` scorer (feature 016; FR-016.18, FR-016.23; FTDD §6.2, §6.6).

The worker imports nothing from ``wsp`` (FTDD §6.1's fence), so the observation tools and program
mode's ``SupportSet`` score rows with this restatement of ``wsp/weight_ops/evaluator.py``: the same
standardisation (a degenerate channel, ``|std| <= 1e-6``, is zeroed, never divided by a floor),
the same six rules and a float32 row score. A unit test pins it to the backend's evaluator, and to
the scorer it replaced (``tests/support/wsp_old_scorer.py``), on every rule at the parity
tolerance (``unit/wsp_worker/test_worker_scoring.py``), so the two cannot drift.

**Vectorised (fix 2026-11-21, run ``wsp_cb092cedad97``).** The scorer it replaced looped in Python
per row, copied every token to float64 and standardised per row: each observation tool took about
three minutes on a 4.26M-token practice set, so a 40-minute session made 18 calls. Now:

- the standardisation is folded into the head once per call, in float64:
  ``(z - mean) / std . w + b = z . w' + (b - mean . w')`` with ``w' = w / std`` and ``w'_j = 0``
  on a degenerate channel (the same zero the old ``np.where`` gave), and likewise for the
  attention query;
- the per-token scores (and the attention logits) are float64 BLAS products over small sub-blocks
  of tokens converted into a reused buffer (``SUB_TOKENS``), over threads when the worker has more
  than one CPU (``workers``), and stored as float32. **float64, not float32**: on the benchmark
  (2026-11-21) the fp16 to float64 copy plus ``dgemv`` ran as fast as the float32 path, while a
  float32 product of a folded head lost up to 1e-4 on scores near 10 when channels carry a large
  mean (cancellation), ten times the bundle's parity tolerance;
- the per-token vector is cached per head digest (:data:`CACHE_ENTRIES`, :data:`CACHE_MAX_BYTES`),
  filled lazily for exactly the rows asked for, so a second tool call on unchanged weights reads
  no activation at all;
- the rules pool those token scores per row from the offsets (``np.add.reduceat``,
  ``np.maximum.reduceat``, a segment softmax); ``rolling_mean_max`` runs :func:`combine` per row
  over the cached float64 values.

**Every activation read goes through :meth:`Scorer.read_rows`** (rows ``first..stop-1`` as a
float32 copy; ``tokens`` is one row of it). Nothing else touches the memory map, so an
instrumented scorer that overrides it records every practice row a tool touches
(``test_observation_tools.py``), and a test that replaces it proves every score flows through it.

Standard library and numpy only. A failure raises :class:`ScoringError`; nothing here reads a file
except ``/sys/fs/cgroup/cpu.max`` (the CPU quota, for the thread count).
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

RULES = ("mean", "max", "last", "softmax", "attention", "rolling_mean_max")
EPS = 1e-6
DEFAULT_TAU = 1.0
DEFAULT_WINDOW = 16
#: Rows are scored in blocks of at most this many tokens (one ``read_rows`` call each, and the
#: unit a thread takes): a block is a view of the map, so its size costs no memory.
BLOCK_TOKENS = 65_536
#: Tokens converted to float64 per BLAS call: 64 x 4096 x 8 B = 2 MiB, inside a core's L2
#: (benchmark 2026-11-21: 2.2 GB/s of fp16 a core at 32-64 tokens, 1.5 at 1,024, 1.3 at 4,096).
SUB_TOKENS = 64
#: Cached heads (a per-token float32 vector, plus the attention logits when the head has a
#: query): 4.26M tokens is 17 MB a vector, so the bound below is far above four entries there.
CACHE_ENTRIES = 4
CACHE_MAX_BYTES = 512 * 1024 * 1024
#: The most threads one scorer starts, whatever the CPU quota says.
MAX_WORKERS = 8
#: The tensors a ``linear_probe/v1`` head is scored with: the cache key's whole input.
HEAD_TENSORS = ("weight", "bias", "mean", "std", "attention_query")
CPU_MAX = Path("/sys/fs/cgroup/cpu.max")


class ScoringError(ValueError):
    """The function or tensors cannot be scored (an unknown rule, a missing head)."""


def _vec(tensors: Mapping[str, np.ndarray], name: str) -> np.ndarray | None:
    value = tensors.get(name)
    return None if value is None else np.asarray(value, dtype=np.float64).reshape(-1)


def _softmax(logits: np.ndarray) -> np.ndarray:
    weights: np.ndarray = np.exp(logits - logits.max())
    out: np.ndarray = weights / weights.sum()
    return out


def combine(
    rule: str,
    scores: np.ndarray,
    *,
    attention_logits: np.ndarray | None,
    tau: float,
    window: int,
) -> float:
    if scores.size == 0:
        raise ScoringError("a practice row has no tokens, so it has no score")
    if rule == "mean":
        return float(scores.mean())
    if rule == "max":
        return float(scores.max())
    if rule == "last":
        return float(scores[-1])
    if rule == "softmax":
        if tau <= 0:
            raise ScoringError(f"softmax tau must be > 0, got {tau}")
        return float((_softmax(scores / tau) * scores).sum())
    if rule == "attention":
        if attention_logits is None:
            raise ScoringError("the attention rule needs an 'attention_query' tensor")
        return float((_softmax(attention_logits) * scores).sum())
    if rule == "rolling_mean_max":
        if window < 1:
            raise ScoringError(f"window must be >= 1, got {window}")
        if scores.size <= window:
            return float(scores.mean())
        cumulative = np.concatenate([[0.0], np.cumsum(scores)])
        return float(((cumulative[window:] - cumulative[:-window]) / window).max())
    raise ScoringError(f"unknown rule {rule!r}")


def available_cpus() -> int:
    """The CPUs this process may use: its affinity, capped by the cgroup's quota (a pod's
    ``limits.cpu``), at most :data:`MAX_WORKERS`."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    try:
        quota, period = CPU_MAX.read_text().split()[:2]
        if quota != "max":
            n = min(n, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, min(n, MAX_WORKERS))


@dataclass(frozen=True)
class Folded:
    """A head with its standardisation folded in: ``token scores = z @ matrix[:, 0] + offset[0]``
    and, with a query, ``logits = z @ matrix[:, 1] + offset[1]``."""

    matrix: np.ndarray  # float64 (d_model, 1 or 2)
    offset: np.ndarray  # float64 (1 or 2,)

    @property
    def has_query(self) -> bool:
        return int(self.matrix.shape[1]) == 2


def fold(tensors: Mapping[str, np.ndarray]) -> Folded:
    """``(z - mean) / std . v = z . (v / std) - mean . (v / std)``, a degenerate channel's term
    zero exactly as the per-row ``np.where`` made it, for the weight and the attention query."""
    weight, bias = _vec(tensors, "weight"), _vec(tensors, "bias")
    if weight is None or bias is None or bias.size != 1:
        raise ScoringError("a linear probe needs a 1-D 'weight' and a [1] 'bias' tensor")
    mean, std, query = _vec(tensors, "mean"), _vec(tensors, "std"), _vec(tensors, "attention_query")
    for name, v in (("mean", mean), ("std", std), ("attention_query", query)):
        if v is not None and v.size != weight.size:
            raise ScoringError(f"'{name}' has {v.size} entries but 'weight' has {weight.size}")
    columns = [weight] if query is None else [weight, query]
    if std is not None:
        degenerate = np.abs(std) <= EPS
        scale = np.where(degenerate, 0.0, 1.0 / np.where(degenerate, 1.0, std))
        columns = [c * scale for c in columns]
    offsets = [float(bias[0]), 0.0][: len(columns)]
    if mean is not None:
        offsets = [o - float(mean @ c) for o, c in zip(offsets, columns, strict=True)]
    matrix = np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float64)
    return Folded(matrix, np.asarray(offsets, dtype=np.float64))


def head_digest(tensors: Mapping[str, np.ndarray]) -> str:
    """The cache key: every tensor the scorer reads, by name, shape and float64 bits."""
    h = hashlib.sha256()
    for name in HEAD_TENSORS:
        v = tensors.get(name)
        if v is None:
            h.update(f"{name}:absent;".encode())
            continue
        a = np.ascontiguousarray(np.asarray(v, dtype=np.float64))
        h.update(f"{name}:{a.shape};".encode())
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass
class _Entry:
    """One head's per-token scores (and logits), valid for the rows marked ``done``. One spare
    slot at the end, so ``reduceat`` may index a row that ends at the last token."""

    scores: np.ndarray
    logits: np.ndarray | None
    done: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(
            self.scores.nbytes
            + (0 if self.logits is None else self.logits.nbytes)
            + self.done.nbytes
        )


class Scorer:
    """Row scores for one function over the practice activations, for any set of head tensors."""

    def __init__(
        self,
        function: Mapping[str, Any],
        activations: np.ndarray,
        offsets: np.ndarray,
        *,
        workers: int | None = None,
    ) -> None:
        if function.get("kind") != "linear_probe/v1":
            raise ScoringError(f"function kind {function.get('kind')!r} is not scored in phase 1")
        spec = function["linear_probe"]
        params = spec.get("params") or {}
        self.rule = str(spec["rule"])
        self.tau = DEFAULT_TAU if params.get("tau") is None else float(params["tau"])
        self.window = DEFAULT_WINDOW if params.get("window") is None else int(params["window"])
        self._activations = activations
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.d_model = int(activations.shape[-1])
        self.workers = available_cpus() if workers is None else max(1, int(workers))
        self._cache: OrderedDict[str, _Entry] = OrderedDict()

    @property
    def rows(self) -> int:
        return int(self.offsets.size) - 1

    # --------------------------------------------------------------------- the one read

    def read_rows(self, first: int, stop: int) -> np.ndarray:
        """Rows ``first..stop-1``'s token activations, as a read-only view of the map (pages are
        read as the scorer converts them). THE ONLY READ of the activations: every other method
        comes through here, and only :meth:`tokens` hands values out, as a copy."""
        if not 0 <= first < stop <= self.rows:
            raise ScoringError(
                f"rows {first}..{stop - 1} are outside the {self.rows} practice rows"
            )
        start, end = int(self.offsets[first]), int(self.offsets[stop])
        view: np.ndarray = self._activations[start:end]
        return view

    def tokens(self, row: int) -> np.ndarray:
        """One row's token activations, as float64 (a copy: the caller cannot reach the memmap)."""
        if not 0 <= row < self.rows:
            raise ScoringError(f"row {row} is outside the {self.rows} practice rows")
        return np.array(self.read_rows(int(row), int(row) + 1), dtype=np.float64)

    # --------------------------------------------------------------------- token scores

    def _entry(self, tensors: Mapping[str, np.ndarray], rows: np.ndarray) -> _Entry:
        """The cached token scores for this head, computed for any of ``rows`` not yet done."""
        weight = _vec(tensors, "weight")
        if weight is not None and weight.size != self.d_model:
            raise ScoringError(f"activations are d_model={self.d_model}, the head {weight.size}")
        key = head_digest(tensors)
        entry = self._cache.get(key)
        folded: Folded | None = None
        if entry is None:
            folded = fold(tensors)
            tokens = int(self.offsets[-1])
            entry = _Entry(
                np.zeros(tokens + 1, dtype=np.float32),
                np.zeros(tokens + 1, dtype=np.float32) if folded.has_query else None,
                np.zeros(self.rows, dtype=bool),
            )
            self._cache[key] = entry
            self._evict()
        else:
            self._cache.move_to_end(key)
        missing = np.unique(rows[~entry.done[rows]])
        if missing.size:
            self._fill(entry, folded or fold(tensors), missing)
        return entry

    def _evict(self) -> None:
        while len(self._cache) > 1 and (
            len(self._cache) > CACHE_ENTRIES
            or sum(e.nbytes for e in self._cache.values()) > CACHE_MAX_BYTES
        ):
            self._cache.popitem(last=False)

    def _blocks(self, missing: np.ndarray) -> list[tuple[int, int]]:
        """``missing`` (sorted, unique) as runs of consecutive rows, each cut into blocks of at
        most ``BLOCK_TOKENS`` tokens (a row longer than that is a block of its own)."""
        budget = BLOCK_TOKENS
        breaks = np.flatnonzero(np.diff(missing) != 1) + 1
        blocks: list[tuple[int, int]] = []
        for run in np.split(missing, breaks):
            first, stop = int(run[0]), int(run[-1]) + 1
            while first < stop:
                limit = int(self.offsets[first]) + budget
                cut = int(np.searchsorted(self.offsets, limit, side="right")) - 1
                cut = min(max(cut, first + 1), stop)
                blocks.append((first, cut))
                first = cut
        return blocks

    def _fill(self, entry: _Entry, folded: Folded, missing: np.ndarray) -> None:
        def one(block: tuple[int, int]) -> None:
            first, stop = block
            z = self.read_rows(first, stop)
            start = int(self.offsets[first])
            buffer = np.empty((min(SUB_TOKENS, z.shape[0]), self.d_model), dtype=np.float64)
            for at in range(0, z.shape[0], SUB_TOKENS):
                part = z[at : at + SUB_TOKENS]
                n = part.shape[0]
                np.copyto(buffer[:n], part)
                out = buffer[:n] @ folded.matrix  # one BLAS call: the scores and the logits
                entry.scores[start + at : start + at + n] = out[:, 0] + folded.offset[0]
                if entry.logits is not None:
                    entry.logits[start + at : start + at + n] = out[:, 1] + folded.offset[1]

        blocks = self._blocks(missing)
        if self.workers > 1 and len(blocks) > 1:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(blocks))) as pool:
                list(pool.map(one, blocks))
        else:
            for block in blocks:
                one(block)
        entry.done[missing] = True

    def token_scores(self, tensors: Mapping[str, np.ndarray], row: int) -> np.ndarray:
        """One row's per-token scores, as float64."""
        if not 0 <= row < self.rows:
            raise ScoringError(f"row {row} is outside the {self.rows} practice rows")
        entry = self._entry(tensors, np.asarray([row], dtype=np.int64))
        start, end = int(self.offsets[row]), int(self.offsets[row + 1])
        return entry.scores[start:end].astype(np.float64)

    # --------------------------------------------------------------------- row scores

    def score(
        self, tensors: Mapping[str, np.ndarray], rows: Sequence[int] | np.ndarray
    ) -> np.ndarray:
        """float32 row scores for ``rows`` (any order, repeats allowed)."""
        chosen = np.asarray(rows, dtype=np.int64).reshape(-1)
        if chosen.size == 0:
            return np.empty(0, dtype=np.float32)
        if chosen.min() < 0 or chosen.max() >= self.rows:
            bad = int(chosen[(chosen < 0) | (chosen >= self.rows)][0])
            raise ScoringError(f"row {bad} is outside the {self.rows} practice rows")
        if self.rule not in RULES:
            raise ScoringError(f"unknown rule {self.rule!r}")
        starts, ends = self.offsets[chosen], self.offsets[chosen + 1]
        lengths = ends - starts
        if (lengths <= 0).any():
            raise ScoringError("a practice row has no tokens, so it has no score")
        if self.rule == "attention" and _vec(tensors, "attention_query") is None:
            raise ScoringError("the attention rule needs an 'attention_query' tensor")
        entry = self._entry(tensors, chosen)
        pooled = pool(
            self.rule, entry.scores, entry.logits, starts, ends, tau=self.tau, window=self.window
        )
        return pooled.astype(np.float32)


def pool(
    rule: str,
    scores: np.ndarray,
    logits: np.ndarray | None,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    tau: float,
    window: int,
) -> np.ndarray:
    """Each row ``[starts[i], ends[i])``'s score under ``rule``, in float64. ``scores`` has a spare
    slot after its last token, so an ``ends`` equal to the token count is a valid index."""
    lengths = (ends - starts).astype(np.float64)
    pairs = np.empty(2 * starts.size, dtype=np.intp)
    pairs[0::2], pairs[1::2] = starts, ends
    if rule == "mean":
        out: np.ndarray = np.add.reduceat(scores, pairs, dtype=np.float64)[0::2] / lengths
        return out
    if rule == "max":
        return np.maximum.reduceat(scores, pairs)[0::2].astype(np.float64)
    if rule == "last":
        last: np.ndarray = scores[ends - 1].astype(np.float64)
        return last
    if rule == "rolling_mean_max":
        if window < 1:
            raise ScoringError(f"window must be >= 1, got {window}")
        return np.fromiter(
            (
                combine(
                    rule,
                    scores[s:e].astype(np.float64),
                    attention_logits=None,
                    tau=tau,
                    window=window,
                )
                for s, e in zip(starts.tolist(), ends.tolist(), strict=True)
            ),
            dtype=np.float64,
            count=starts.size,
        )
    # softmax and attention: a segment softmax over the rows' tokens, gathered contiguously.
    if rule == "softmax" and tau <= 0:
        raise ScoringError(f"softmax tau must be > 0, got {tau}")
    if rule == "attention" and logits is None:
        raise ScoringError("the attention rule needs an 'attention_query' tensor")
    counts = (ends - starts).astype(np.int64)
    local = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.intp)
    gather = np.repeat(starts - local, counts) + np.arange(int(counts.sum()))
    values = scores[gather].astype(np.float64)
    if rule == "softmax":
        weights_from = values / tau
    else:
        assert logits is not None
        weights_from = logits[gather].astype(np.float64)
    row_of = np.repeat(np.arange(starts.size), counts)
    peak = np.maximum.reduceat(weights_from, local)
    e = np.exp(weights_from - peak[row_of])
    result: np.ndarray = np.add.reduceat(e * values, local) / np.add.reduceat(e, local)
    return result


# ----------------------------------------------------------------------------- metrics


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney AUROC with ties counted half; ``None`` with only one class present."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    pos, neg = s[y], s[~y]
    if pos.size == 0 or neg.size == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))


def bar_at(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float | None:
    """The practice-cut bar: the lowest threshold whose false-alarm rate on the negatives is at
    most ``target_fpr`` (a row fires when its score is strictly above the bar)."""
    neg = np.sort(np.asarray(scores, dtype=np.float64)[~np.asarray(labels).astype(bool)])
    if neg.size == 0:
        return None
    allowed = int(np.floor(target_fpr * neg.size))
    return float(neg[neg.size - 1 - allowed]) if allowed < neg.size else float(neg[0]) - 1.0


def rates(scores: np.ndarray, labels: np.ndarray, bar: float | None) -> dict[str, float | None]:
    """Recall and false-alarm rate of ``score > bar``."""
    if bar is None:
        return {"recall": None, "false_alarm_rate": None}
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    fired = s > bar
    return {
        "recall": None if not y.any() else float(fired[y].mean()),
        "false_alarm_rate": None if y.all() else float(fired[~y].mean()),
    }
