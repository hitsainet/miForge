"""The worker's scorer equals the backend's evaluator on every rule (feature 016; FR-016.18,
FR-016.23; FTDD §6.1's fence).

The worker may not import ``wsp`` (``test_wsp_fences.py``), so ``wsp_worker/scoring.py`` restates
``wsp/weight_ops/evaluator.py``; this test is what keeps the two from drifting.

Since the 2026-11-21 vectorisation (BLAS over blocks of tokens, the standardisation folded into the
head, a per-head token-score cache) the worker's scores equal the float64 references at the bundle
parity tolerance, not bit for bit (memory: goldens need cross-CPU tolerance): ``SCORE_RTOL`` and
``SCORE_ATOL`` below, against the backend's evaluator AND the old per-row scorer kept verbatim in
``tests/support/wsp_old_scorer.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.support.wsp_fixtures import RULES, make_bundle
from wsp.weight_ops.evaluator import score_rows
from wsp_worker.scoring import Scorer, ScoringError, auroc, bar_at, rates

#: The bundle parity tolerance (``wsp/fixture_bundle.py``: 1e-5), relative and absolute.
SCORE_RTOL = 1e-5
SCORE_ATOL = 1e-5


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("practice_dtype", ["float32", "float16"])
def test_the_worker_scorer_equals_the_backend_evaluator(
    tmp_path: Path, rule: str, practice_dtype: str
) -> None:
    fx = make_bundle(tmp_path / rule, rule=rule, practice_dtype=practice_dtype, window=5)
    rows = list(range(fx.manifest["practice"]["rows"]))
    ours = Scorer(fx.manifest["function"], fx.activations, fx.offsets).score(
        {n: t.values64() for n, t in fx.tensors.items()}, rows
    )
    theirs = score_rows(fx.manifest["function"], fx.tensors, fx.activations, fx.offsets, rows)
    assert ours.dtype == np.float32
    np.testing.assert_allclose(ours, theirs, rtol=SCORE_RTOL, atol=SCORE_ATOL)


def test_a_dead_channel_is_zeroed_not_divided(tmp_path: Path) -> None:
    fx = make_bundle(tmp_path, rule="mean", degenerate_channel=True)
    tensors = {n: t.values64() for n, t in fx.tensors.items()}
    scorer = Scorer(fx.manifest["function"], fx.activations, fx.offsets)
    before = scorer.score(tensors, [0, 1])
    tensors["weight"] = tensors["weight"].copy()
    tensors["weight"][-1] += 1e6  # the dead channel's weight cannot move any score
    assert np.array_equal(scorer.score(tensors, [0, 1]), before)


def test_an_unknown_rule_and_a_row_out_of_range_are_refused(tmp_path: Path) -> None:
    fx = make_bundle(tmp_path, rule="mean")
    function = {**fx.manifest["function"]}
    function["linear_probe"] = {**function["linear_probe"], "rule": "median"}
    tensors = {n: t.values64() for n, t in fx.tensors.items()}
    with pytest.raises(ScoringError):
        Scorer(function, fx.activations, fx.offsets).score(tensors, [0])
    with pytest.raises(ScoringError):
        Scorer(fx.manifest["function"], fx.activations, fx.offsets).tokens(10_000)


def test_metrics() -> None:
    scores = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == 0.75
    assert auroc(scores, np.array([1, 1, 1, 1])) is None
    bar = bar_at(scores, labels, 0.0)
    assert bar == pytest.approx(0.4)
    assert rates(scores, labels, bar) == {"recall": 0.5, "false_alarm_rate": 0.0}


# ------------------------------------------------- the vectorised scorer (fix 2026-11-21)

import wsp_worker.scoring as scoring_mod  # noqa: E402
from tests.support.wsp_old_scorer import OldScorer  # noqa: E402


def _practice(
    seed: int, *, rows: int = 60, d_model: int = 48, dtype: type = np.float16, offset: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Ragged rows (length 1, rows shorter and longer than the window), channels with a large
    shared offset (so folding the mean into the head is exercised against cancellation)."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 70, size=rows)
    lengths[0], lengths[1] = 1, 2
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    scale = rng.uniform(0.3, 4.0, size=d_model)
    shift = rng.normal(size=d_model) * offset
    acts = (rng.normal(size=(int(offsets[-1]), d_model)) * scale + shift).astype(dtype)
    return acts, offsets


def _head(
    acts: np.ndarray, seed: int, *, standardise: bool, degenerate: bool, query: bool
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 1000)
    d = acts.shape[1]
    head: dict[str, np.ndarray] = {
        "weight": rng.normal(size=d),
        "bias": np.array([rng.normal() * 0.1]),
    }
    if standardise:
        a64 = acts.astype(np.float64)
        head["mean"] = a64.mean(axis=0)
        std = a64.std(axis=0)
        if degenerate:
            std[3] = 0.0
            std[7] = 5e-7  # under EPS: zeroed too, never divided
        head["std"] = std
    if query:
        head["attention_query"] = rng.normal(size=d) * 0.3
    return head


def _function(rule: str, *, tau: float | None = None, window: int | None = None) -> dict:
    return {
        "kind": "linear_probe/v1",
        "linear_probe": {"rule": rule, "params": {"tau": tau, "window": window}},
    }


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize(
    ("standardise", "degenerate"), [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_golden_new_equals_old_on_random_data(
    rule: str, standardise: bool, degenerate: bool, dtype: type
) -> None:
    """Old (per-row float64) against new (blocked float32, folded, pooled), every rule, with and
    without mean/std, with degenerate std channels, on rows in random order with repeats."""
    acts, offsets = _practice(11, dtype=dtype)
    head = _head(acts, 11, standardise=standardise, degenerate=degenerate, query=True)
    function = _function(rule, tau=0.7 if rule == "softmax" else None, window=5)
    rng = np.random.default_rng(5)
    rows = np.concatenate([rng.permutation(offsets.size - 1), [0, 0, 7]])
    old = OldScorer(function, acts, offsets).score(head, rows)
    new = Scorer(function, acts, offsets).score(head, rows)
    assert new.dtype == np.float32 and new.shape == old.shape
    np.testing.assert_allclose(new, old, rtol=SCORE_RTOL, atol=SCORE_ATOL)
    if degenerate:
        # A dead channel's weight still cannot move any score (zeroed, never divided).
        moved = {**head, "weight": head["weight"].copy()}
        moved["weight"][3] += 1e6
        moved["weight"][7] += 1e6
        np.testing.assert_array_equal(Scorer(function, acts, offsets).score(moved, rows), new)


@pytest.mark.parametrize("rule", RULES)
def test_blocks_and_threads_give_the_same_scores(
    rule: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    acts, offsets = _practice(3, dtype=np.float16)
    head = _head(acts, 3, standardise=True, degenerate=True, query=True)
    function = _function(rule, window=4)
    rows = np.arange(offsets.size - 1)
    one = Scorer(function, acts, offsets, workers=1).score(head, rows)
    # Blocks of ~100 tokens (several per run of rows) over three threads.
    monkeypatch.setattr(scoring_mod, "BLOCK_TOKENS", 100)
    monkeypatch.setattr(scoring_mod, "SUB_TOKENS", 7)
    many = Scorer(function, acts, offsets, workers=3)
    assert len(many._blocks(rows)) > 5
    # BLAS may take another kernel for another block shape: equal to the float32 last bits.
    np.testing.assert_allclose(many.score(head, rows), one, rtol=SCORE_RTOL, atol=SCORE_ATOL)


class _CountingScorer(Scorer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.reads: list[tuple[int, int]] = []

    def read_rows(self, first: int, stop: int) -> np.ndarray:
        self.reads.append((first, stop))
        return super().read_rows(first, stop)


def test_a_second_call_on_unchanged_weights_reads_nothing() -> None:
    acts, offsets = _practice(4)
    head = _head(acts, 4, standardise=True, degenerate=False, query=False)
    scorer = _CountingScorer(_function("mean"), acts, offsets)
    rows = np.arange(offsets.size - 1)
    first = scorer.score(head, rows)
    assert scorer.reads
    scorer.reads.clear()
    # The same head again, as fresh arrays (a tool call rebuilds them): cached by digest.
    again = scorer.score({k: v.copy() for k, v in head.items()}, rows[::-1])
    assert scorer.reads == []
    np.testing.assert_array_equal(again, first[::-1])
    scorer.rule = "max"  # the rule pools the same cached token scores
    scorer.score(head, rows)
    assert scorer.reads == []
    # A changed head is a new entry: its rows are read again.
    edited = {**head, "weight": head["weight"] + 0.01}
    scorer.score(edited, rows[:5])
    assert scorer.reads and {r for a, b in scorer.reads for r in range(a, b)} == set(range(5))


def test_only_the_rows_asked_for_are_read_and_the_rest_later() -> None:
    acts, offsets = _practice(6)
    head = _head(acts, 6, standardise=False, degenerate=False, query=False)
    scorer = _CountingScorer(_function("mean"), acts, offsets)
    scorer.score(head, [40, 3, 4, 5])
    assert {r for a, b in scorer.reads for r in range(a, b)} == {3, 4, 5, 40}
    scorer.reads.clear()
    scorer.score(head, [3, 4, 9])
    assert {r for a, b in scorer.reads for r in range(a, b)} == {9}


def test_the_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    acts, offsets = _practice(7)
    scorer = Scorer(_function("mean"), acts, offsets)
    base = _head(acts, 7, standardise=False, degenerate=False, query=False)
    for i in range(scoring_mod.CACHE_ENTRIES + 3):
        scorer.score({**base, "bias": np.array([float(i)])}, [0, 1])
    assert len(scorer._cache) == scoring_mod.CACHE_ENTRIES
    entry_bytes = next(iter(scorer._cache.values())).nbytes
    monkeypatch.setattr(scoring_mod, "CACHE_MAX_BYTES", entry_bytes * 2)
    scorer.score({**base, "bias": np.array([99.0])}, [0])
    assert len(scorer._cache) == 2


def test_every_score_flows_through_read_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """The recording tests mean something only if ``read_rows`` is the one read: replace it with
    zeros and every row's score becomes the head's value at z = 0, ``b - mean . w'``."""
    acts, offsets = _practice(8)
    head = _head(acts, 8, standardise=True, degenerate=True, query=True)
    rows = np.arange(offsets.size - 1)
    for rule in RULES:
        real = Scorer(_function(rule), acts, offsets).score(head, rows)
        blind = Scorer(_function(rule), acts, offsets)
        monkeypatch.setattr(
            blind, "read_rows", lambda a, b: np.zeros((int(offsets[b] - offsets[a]), 48), "f2")
        )
        folded = scoring_mod.fold(head)
        at_zero = np.float32(folded.offset[0])
        got = blind.score(head, rows)
        np.testing.assert_allclose(got, np.full(rows.size, at_zero), rtol=1e-6, atol=1e-6)
        assert not np.allclose(real, got)
        monkeypatch.undo()


def test_refusals_are_kept() -> None:
    acts, offsets = _practice(9)
    head = _head(acts, 9, standardise=False, degenerate=False, query=False)
    with pytest.raises(ScoringError, match="attention_query"):
        Scorer(_function("attention"), acts, offsets).score(head, [0])
    with pytest.raises(ScoringError, match="outside"):
        Scorer(_function("mean"), acts, offsets).score(head, [offsets.size - 1])
    with pytest.raises(ScoringError, match="d_model"):
        Scorer(_function("mean"), acts, offsets).score({**head, "weight": np.ones(3)}, [0])
    with pytest.raises(ScoringError, match="tau"):
        Scorer(_function("softmax", tau=-1.0), acts, offsets).score(head, [0])
    with pytest.raises(ScoringError, match="window"):
        Scorer(_function("rolling_mean_max", window=0), acts, offsets).score(head, [0])
    empty = offsets.copy()
    empty[2] = empty[1]  # row 1 has no tokens
    with pytest.raises(ScoringError, match="no tokens"):
        Scorer(_function("mean"), acts, empty).score(head, [1])
    assert Scorer(_function("mean"), acts, offsets).score(head, []).shape == (0,)


def test_available_cpus_reads_the_cgroup_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quota = tmp_path / "cpu.max"
    monkeypatch.setattr(scoring_mod, "CPU_MAX", quota)
    monkeypatch.setattr(scoring_mod.os, "sched_getaffinity", lambda _pid: set(range(16)))
    quota.write_text("200000 100000\n")
    assert scoring_mod.available_cpus() == 2
    quota.write_text("150000 100000\n")
    assert scoring_mod.available_cpus() == 2
    quota.write_text("max 100000\n")
    assert scoring_mod.available_cpus() == scoring_mod.MAX_WORKERS
    quota.unlink()
    assert scoring_mod.available_cpus() == scoring_mod.MAX_WORKERS


@pytest.mark.parametrize("rule", RULES)
def test_golden_holds_with_massive_channels(rule: str) -> None:
    """A residual stream's few huge-mean channels (hundreds) and a wide head: folding the mean
    into the head cancels large terms, which a float32 product loses (up to 1e-4 on scores near
    10 on the 2026-11-21 benchmark). The float64 product keeps the old scorer's answers."""
    acts, offsets = _practice(21, rows=60, d_model=4096, offset=20.0)
    acts = acts.astype(np.float32)
    acts[:, :4] += np.array([300.0, -500.0, 800.0, -150.0], dtype=np.float32)
    acts = acts.astype(np.float16)
    head = _head(acts, 21, standardise=True, degenerate=False, query=True)
    head["weight"] *= 0.5
    head["attention_query"] *= 0.1
    function = _function(rule, window=5)
    rows = np.arange(offsets.size - 1)
    old = OldScorer(function, acts, offsets).score(head, rows)
    new = Scorer(function, acts, offsets).score(head, rows)
    np.testing.assert_allclose(new, old, rtol=SCORE_RTOL, atol=SCORE_ATOL)


@pytest.mark.parametrize("name", ["weight", "bias", "mean", "std", "attention_query"])
def test_the_cache_key_is_every_tensor_the_scorer_reads(name: str) -> None:
    """A head that differs in ANY of the tensors the scorer reads is a new cache entry: a key
    that missed one would serve the old head's scores for the new one (mutation M3, 2026-11-21:
    a key without ``mean`` passed every other test)."""
    acts, offsets = _practice(31)
    head = _head(acts, 31, standardise=True, degenerate=False, query=True)
    rows = np.arange(offsets.size - 1)
    scorer = Scorer(_function("attention"), acts, offsets)
    before = scorer.score(head, rows)
    changed = {**head, name: np.asarray(head[name], dtype=np.float64) * 1.5 + 0.25}
    fresh = Scorer(_function("attention"), acts, offsets).score(changed, rows)
    assert not np.allclose(fresh, before)
    np.testing.assert_array_equal(scorer.score(changed, rows), fresh)
