"""The verdict (feature 016; FTASKS 14.4; FR-016.28's AC; P53, R1B-M16, R1B-M17, R1B-M18).

Synthetic held-out sets with known truth: each head's scores are ``signal * label + noise``; its
bar is cut at a 10% false-alarm rate on its own primary negatives unless a case moves it. The
full-sample metrics a test hands in are computed here, the way miWorkbench reports them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from wsp.verdict import BASELINES, Head, HeldoutSet, decide

ROWS, FA_ROWS = 400, 1200


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (pos.size * neg.size))


def _head(label: str, scores: np.ndarray, labels: np.ndarray, threshold: float) -> Head:
    fired = scores >= threshold
    metrics: dict[str, Any] = {
        "auroc": _auroc(scores, labels) if 0 < labels.sum() < labels.size else None,
        "recall_at_bar": float(fired[labels == 1].mean()) if labels.any() else None,
        "false_alarms_at_bar": int(fired[labels == 0].sum()),
    }
    return Head(label, scores.astype(np.float32), threshold, metrics)


def world(
    signals: dict[str, float] | None = None,
    bars: dict[str, float] | None = None,
    *,
    rows: int = ROWS,
    positives: int | None = None,
    drop: tuple[str, ...] = (),
    drop_sets: tuple[str, ...] = (),
    seed: int = 0,
) -> list[HeldoutSet]:
    signal = {
        "parent": 1.0,
        "candidate": 2.0,
        "control": 1.0,
        **{f"baseline:{k}": 1.0 for k in BASELINES},
        **(signals or {}),
    }
    rng = np.random.default_rng(seed)
    n_pos = rows // 2 if positives is None else positives
    labels = np.zeros(rows, dtype=np.int64)
    labels[:n_pos] = 1
    noise = {h: rng.standard_normal(rows) for h in signal}
    primary_scores = {h: signal[h] * labels + noise[h] for h in signal if h not in drop}
    cut = {
        h: float(np.quantile(s[labels == 0], 0.9)) for h, s in primary_scores.items()
    }  # a 10% false-alarm bar per head
    cut.update(bars or {})
    sets = [
        HeldoutSet(
            "eval",
            "eval",
            labels,
            {h: _head(h, s, labels, cut[h]) for h, s in primary_scores.items()},
        )
    ]
    for name in ("everyday_text", "nonsense_text"):
        if name in drop_sets:
            continue
        fa = np.zeros(FA_ROWS, dtype=np.int64)
        heads = {h: _head(h, rng.standard_normal(FA_ROWS), fa, cut[h]) for h in primary_scores}
        sets.append(HeldoutSet(name, name, fa, heads))
    return sets


def verdict(sets: list[HeldoutSet], **kw: Any) -> Any:
    return decide(
        sets,
        ["eval"],
        changed_entries=kw.pop("changed", 5),
        seed=kw.pop("seed", 7),
        b=kw.pop("b", 500),
        **kw,
    )


def test_each_of_the_five_values_from_known_truth() -> None:
    assert verdict(world()).value == "improved"
    assert verdict(world({"candidate": 0.2})).value == "regressed"
    assert verdict(world({"candidate": 1.0})).value == "not_significant"
    assert verdict(world(), changed=0).value == "no_change"
    assert verdict(world(drop_sets=("nonsense_text",))).value == "unmeasured"


def test_beating_the_parent_but_not_lda_is_not_significant() -> None:
    out = verdict(world({"baseline:lda": 2.0}))
    assert out.value == "not_significant"
    assert any("auroc_not_above_baseline:lda" in r for r in out.reasons)


def test_beating_everything_on_auroc_but_losing_recall_to_the_recut_is_not_significant() -> None:
    # Candidate and parent both sit at a high bar (low recall); the recut keeps the 10% bar.
    out = verdict(world(bars={"candidate": 3.4, "parent": 2.4}))
    assert out.value == "not_significant"
    assert any("recall_at_bar_worse_than_threshold_recut" in r for r in out.reasons)


def test_affine_with_higher_recall_at_a_higher_false_alarm_rate_is_still_improved() -> None:
    out = verdict(world(bars={"baseline:affine_recalibration": -1.0}))
    assert out.value == "improved", out.reasons


def test_affine_better_on_both_blocks_improved() -> None:
    out = verdict(
        world({"baseline:affine_recalibration": 4.0}, bars={"baseline:affine_recalibration": 2.4})
    )
    assert out.value == "not_significant"
    assert "affine_recalibration_dominates" in out.reasons


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"drop_sets": ("nonsense_text",)}, "heldout_set_missing:nonsense_text"),
        ({"positives": 19}, "heldout_set_too_small:eval"),
        ({"drop": ("baseline:lda",)}, "baseline_missing:lda"),
    ],
    ids=["no-nonsense-set", "19-positives", "no-lda"],
)
def test_completeness_failures_are_unmeasured_and_named(
    kwargs: dict[str, Any], reason: str
) -> None:
    out = verdict(world(**kwargs))
    assert out.value == "unmeasured" and reason in out.reasons
    assert out.completeness == {"ok": False, "missing": out.reasons}


def test_two_controls_losing_to_either_blocks_improved() -> None:
    sets = world({"control:1": 1.0, "control:2": 2.0})
    out = verdict(sets)
    assert out.value == "not_significant"
    assert any("auroc_not_above_control:2" in r for r in out.reasons)
    assert not any("control:1" in r for r in out.reasons)


def test_the_guarded_comparisons_are_recorded() -> None:
    out = verdict(world())
    # recall vs parent on the one primary set, false-alarm rate vs parent on the two FA sets.
    assert out.guarded_comparisons == 3
    deltas = {(d["metric"], d["against"]) for s in out.per_set for d in s["deltas"]}
    assert ("false_alarm_rate", "parent") in deltas and ("auroc", "control") in deltas


def test_the_resample_is_stratified_and_reproducible() -> None:
    a, b = verdict(world(), seed=11), verdict(world(), seed=11)
    assert a.per_set == b.per_set
    assert verdict(world(), seed=12).per_set != a.per_set


def test_the_estimate_is_miworkbenchs_full_sample_difference() -> None:
    sets = world()
    out = verdict(sets)
    eval_deltas = {(d["metric"], d["against"]): d for d in out.per_set[0]["deltas"]}
    heads = sets[0].heads
    expected = heads["candidate"].metrics["auroc"] - heads["parent"].metrics["auroc"]
    assert eval_deltas[("auroc", "parent")]["estimate"] == pytest.approx(expected)
    d = eval_deltas[("auroc", "parent")]
    assert d["lower"] <= d["estimate"] <= d["upper"]


def test_a_guarded_metric_alone_can_regress() -> None:
    """A better AUROC does not hide a false-alarm rate significantly worse than the parent's."""
    out = verdict(world(bars={"candidate": -1.5}))
    assert out.value == "regressed"
    assert any("false_alarm_rate_worse_than_parent" in r for r in out.reasons)


def test_every_primary_set_must_improve() -> None:
    sets = world()
    flat = world({"candidate": 1.0}, seed=5)[0]
    sets.append(HeldoutSet("eval2", "eval", flat.labels, flat.heads))
    out = decide(sets, ["eval", "eval2"], changed_entries=5, seed=7, b=500)
    assert out.value == "not_significant"
    assert all(r.startswith("eval2:") for r in out.reasons), out.reasons


def test_each_resample_keeps_the_class_counts() -> None:
    """PF-9: positives are drawn from the positives and negatives from the negatives."""
    from wsp.verdict import _counts

    labels = np.array([1] * 7 + [0] * 13)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    counts = _counts(np.random.default_rng(3), pos, neg, labels.size, 50)
    assert np.all(counts[:, pos].sum(axis=1) == 7)
    assert np.all(counts[:, neg].sum(axis=1) == 13)


# ------------------------------------------------------------------ review 016-P2 (18.4, r016p2)
#
# Each test below was written for a mutation of ``verdict.py`` that the whole WSP tree let pass
# (controls V1, V2, V3, V11, V12, V15, V17, V18 in 0xcc/reviews/016_P2_review.md).


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"positives": ROWS - 19}, "heldout_set_too_small:eval"),
        ({"drop": ("baseline:threshold_recut",)}, "baseline_missing:threshold_recut"),
        ({"drop": ("baseline:affine_recalibration",)}, "baseline_missing:affine_recalibration"),
    ],
    ids=["19-negatives", "no-threshold-recut", "no-affine"],
)
def test_the_other_completeness_failures_are_unmeasured_and_named(
    kwargs: dict[str, Any], reason: str
) -> None:
    """FR-016.28: at least 20 rows of EACH class (V1), and BOTH operating-point baselines (V3)."""
    out = verdict(world(**kwargs))
    assert out.value == "unmeasured" and reason in out.reasons


def test_the_false_alarm_floor_is_at_least_not_more_than() -> None:
    """``WSP_MIN_FALSE_ALARM_ROWS`` negatives exactly is enough; one fewer is not (V2)."""
    assert verdict(world(), min_false_alarm_rows=FA_ROWS).value == "improved"
    out = verdict(world(), min_false_alarm_rows=FA_ROWS + 1)
    assert out.value == "unmeasured" and "heldout_set_too_small:everyday_text" in out.reasons


def test_a_false_alarm_rate_worse_than_the_recut_blocks_improved() -> None:
    """The recut is compared on the false-alarm sets too (V15): candidate and parent share a
    bar so low that most negatives fire, so neither is worse than the other, but both are far
    worse than the recut's 10% bar."""
    out = verdict(world(bars={"candidate": -1.5, "parent": -1.5}))
    assert out.value == "not_significant", out.reasons
    assert "everyday_text:false_alarm_rate_worse_than_threshold_recut" in out.reasons


def test_the_false_alarm_estimate_is_a_rate() -> None:
    """The reported estimate is ``false_alarms_at_bar / negatives`` minus the other's (V12), and
    the interval holds it."""
    sets = world(bars={"candidate": 1.0})
    out = verdict(sets)
    fa = next(s for s in out.per_set if s["name"] == "everyday_text")
    d = next(x for x in fa["deltas"] if x["against"] == "parent")
    heads = sets[1].heads
    expected = (
        heads["candidate"].metrics["false_alarms_at_bar"]
        - heads["parent"].metrics["false_alarms_at_bar"]
    ) / FA_ROWS
    assert d["metric"] == "false_alarm_rate" and d["estimate"] == pytest.approx(expected)
    assert d["lower"] <= d["estimate"] <= d["upper"]


def test_a_head_without_a_bar_never_fires_in_the_resample() -> None:
    """A ``None`` threshold fires on no row, as the metric guard reads it (V11): the recall
    interval against the parent sits around the candidate's recall of 0."""
    sets = world()
    eval_set = sets[0]
    c = eval_set.heads["candidate"]
    eval_set.heads["candidate"] = Head(
        "candidate", c.scores, None, {**c.metrics, "recall_at_bar": 0.0}
    )
    out = verdict(sets)
    d = next(
        x
        for x in out.per_set[0]["deltas"]
        if x["metric"] == "recall_at_bar" and x["against"] == "parent"
    )
    parent = eval_set.heads["parent"].metrics["recall_at_bar"]
    assert d["estimate"] == pytest.approx(-parent)
    assert d["lower"] <= d["estimate"] <= d["upper"] < 0


def test_auroc_counts_a_tie_as_half() -> None:
    """Mann-Whitney with ties counted half, in the full-sample AUROC (V17: the metric guard and
    the practice qualification use it) and in each resample's weighted AUROC (V18)."""
    from wsp.verdict import _Ranked, auroc

    scores = np.array([0.1, 0.5, 0.5, 0.5, 0.9, 0.2, 0.5, 0.9], dtype=np.float32)
    labels = np.array([0, 0, 1, 1, 1, 0, 0, 1], dtype=np.int64)
    expected = _auroc(scores.astype(np.float64), labels)
    assert expected == pytest.approx(0.875)  # 14 of 16 pairs, two of them ties, by hand
    assert auroc(scores, labels) == pytest.approx(expected, abs=1e-15)
    counts = np.ones((2, labels.size))
    counts[1, 0] = 3.0  # the second resample draws row 0 three times
    got = _Ranked(scores, labels).auroc(counts)
    repeated = np.repeat(np.arange(labels.size), counts[1].astype(int))
    assert got[0] == pytest.approx(expected, abs=1e-15)
    assert got[1] == pytest.approx(_auroc(scores[repeated].astype(np.float64), labels[repeated]))


def test_a_set_blocked_inside_a_regressed_verdict_is_not_improved() -> None:
    """R16P2-2: the false-alarm sets regress, and on the primary set the candidate does not beat
    the control. The verdict is ``regressed``; the primary set's own value must not read
    ``improved`` (it did, because a regressed outcome kept only the regressions as reasons)."""
    from wsp.evaluate import _set_value

    out = verdict(world({"control": 2.0}, bars={"candidate": -1.5}))
    assert out.value == "regressed", out.reasons
    assert any(r.startswith("eval:auroc_not_above_control") for r in out.blocked)
    values = {s["name"]: _set_value(s["name"], out, bool(s["deltas"])) for s in out.per_set}
    assert values == {
        "eval": "not_significant",
        "everyday_text": "regressed",
        "nonsense_text": "regressed",
    }
    # And a set that was compared and blocked by nothing still reads improved.
    clean = verdict(world())
    assert {_set_value(s["name"], clean, True) for s in clean.per_set} == {"improved"}
