"""The verdict: completeness, the paired stratified bootstrap and the decision (feature 016;
FR-016.28; FTDD §6.3; P6, P20, P53, PF-9, PF-23, R1B-M16, R1B-M17, R1B-M18). Pure: numpy only.

The point estimates are miWorkbench's full-sample metrics (P29); this module computes the
intervals only. Each resample draws positives from the positives and negatives from the negatives
(PF-9), the SAME indices for every head; bars are fixed (miWorkbench's), never re-cut per resample.
AUROC is Mann-Whitney with ties counted half, computed per resample from each head's sorted order
and the resample's per-row counts, so B = 2,000 over 10^4 rows stays inside the timing budget.

Decision order: ``no_change``; ``unmeasured`` (completeness, named); ``regressed``; ``improved``;
``not_significant``. Multiplicity is recorded (``guarded_comparisons``), not corrected (PF-23).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

B_DEFAULT = 2000
CI_LOW, CI_HIGH = 2.5, 97.5
REFITS = ("logistic_regression", "lda", "diff_of_means")
OPERATING_POINT = ("threshold_recut", "affine_recalibration")
BASELINES = REFITS + OPERATING_POINT
FALSE_ALARM_ROLES = ("everyday_text", "nonsense_text")
BATCH = 250


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney with average ranks (a tied pair counts half); ``None`` for one class."""
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = int(labels.size) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    _, inverse, counts = np.unique(
        np.asarray(scores, dtype=np.float64), return_inverse=True, return_counts=True
    )
    ends = np.cumsum(counts).astype(np.float64)
    ranks = ((ends - counts + 1.0 + ends) / 2.0)[inverse]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass(frozen=True)
class Head:
    label: str
    scores: np.ndarray  # float32, one per row
    threshold: float | None
    #: miWorkbench's full-sample metrics: auroc, recall_at_bar, false_alarms_at_bar (a count).
    metrics: dict[str, Any]


@dataclass(frozen=True)
class HeldoutSet:
    name: str
    role: str
    labels: np.ndarray  # 0/1 per row
    heads: dict[str, Head]


@dataclass
class Outcome:
    value: str
    reasons: list[str]
    completeness: dict[str, Any]
    guarded_comparisons: int
    per_set: list[dict[str, Any]] = field(default_factory=list)
    #: Every per-set reason that blocks ``improved``, kept when the value is ``regressed`` too
    #: (whose ``reasons`` name only the regressions), so a set's own value never reads
    #: ``improved`` where it was blocked (review 016-P2, R16P2-2).
    blocked: list[str] = field(default_factory=list)


def _counts(
    rng: np.random.Generator, pos: np.ndarray, neg: np.ndarray, n: int, b: int
) -> np.ndarray:
    """``b`` stratified resamples as per-row counts, shape ``(b, n)``."""
    draws = np.concatenate(
        [
            pos[rng.integers(0, pos.size, size=(b, pos.size))],
            neg[rng.integers(0, neg.size, size=(b, neg.size))],
        ],
        axis=1,
    )
    flat = draws + (np.arange(b)[:, None] * n)
    return np.bincount(flat.reshape(-1), minlength=b * n).reshape(b, n).astype(np.float64)


class _Ranked:
    """One head's sorted order and tie groups, for weighted AUROC on any count matrix."""

    def __init__(self, scores: np.ndarray, labels: np.ndarray) -> None:
        s = np.asarray(scores, dtype=np.float64)
        self.order = np.argsort(s, kind="stable")
        ss = s[self.order]
        self.starts = np.flatnonzero(np.concatenate([[True], ss[1:] != ss[:-1]]))
        self.group = np.cumsum(np.concatenate([[True], ss[1:] != ss[:-1]])) - 1
        self.pos = np.asarray(labels, dtype=bool)[self.order]

    def auroc(self, counts: np.ndarray) -> np.ndarray:
        w = counts[:, self.order]
        wp, wn = w * self.pos, w * ~self.pos
        neg_group = np.add.reduceat(wn, self.starts, axis=1)
        below = np.cumsum(neg_group, axis=1) - neg_group  # negatives strictly below each group
        score = (wp * (below[:, self.group] + 0.5 * neg_group[:, self.group])).sum(axis=1)
        total = wp.sum(axis=1) * wn.sum(axis=1)
        out: np.ndarray = score / total
        return out


def _rate(counts: np.ndarray, fired: np.ndarray, mask: np.ndarray) -> np.ndarray:
    w = counts * mask
    out: np.ndarray = (w * fired).sum(axis=1) / w.sum(axis=1)
    return out


def _metric(
    cache: dict[tuple[str, str], np.ndarray],
    m: str,
    h: str,
    counts: np.ndarray,
    ranked: dict[str, _Ranked],
    fired: dict[str, np.ndarray],
    labels: np.ndarray,
) -> np.ndarray:
    """One head's metric over a batch of resamples, computed once per batch."""
    key = (m, h)
    if key not in cache:
        if m == "auroc":
            cache[key] = ranked[h].auroc(counts)
        elif m == "recall_at_bar":
            cache[key] = _rate(counts, fired[h], labels == 1)
        else:
            cache[key] = _rate(counts, fired[h], labels == 0)
    return cache[key]


def completeness(
    sets: Sequence[HeldoutSet],
    primary: Sequence[str],
    min_per_class: int,
    min_false_alarm_rows: int,
) -> list[str]:
    missing: list[str] = []
    by_name = {s.name: s for s in sets}
    for name in primary:
        s = by_name.get(name)
        if s is None:
            missing.append(f"heldout_set_missing:{name}")
            continue
        positives = int(np.sum(s.labels == 1))
        if min(positives, int(s.labels.size) - positives) < min_per_class:
            missing.append(f"heldout_set_too_small:{name}")
    for role in FALSE_ALARM_ROLES:
        found = [s for s in sets if s.role == role]
        if not found:
            missing.append(f"heldout_set_missing:{role}")
        elif not any(int(np.sum(s.labels == 0)) >= min_false_alarm_rows for s in found):
            missing.extend(f"heldout_set_too_small:{s.name}" for s in found)
    for kind in BASELINES:
        if any(f"baseline:{kind}" not in s.heads for s in sets if s.name in primary):
            missing.append(f"baseline_missing:{kind}")
    return missing


def _interval(deltas: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(deltas, [CI_LOW, CI_HIGH])
    return float(low), float(high)


def _full(head: Head, metric: str, negatives: int) -> float:
    if metric == "false_alarm_rate":
        return float(head.metrics["false_alarms_at_bar"]) / negatives
    return float(head.metrics[metric])


def decide(
    sets: Sequence[HeldoutSet],
    primary: Sequence[str],
    *,
    changed_entries: int,
    seed: int,
    b: int = B_DEFAULT,
    min_per_class: int = 20,
    min_false_alarm_rows: int = 1000,
) -> Outcome:
    """The verdict's value, reasons, completeness, multiplicity count and per-set deltas."""
    if changed_entries == 0:
        return Outcome("no_change", ["the patch changes no entry"], {"ok": True, "missing": []}, 0)
    missing = completeness(sets, primary, min_per_class, min_false_alarm_rows)
    if missing:
        return Outcome("unmeasured", missing, {"ok": False, "missing": missing}, 0)
    regressed: list[str] = []
    blocked: list[str] = []
    guarded = 0
    affine_recall_better = affine_far_better = False
    per_set: list[dict[str, Any]] = []
    for index, s in enumerate(sets):
        is_primary = s.name in primary
        if not is_primary and s.role not in FALSE_ALARM_ROLES:
            continue
        labels = np.asarray(s.labels).astype(np.int64)
        pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
        n, negatives = int(labels.size), int(neg.size)
        rng = np.random.default_rng(seed + index)
        heads = s.heads
        controls = sorted(h for h in heads if h == "control" or h.startswith("control:"))
        wanted: list[tuple[str, str]] = []  # (metric, against): delta = candidate - against
        if is_primary:
            refits = [f"baseline:{k}" for k in REFITS]
            best = max(refits, key=lambda h: float(heads[h].metrics["auroc"]))
            wanted += [("auroc", "parent"), *[("auroc", c) for c in controls], ("auroc", best)]
            wanted += [("recall_at_bar", "parent"), ("recall_at_bar", "baseline:threshold_recut")]
            wanted += [("recall_at_bar", "baseline:affine_recalibration")]
        else:
            wanted += [
                ("false_alarm_rate", "parent"),
                ("false_alarm_rate", "baseline:threshold_recut"),
            ]
            if "baseline:affine_recalibration" in heads:
                wanted += [("false_alarm_rate", "baseline:affine_recalibration")]
        labels_needed = {"candidate", *(a for _, a in wanted)}
        ranked = {h: _Ranked(heads[h].scores, labels) for h in labels_needed} if is_primary else {}
        fired = {
            h: (
                np.asarray(heads[h].scores, dtype=np.float64)
                >= (heads[h].threshold if heads[h].threshold is not None else np.inf)
            ).astype(np.float64)
            for h in labels_needed
        }
        samples: dict[tuple[str, str], list[np.ndarray]] = {k: [] for k in wanted}
        for start in range(0, b, BATCH):
            counts = _counts(rng, pos, neg, n, min(BATCH, b - start))
            cache: dict[tuple[str, str], np.ndarray] = {}

            for m, against in wanted:
                one = _metric(cache, m, "candidate", counts, ranked, fired, labels)
                other = _metric(cache, m, against, counts, ranked, fired, labels)
                samples[(m, against)].append(one - other)
        deltas = []
        for (m, against), parts in samples.items():
            low, high = _interval(np.concatenate(parts))
            worse_if = "higher" if m == "false_alarm_rate" else "lower"
            estimate = _full(heads["candidate"], m, negatives) - _full(heads[against], m, negatives)
            deltas.append(
                {
                    "metric": m,
                    "against": against,
                    "estimate": estimate,
                    "lower": low,
                    "upper": high,
                    "worse_if": worse_if,
                }
            )
            worse = high < 0 if worse_if == "lower" else low > 0
            better = low > 0 if worse_if == "lower" else high < 0
            if against == "parent" and m != "auroc":
                guarded += 1
            if against == "parent" and worse:
                regressed.append(f"{s.name}:{m}_worse_than_parent")
            if m == "auroc" and not better:
                blocked.append(f"{s.name}:auroc_not_above_{against}")
            if against == "baseline:threshold_recut" and worse:
                blocked.append(f"{s.name}:{m}_worse_than_threshold_recut")
            if against == "baseline:affine_recalibration":
                # Affine's own bar is another operating point: it only matters when it is
                # significantly better on recall AND on false alarms at once (R1B-M17).
                if m == "recall_at_bar" and worse:
                    affine_recall_better = True
                if m == "false_alarm_rate" and worse:
                    affine_far_better = True
        per_set.append({"name": s.name, "role": s.role, "primary": is_primary, "deltas": deltas})
    complete = {"ok": True, "missing": []}
    if regressed:
        return Outcome("regressed", regressed, complete, guarded, per_set, blocked)
    if affine_recall_better and affine_far_better:
        blocked.append("affine_recalibration_dominates")
    if not blocked:
        return Outcome("improved", [], complete, guarded, per_set)
    return Outcome("not_significant", blocked, complete, guarded, per_set, blocked)
