"""Observation tools: the editor's forward passes over the practice set (feature 016; FR-016.18;
FTDD §6.5; WR-8, WR-14, D11).

Each tool is versioned, declares a ``cost`` drawn from the run's ``forward_passes`` budget (the
adapter wraps it as the library's evaluation callback), reads ONLY the practice set, and answers
numbers and **row ids** (practice row indices): never text, token strings, or activation rows.

The guarantee is enforced twice:

- every tool reads token activations only through the view's scorer, whose one read of the map is
  :meth:`wsp_worker.scoring.Scorer.read_rows` (blocks of whole rows, fix 2026-11-21: the scorer is
  vectorised and caches each head's token scores), so a test that overrides it records every row
  a tool touches, and a test that replaces it proves every score flows through it;
- every answer passes :func:`numbers_and_names_only` before it leaves the tool: a string must be a
  name (``^[A-Za-z0-9_.\\-]{1,64}$``), and a list may hold integers (row ids, positions) but never
  floats, so a raw activation vector cannot be returned even as numbers (control C18).
  Since 2026-11-28 it also bounds an answer's size (an object's number values and keys, a list's
  length, and the answer's scalars in all), so a vector cannot leave as a dict of named numbers,
  a nested dict, flags or numeric names either (review of train F-1, 2026-11-28).
  The one exception to "a string is a name" is a note from :data:`wsp_worker.judging.FIXED_TEXTS`:
  module literals, so no data can ride in one.

**Version 3 (fix 2026-11-22, D128 round 7): the tools measure the way the verdict judges.** The
held-out verdict cuts EVERY head's bar afresh to the probe's target false-alarm rate and ranks by
AUROC, so it is blind to a bias shift. Version 2's ``at_parent_bar`` held the parent's shipped bar
fixed, where a bias shift does move false alarms, and the round-7 editor optimised that figure
into three candidates equal to the parent bit for bit. Version 3 (``practice_metrics``,
``compare_to_parent``; ``stratified_false_alarms`` version 2) reports AUROC and ``at_own_bar``: the
head at its own bar, re-cut to ``operating_point.target_fpr`` on the scorer's practice basis
(the practice VALIDATION negatives that were not truncated, miWorkbench's
``practice_bar_indices``) by miWorkbench's ``calibrate`` rule (:func:`wsp_worker.scoring.cut_bar`,
firing at ``score >= bar``). ``at_parent_bar`` is gone from version 3: a fixed-bar figure is one
the verdict never uses. Each version-3 answer carries the invariance note and, when the head is a
positive affine map of the parent (:mod:`wsp_worker.equivalence`), says the verdict cannot
differ from the parent's. Versions 1 and 2 stay servable, unchanged, for replays.

**Version 4 (plan A1, 2026-11-22, D128 round 9): a difference comes with its uncertainty.**
``compare_to_parent`` 4 adds ``auroc_against_parent``: the verdict's paired-bootstrap interval on
the practice rows (:mod:`wsp_worker.practice_interval`), and says so in words when it includes
zero. ``practice_metrics`` 4 measures on validation unless asked: 3's default was train, the rows
the parent was fitted on. Version 3 stays servable, unchanged, for replays.

Library-free (numpy only): program mode's ``test_program`` reuses the same view in a rerun child
that must never load a model client (R1B-A9).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from wsp_worker import equivalence, judging, practice_interval
from wsp_worker.scoring import RULES, Scorer, auroc, bar_at, cut_bar, rates, rates_at_or_above

NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
SPLITS = ("train", "validation", "all")
STRATA = ("length_band", "truncation")
LENGTH_BANDS = ((1, 16), (17, 64), (65, 256), (257, 1 << 31))
RANDOM_DIRECTIONS = 8
MAX_ROW_IDS = 50


class ObservationToolError(ValueError):
    def __init__(self, message: str, code: str = "worker_observation_tool_invalid") -> None:
        super().__init__(message)
        self.code = code


#: The answer guard's size limits (fix 2026-11-28, E7 §5.3 gap 1: a dict of floats passed it, so a
#: tool could hand the editor a whole activation vector as ``{"d0": 0.1, ..., "d4095": 0.3}``). A
#: number is an ``int`` or a ``float`` (an int can carry a quantised value as well as a float).
#: Measured on every tool's answers in the worker and WSP test trees (2026-11-28): at most 8
#: numbers in one object, 24 items in one list and 120 numbers in one answer
#: (``firing_positions``, which at its ``top_k`` maximum of 50 answers 250). ``supplement_metrics``
#: answers about 17 numbers per supplement, so 2,048 admits about 120 supplements.
#:
#: Review of train F-1 (2026-11-28): one level of nesting (``{"d0": {"v": 0.1}, ...}``, each inner
#: object holding one number), a bool per dimension (a sign vector) and numbers written as names
#: (``"0.1234"`` matches :data:`NAME`) each passed the first form. So an object may hold at most
#: :data:`MAX_OBJECT_KEYS` keys (an object keyed by names is a list by another spelling), a bool
#: counts as a number (one bit of a quantised value), and every scalar (number, bool or string)
#: counts toward the answer's budget. Measured again over the same trees (2026-11-28): at most 13
#: keys in one object, 2 bools and 10 strings in one answer, 121 scalars in one answer.
#: ``supplement_metrics`` keys its ``sources`` object by supplement, so a bundle may carry at most
#: 64 supplements before that tool is refused (today's bundles carry 10).
MAX_NUMBERS_PER_OBJECT = 32
MAX_LIST_ITEMS = 64
MAX_OBJECT_KEYS = MAX_LIST_ITEMS
MAX_NUMBERS_PER_ANSWER = 2048


def numbers_and_names_only(value: Any, path: str = "$") -> Any:
    """Refuse any string that is not a name, any list holding a float (a vector), and any answer
    too large to be a summary: an object with more than :data:`MAX_NUMBERS_PER_OBJECT` number or
    bool values (a vector keyed by names) or more than :data:`MAX_OBJECT_KEYS` keys, a list longer
    than :data:`MAX_LIST_ITEMS`, or more than :data:`MAX_NUMBERS_PER_ANSWER` scalars (numbers,
    bools and strings) in all (a vector split across objects, as flags or as numeric names)."""
    count = [0]
    _check_answer(value, path, count)
    return value


def _is_number(value: Any) -> bool:
    """An int, a float or a bool (a bool is one bit of a quantised value; review 2026-11-28)."""
    return isinstance(value, int | float)


def _count(path: str, count: list[int]) -> None:
    count[0] += 1
    if count[0] > MAX_NUMBERS_PER_ANSWER:
        raise ObservationToolError(
            f"{path}: more than {MAX_NUMBERS_PER_ANSWER} numbers in one answer"
        )


def _check_answer(value: Any, path: str, count: list[int]) -> None:
    if value is None:
        return
    if _is_number(value):
        _count(path, count)
        return
    if isinstance(value, str):
        if not NAME.match(value) and value not in judging.FIXED_TEXTS:
            raise ObservationToolError(f"{path}: a string that is not a name")
        _count(path, count)  # a name can spell a number ("0.1234")
        return
    if isinstance(value, dict):
        numbers = sum(1 for item in value.values() if _is_number(item))
        if numbers > MAX_NUMBERS_PER_OBJECT:
            raise ObservationToolError(
                f"{path}: an object with {numbers} number values (at most "
                f"{MAX_NUMBERS_PER_OBJECT}; a vector keyed by names)"
            )
        if len(value) > MAX_OBJECT_KEYS:
            raise ObservationToolError(
                f"{path}: an object with {len(value)} keys (at most {MAX_OBJECT_KEYS}; a list "
                "keyed by names)"
            )
        for key, item in value.items():
            if not isinstance(key, str) or not NAME.match(key):
                raise ObservationToolError(f"{path}: a key that is not a name")
            _check_answer(item, f"{path}.{key}", count)
        return
    if isinstance(value, list):
        if len(value) > MAX_LIST_ITEMS:
            raise ObservationToolError(
                f"{path}: a list of {len(value)} items (at most {MAX_LIST_ITEMS})"
            )
        for i, item in enumerate(value):
            if isinstance(item, float):
                raise ObservationToolError(f"{path}[{i}]: a list of floats (a vector)")
            _check_answer(item, f"{path}[{i}]", count)
        return
    raise ObservationToolError(f"{path}: a {type(value).__name__}")


@dataclass(frozen=True)
class SupplementView:
    """The bundle's validation-only supplements (revision 2026-11-25, FU-7a C2), as
    :func:`supplement_metrics` sees them: their rows concatenated behind their own scorer, NEVER
    in a :class:`PracticeView` split, so no train, validation or bar-basis row is one of them."""

    scorer: Scorer
    labels: np.ndarray
    truncated: np.ndarray
    entries: tuple[dict[str, Any], ...]


@dataclass
class PracticeView:
    """The practice set as the tools see it. Token activations only through ``scorer``."""

    scorer: Scorer
    labels: np.ndarray
    validation: np.ndarray
    truncated: np.ndarray
    target_fpr: float
    #: The probe's shipped bar, the bundle's ``operating_point.threshold`` (the parent's bar:
    #: cut on its calibration set or practice negatives), or ``None`` when the bundle has none.
    shipped_bar: float | None = None
    #: The bundle's validation-only supplements, or ``None`` (a bundle without them).
    supplement: SupplementView | None = None
    #: The parent's practice cut, per scoring rule, computed once each.
    _parent_cut: dict[str, float | None] = field(default_factory=dict, repr=False)

    @property
    def rows(self) -> int:
        return self.scorer.rows

    def split(self, name: str) -> np.ndarray:
        if name not in SPLITS:
            raise ObservationToolError(f"split must be one of {list(SPLITS)}")
        everything = np.arange(self.rows, dtype=np.int64)
        if name == "all":
            return everything
        mask = np.asarray(self.validation, dtype=bool)
        chosen: np.ndarray = everything[mask] if name == "validation" else everything[~mask]
        return chosen

    def row_length(self, row: int) -> int:
        return int(self.scorer.offsets[row + 1]) - int(self.scorer.offsets[row])

    def bar(self, tensors: Mapping[str, np.ndarray]) -> float | None:
        """The practice-cut bar: on the TRAIN split's negatives at the bundle's target FPR."""
        train = self.split("train")
        return bar_at(self.scorer.score(tensors, train), self.labels[train], self.target_fpr)

    def judged_rows(self) -> np.ndarray:
        """The scorer's practice basis (miWorkbench ``practice_bar_indices``): the VALIDATION
        negatives that were not truncated. The verdict's ``threshold_recut`` baseline and a
        practice-basis bundle's bars are cut on exactly these rows."""
        mask = (
            np.asarray(self.validation, dtype=bool)
            & ~np.asarray(self.truncated, dtype=bool)
            & ~np.asarray(self.labels).astype(bool)
        )
        rows: np.ndarray = np.flatnonzero(mask).astype(np.int64)
        return rows

    def judged_bar(self, tensors: Mapping[str, np.ndarray]) -> float | None:
        """``tensors``' OWN bar, re-cut as the verdict re-cuts every head's: on
        :meth:`judged_rows` at the target false-alarm rate, by :func:`cut_bar`."""
        return cut_bar(self.scorer.score(tensors, self.judged_rows()), self.target_fpr)

    def parent_bar(
        self, parent: Mapping[str, np.ndarray], *, own_rule: bool = True
    ) -> FixedBar | None:
        """The PARENT's bar, held fixed whatever the edit (fix 2026-11-21, run
        ``wsp_855a74f820ff``): the bundle's shipped bar when it has one and the scorer runs the
        probe's own rule (a row fires at ``score >= bar``, the evaluator's rule); else the
        parent's own practice cut at the scorer's rule, computed once from the PARENT's tensors
        (a row fires at ``score > bar``, the practice cut's rule)."""
        if self.shipped_bar is not None and own_rule:
            return FixedBar(float(self.shipped_bar), "shipped", inclusive=True)
        rule = str(self.scorer.rule)
        if rule not in self._parent_cut:
            self._parent_cut[rule] = self.bar(parent)
        cut = self._parent_cut[rule]
        return None if cut is None else FixedBar(cut, "parent_practice_cut", inclusive=False)


@dataclass(frozen=True)
class FixedBar:
    bar: float
    #: ``shipped`` (the bundle's operating point) or ``parent_practice_cut``.
    source: str
    #: A row fires at ``score >= bar`` (``True``) or ``score > bar``.
    inclusive: bool

    def rates(self, scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels).astype(bool)
        fired = s >= self.bar if self.inclusive else s > self.bar
        return {
            "bar": self.bar,
            "source": self.source,
            "recall": None if not y.any() else float(fired[y].mean()),
            "false_alarm_rate": None if y.all() else float(fired[~y].mean()),
        }


def shipped_bar_of(manifest: Mapping[str, Any]) -> float | None:
    """The bundle's shipped bar (``operating_point.threshold``: the parent's bar), else ``None``
    (an SAE bundle has no operating point)."""
    point = manifest.get("operating_point")
    value = point.get("threshold") if isinstance(point, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


#: The target a bundle without ``operating_point.target_fpr`` (or with none set) is cut at: a
#: silent fallback, recorded as debt (an SAE bundle has no operating point; a linear-probe bundle's
#: contract requires the field).
DEFAULT_TARGET_FPR = 0.01


def target_fpr_of(manifest: Mapping[str, Any]) -> float:
    """The probe's target false-alarm rate, ``operating_point.target_fpr``, the rate the held-out
    verdict re-cuts every head's bar to; :data:`DEFAULT_TARGET_FPR` without one. THE one reading,
    for the engine's and the adapter's observation context and for ``ProgramTools`` (fix
    2026-11-22: three inline copies, none pinned by a test whose target differed from the
    fallback)."""
    point = manifest.get("operating_point")
    value = point.get("target_fpr") if isinstance(point, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int | float) or not value:
        return DEFAULT_TARGET_FPR
    return float(value)


@dataclass(frozen=True)
class Context:
    view: PracticeView
    current: Callable[[], dict[str, np.ndarray]]
    parent: dict[str, np.ndarray]
    seed: int


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    cost: int
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., dict[str, Any]]


def _metrics(
    view: PracticeView,
    tensors: Mapping[str, np.ndarray],
    rows: np.ndarray,
    parent: Mapping[str, np.ndarray] | None,
    *,
    own_rule: bool = True,
) -> dict[str, Any]:
    """The split's metrics at the re-cut bar (``bar``: cut on ``tensors`` themselves) and, given
    the ``parent`` (version 2), at the parent's bar held fixed (``at_parent_bar``,
    :meth:`PracticeView.parent_bar`). Version 1 passes no parent and answers exactly as before."""
    scores = view.scorer.score(tensors, rows)
    labels = view.labels[rows]
    bar = view.bar(tensors)
    out: dict[str, Any] = {
        "rows": int(rows.size),
        "auroc": auroc(scores, labels),
        "bar": bar,
        **rates(scores, labels, bar),
    }
    if parent is not None:
        fixed = view.parent_bar(parent, own_rule=own_rule)
        out["at_parent_bar"] = None if fixed is None else fixed.rates(scores, labels)
    return out


def practice_metrics(ctx: Context, split: str = "train", rule: str | None = None) -> dict[str, Any]:
    return _practice_metrics(ctx, split, rule, fixed=True)


def practice_metrics_v1(
    ctx: Context, split: str = "train", rule: str | None = None
) -> dict[str, Any]:
    """Version 1, kept so a run that named it (a replay) runs it exactly: the re-cut bar only."""
    return _practice_metrics(ctx, split, rule, fixed=False)


def _practice_metrics(ctx: Context, split: str, rule: str | None, *, fixed: bool) -> dict[str, Any]:
    view = ctx.view
    rows = view.split(split)
    if rule is not None and rule not in RULES:
        raise ObservationToolError(f"rule must be one of {list(RULES)}")
    saved = view.scorer.rule
    view.scorer.rule = rule or saved
    try:
        out = _metrics(
            view,
            ctx.current(),
            rows,
            ctx.parent if fixed else None,
            own_rule=view.scorer.rule == saved,
        )
    finally:
        view.scorer.rule = saved
    return {"split": split, "rule": rule or saved, **out}


def stratified_false_alarms(ctx: Context, by: str = "length_band") -> dict[str, Any]:
    if by not in STRATA:
        raise ObservationToolError(
            f"by must be one of {list(STRATA)} (template positions are not in a practice bundle)"
        )
    view = ctx.view
    tensors = ctx.current()
    rows = view.split("all")
    scores = view.scorer.score(tensors, rows)
    bar = view.bar(tensors)
    negatives = ~np.asarray(view.labels[rows]).astype(bool)
    strata: dict[str, dict[str, Any]] = {}
    for i, r in enumerate(rows):
        if not negatives[i]:
            continue
        if by == "truncation":
            key = "truncated" if bool(view.truncated[r]) else "complete"
        else:
            n = view.row_length(int(r))
            low, high = next(b for b in LENGTH_BANDS if b[0] <= n <= b[1])
            key = f"len_{low}_{high}" if high < (1 << 31) else f"len_{low}_up"
        cell = strata.setdefault(key, {"negatives": 0, "false_alarms": 0})
        cell["negatives"] += 1
        cell["false_alarms"] += int(bar is not None and float(scores[i]) > bar)
    for cell in strata.values():
        cell["rate"] = cell["false_alarms"] / cell["negatives"]
    return {"by": by, "bar": bar, "strata": strata}


def direction_cosines(ctx: Context, names: list[str] | None = None) -> dict[str, Any]:
    tensors = ctx.current()
    chosen = list(names) if names else [n for n in ("weight", "attention_query") if n in tensors]
    out: dict[str, Any] = {}
    for name in chosen:
        if not isinstance(name, str) or name not in tensors:
            raise ObservationToolError("names must name head tensors")
        v = np.asarray(tensors[name], dtype=np.float64).reshape(-1)
        p = np.asarray(ctx.parent[name], dtype=np.float64).reshape(-1)
        digest = hashlib.sha256(f"{ctx.seed}:{name}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        randoms = rng.standard_normal((RANDOM_DIRECTIONS, v.size))
        cos = [abs(_cos(v, r) or 0.0) for r in randoms]
        out[name] = {
            "cos_parent": _cos(v, p),
            "random_mean_abs": float(np.mean(cos)),
            "random_max_abs": float(np.max(cos)),
            "directions": RANDOM_DIRECTIONS,
        }
    return {"cosines": out}


def _cos(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(a @ b / (na * nb))


def firing_positions(ctx: Context, top_k: int = 10) -> dict[str, Any]:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_ROW_IDS:
        raise ObservationToolError(f"top_k is an integer in 1..{MAX_ROW_IDS}")
    view = ctx.view
    tensors = ctx.current()
    rows = view.split("all")
    scores = view.scorer.score(tensors, rows)
    order = np.argsort(-scores.astype(np.float64), kind="stable")[:top_k]
    top = []
    for i in order:
        r = int(rows[i])
        token_scores = view.scorer.token_scores(tensors, r)
        top.append(
            {
                "row": r,
                "score": float(scores[i]),
                "label": int(view.labels[r]),
                "peak_position": int(np.argmax(token_scores)),
                "length": view.row_length(r),
            }
        )
    return {"top": top}


def compare_to_parent(ctx: Context, split: str = "validation") -> dict[str, Any]:
    return _compare_to_parent(ctx, split, fixed=True)


def compare_to_parent_v1(ctx: Context, split: str = "validation") -> dict[str, Any]:
    """Version 1, kept so a run that named it (a replay) runs it exactly: re-cut bars only."""
    return _compare_to_parent(ctx, split, fixed=False)


def _compare_to_parent(ctx: Context, split: str, *, fixed: bool) -> dict[str, Any]:
    view = ctx.view
    rows = view.split(split)
    tensors = ctx.current()
    parent = ctx.parent if fixed else None
    now = _metrics(view, tensors, rows, parent)
    before = _metrics(view, ctx.parent, rows, parent)
    fired_now = view.scorer.score(tensors, rows) > (np.inf if now["bar"] is None else now["bar"])
    fired_before = view.scorer.score(ctx.parent, rows) > (
        np.inf if before["bar"] is None else before["bar"]
    )
    flipped = [int(r) for r, a, b in zip(rows, fired_now, fired_before, strict=True) if a != b]
    return {
        "split": split,
        "now": now,
        "parent": before,
        "flipped": len(flipped),
        "flipped_rows": flipped[:MAX_ROW_IDS],
    }


# ----------------------------------------------------------------- version 3: as the verdict judges


BASIS = "practice_validation_negatives"


def _judged(
    view: PracticeView, tensors: Mapping[str, np.ndarray], rows: np.ndarray
) -> dict[str, Any]:
    """AUROC and the head at its OWN bar (:meth:`PracticeView.judged_bar`) on ``rows``."""
    scores = view.scorer.score(tensors, rows)
    labels = view.labels[rows]
    bar = view.judged_bar(tensors)
    return {
        "rows": int(rows.size),
        "auroc": auroc(scores, labels),
        "at_own_bar": {
            "bar": bar,
            "basis": BASIS,
            "basis_negatives": int(view.judged_rows().size),
            "target_false_alarm_rate": float(view.target_fpr),
            "fires": "at_or_above",
            **rates_at_or_above(scores, labels, bar),
        },
    }


def _notes(ctx: Context, rule: str) -> dict[str, Any]:
    """The invariance note, and whether (and so that) the current head cannot change the verdict:
    at the probe's OWN rule, the rule the verdict scores with."""
    same = equivalence.affine_to_parent(rule, ctx.parent, ctx.current()) is not None
    out: dict[str, Any] = {
        "affine_equivalent_to_parent": same,
        "note": judging.invariance_note(rule),
    }
    if same:
        out["same_verdict_note"] = judging.SAME_VERDICT
    return out


def practice_metrics_v3(
    ctx: Context, split: str = "train", rule: str | None = None
) -> dict[str, Any]:
    view = ctx.view
    rows = view.split(split)
    if rule is not None and rule not in RULES:
        raise ObservationToolError(f"rule must be one of {list(RULES)}")
    saved = str(view.scorer.rule)
    view.scorer.rule = rule or saved
    try:
        out = _judged(view, ctx.current(), rows)
    finally:
        view.scorer.rule = saved
    return {
        "split": split,
        "rule": rule or saved,
        "primary_metric": judging.PRIMARY_METRIC,
        **out,
        **_notes(ctx, saved),
    }


def compare_to_parent_v3(ctx: Context, split: str = "validation") -> dict[str, Any]:
    view = ctx.view
    rows = view.split(split)
    tensors = ctx.current()
    now = _judged(view, tensors, rows)
    before = _judged(view, ctx.parent, rows)

    def fired(t: Mapping[str, np.ndarray], bar: float | None) -> np.ndarray:
        scores = view.scorer.score(t, rows).astype(np.float64)
        out: np.ndarray = np.zeros(rows.size, dtype=bool) if bar is None else scores >= bar
        return out

    a = fired(tensors, now["at_own_bar"]["bar"])
    b = fired(ctx.parent, before["at_own_bar"]["bar"])
    flipped = [int(r) for r, x, y in zip(rows, a, b, strict=True) if x != y]

    def delta(x: Any, y: Any) -> float | None:
        return None if x is None or y is None else float(x) - float(y)

    return {
        "split": split,
        "primary_metric": judging.PRIMARY_METRIC,
        "now": now,
        "parent": before,
        "delta": {
            "auroc": delta(now["auroc"], before["auroc"]),
            "recall_at_own_bar": delta(now["at_own_bar"]["recall"], before["at_own_bar"]["recall"]),
            "false_alarm_rate_at_own_bar": delta(
                now["at_own_bar"]["false_alarm_rate"], before["at_own_bar"]["false_alarm_rate"]
            ),
        },
        "flipped": len(flipped),
        "flipped_rows": flipped[:MAX_ROW_IDS],
        **_notes(ctx, str(view.scorer.rule)),
    }


def practice_metrics_v4(
    ctx: Context, split: str = "validation", rule: str | None = None
) -> dict[str, Any]:
    return practice_metrics_v3(ctx, split, rule)


def uncertainty(
    after: np.ndarray, before: np.ndarray, labels: np.ndarray, seed: int
) -> dict[str, Any]:
    """The ``auroc_against_parent`` block, and the words for it when it is within noise."""
    interval = practice_interval.against_parent(after, before, labels, seed)
    out: dict[str, Any] = {"auroc_against_parent": interval}
    if interval is not None and interval["within_noise"]:
        out["noise_note"] = judging.WITHIN_NOISE
    return out


def compare_to_parent_v4(ctx: Context, split: str = "validation") -> dict[str, Any]:
    view = ctx.view
    rows = view.split(split)
    out = compare_to_parent_v3(ctx, split)
    after = view.scorer.score(ctx.current(), rows)
    before = view.scorer.score(ctx.parent, rows)
    return {**out, **uncertainty(after, before, view.labels[rows], ctx.seed)}


def stratified_false_alarms_v2(ctx: Context, by: str = "length_band") -> dict[str, Any]:
    """Version 1's strata, at the head's OWN judged bar (``score >= bar``)."""
    if by not in STRATA:
        raise ObservationToolError(
            f"by must be one of {list(STRATA)} (template positions are not in a practice bundle)"
        )
    view = ctx.view
    tensors = ctx.current()
    rows = view.split("all")
    scores = view.scorer.score(tensors, rows)
    bar = view.judged_bar(tensors)
    negatives = ~np.asarray(view.labels[rows]).astype(bool)
    strata: dict[str, dict[str, Any]] = {}
    for i, r in enumerate(rows):
        if not negatives[i]:
            continue
        if by == "truncation":
            key = "truncated" if bool(view.truncated[r]) else "complete"
        else:
            n = view.row_length(int(r))
            low, high = next(b for b in LENGTH_BANDS if b[0] <= n <= b[1])
            key = f"len_{low}_{high}" if high < (1 << 31) else f"len_{low}_up"
        cell = strata.setdefault(key, {"negatives": 0, "false_alarms": 0})
        cell["negatives"] += 1
        cell["false_alarms"] += int(bar is not None and float(scores[i]) >= bar)
    for cell in strata.values():
        cell["rate"] = cell["false_alarms"] / cell["negatives"]
    return {
        "by": by,
        "at_own_bar": {
            "bar": bar,
            "basis": BASIS,
            "target_false_alarm_rate": float(view.target_fpr),
            "fires": "at_or_above",
        },
        "strata": strata,
        **_notes(ctx, str(view.scorer.rule)),
    }


def _supplement_cell(scores: np.ndarray, labels: np.ndarray, bar: float | None) -> dict[str, Any]:
    return {"auroc": auroc(scores, labels), "bar": bar, **rates_at_or_above(scores, labels, bar)}


def supplement_metrics(ctx: Context) -> dict[str, Any]:
    """Revision 2026-11-25 (FU-7a C2; measure by cell): each validation-only supplement on its
    own, the current head and the parent, each at ITS OWN judged bar (re-cut on the practice
    validation negatives, as practice_metrics 3 and 4 cut it), over the supplement's rows that
    were not truncated (the verdict leaves truncated rows out). A supplement is never in a
    practice split, so no other tool's figure includes it. A bundle without supplements answers
    ``supplements: 0``."""
    view = ctx.view
    supplement = view.supplement
    head = {
        "basis": BASIS,
        "target_false_alarm_rate": float(view.target_fpr),
        "fires": "at_or_above",
    }
    if supplement is None:
        return {"supplements": 0, "sources": {}, **head}
    tensors = ctx.current()
    bars = {"now": view.judged_bar(tensors), "parent": view.judged_bar(ctx.parent)}
    sources: dict[str, Any] = {}
    first = 0
    for index, entry in enumerate(supplement.entries):
        n = int(entry["rows"])
        rows = np.arange(first, first + n, dtype=np.int64)
        first += n
        counted = rows[~np.asarray(supplement.truncated, dtype=bool)[rows]]
        labels = np.asarray(supplement.labels)[counted]
        cell: dict[str, Any] = {"rows": n, "rows_counted": int(counted.size)}
        view_id = str(entry.get("view_id", ""))
        if NAME.match(view_id):
            cell["view_id"] = view_id
        for side, weights in (("now", tensors), ("parent", ctx.parent)):
            cell[side] = _supplement_cell(
                supplement.scorer.score(weights, counted), labels, bars[side]
            )
        sources[f"supplement_{index}"] = cell
    return {"supplements": len(supplement.entries), "sources": sources, **head}


#: The judged figures, in words, for the version-3 descriptions.
_JUDGED = (
    "'auroc' is the verdict's primary metric. 'at_own_bar' is the head at ITS OWN bar, re-cut on "
    "the current weights to the probe's target false-alarm rate on the practice validation "
    "negatives that were not truncated, firing at score >= bar: the way the held-out verdict "
    "re-cuts every head's bar. Its false_alarm_rate stays near the target on the validation "
    "split by construction; read its recall. A uniform bias shift (or a positive rescale of "
    "weight and bias) changes none of these figures, and the verdict cannot see it either."
)


class Entry(NamedTuple):
    cost: int
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., dict[str, Any]]


_SPLIT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"split": {"type": "string", "enum": list(SPLITS)}},
    "required": [],
}
_METRICS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "split": {"type": "string", "enum": list(SPLITS)},
        "rule": {"type": "string", "enum": list(RULES)},
    },
    "required": [],
}

#: (name, version) -> its cost, description, parameters and function. EVERY version a recorded run
#: may name stays here (fix 2026-11-21): a frozen request naming version 1 (a replay of a run
#: recorded before version 2) runs version 1 exactly, or the replay could not reproduce.
#: ``wsp.freeze.OBSERVATION_TOOLS`` is pinned equal to it.
CATALOGUE: dict[tuple[str, str], Entry] = {
    ("practice_metrics", "1"): Entry(
        1,
        "AUROC, recall and false-alarm rate at the practice-cut bar, on a practice split.",
        _METRICS_PARAMETERS,
        practice_metrics_v1,
    ),
    ("compare_to_parent", "1"): Entry(
        2,
        "Metrics now against the parent's on a practice split, and the rows whose decision "
        "flipped.",
        _SPLIT_PARAMETERS,
        compare_to_parent_v1,
    ),
    ("practice_metrics", "2"): Entry(
        1,
        "AUROC, recall and false-alarm rate on a practice split, at two bars. 'bar' is re-cut on "
        "the CURRENT weights at the target false-alarm rate on the train negatives, so its "
        "false_alarm_rate stays near the target whatever you edit: read its recall (recall at a "
        "fixed false-alarm rate). It moves with the scores, so a bias or scale edit changes "
        "nothing there. 'at_parent_bar' holds the PARENT's bar fixed (the probe's shipped bar, "
        "source 'shipped', firing at score >= bar; else the parent's own practice cut, source "
        "'parent_practice_cut', firing at score > bar): there a bias or scale edit shows as a "
        "change in false alarms and recall at the shipped operating point.",
        _METRICS_PARAMETERS,
        practice_metrics,
    ),
    ("stratified_false_alarms", "1"): Entry(
        1,
        "False alarms on the practice negatives, by length band or truncation.",
        {
            "type": "object",
            "properties": {"by": {"type": "string", "enum": list(STRATA)}},
            "required": [],
        },
        stratified_false_alarms,
    ),
    ("direction_cosines", "1"): Entry(
        1,
        "Cosines of head directions against the parent and against seeded random directions.",
        {
            "type": "object",
            "properties": {"names": {"type": "array", "items": {"type": "string"}}},
            "required": [],
        },
        direction_cosines,
    ),
    ("firing_positions", "1"): Entry(
        1,
        "The practice rows that score highest: row id, score, label and peak token position.",
        {
            "type": "object",
            "properties": {"top_k": {"type": "integer", "minimum": 1, "maximum": MAX_ROW_IDS}},
            "required": [],
        },
        firing_positions,
    ),
    ("compare_to_parent", "2"): Entry(
        2,
        "Metrics now against the parent's on a practice split, and the rows whose decision "
        "flipped. Each side's 'bar' is re-cut on its own weights at the target false-alarm rate "
        "(so false_alarm_rate stays near the target on both; compare recall), and "
        "'at_parent_bar' holds the parent's bar fixed for both (compare false alarms and recall "
        "at the shipped operating point; see practice_metrics).",
        _SPLIT_PARAMETERS,
        compare_to_parent,
    ),
    ("practice_metrics", "3"): Entry(
        1,
        "AUROC, and recall and false-alarm rate at the head's own re-cut bar, on a practice "
        "split. " + _JUDGED,
        _METRICS_PARAMETERS,
        practice_metrics_v3,
    ),
    ("compare_to_parent", "3"): Entry(
        2,
        "The current weights against the parent's on a practice split: each side's AUROC and "
        "each at its own re-cut bar, their differences, and the rows whose decision flipped. "
        + _JUDGED,
        _SPLIT_PARAMETERS,
        compare_to_parent_v3,
    ),
    ("practice_metrics", "4"): Entry(
        1,
        "AUROC, and recall and false-alarm rate at the head's own re-cut bar, on a practice "
        "split (validation unless you name another). " + _JUDGED,
        _METRICS_PARAMETERS,
        practice_metrics_v4,
    ),
    ("compare_to_parent", "4"): Entry(
        2,
        "The current weights against the parent's on a practice split (validation unless you name "
        "another): each side's AUROC and each at its own re-cut bar, their differences, "
        "'auroc_against_parent' (the AUROC difference's 95% paired-bootstrap interval; "
        "within_noise when it includes zero), and the rows whose decision flipped. " + _JUDGED,
        _SPLIT_PARAMETERS,
        compare_to_parent_v4,
    ),
    ("stratified_false_alarms", "2"): Entry(
        1,
        "False alarms on the practice negatives at the head's own re-cut bar, by length band or "
        "truncation. " + _JUDGED,
        {
            "type": "object",
            "properties": {"by": {"type": "string", "enum": list(STRATA)}},
            "required": [],
        },
        stratified_false_alarms_v2,
    ),
    ("supplement_metrics", "1"): Entry(
        2,
        "The bundle's validation-only supplements (rows from other sources, such as another "
        "generator's negatives; never in a practice split), one by one: for the current weights "
        "and the parent's, AUROC (when the supplement holds both classes) and the recall, false "
        "alarms and false-alarm rate at each one's OWN bar, re-cut on the practice validation "
        "negatives to the target false-alarm rate (firing at score >= bar), over the rows that "
        "were not truncated. 'supplements' is 0 when the bundle has none.",
        {"type": "object", "properties": {}, "required": []},
        supplement_metrics,
    ),
}


#: The version a start body gets when it names none: each tool's newest.
CURRENT: dict[str, str] = {
    name: max((v for n, v in CATALOGUE if n == name), key=int) for name, _ in CATALOGUE
}


def build(requested: list[dict[str, Any]], ctx: Context) -> list[ToolSpec]:
    """The request's named tools, each pinned to its catalogue version and cost."""
    specs: list[ToolSpec] = []
    for entry in requested:
        name, version, cost = entry.get("name"), entry.get("version"), entry.get("cost")
        known = CATALOGUE.get((str(name), str(version)))
        if known is None or known.cost != cost:
            raise ObservationToolError(
                f"observation tool {name!r} v{version} at cost {cost} is not in this worker's "
                "catalogue",
                "worker_observation_tool_unknown",
            )
        _cost, description, parameters, fn = known

        def bound(
            _fn: Callable[..., dict[str, Any]] = fn, _name: str = str(name), **kw: Any
        ) -> dict[str, Any]:
            answer: dict[str, Any] = numbers_and_names_only({"tool": _name, **_fn(ctx, **kw)})
            return answer

        specs.append(ToolSpec(str(name), str(version), cost, description, parameters, bound))
    return specs
