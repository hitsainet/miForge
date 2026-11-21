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

Library-free (numpy only): program mode's ``test_program`` reuses the same view in a child that
must never import ``weightspace`` (R1B-A9).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from wsp_worker.scoring import RULES, Scorer, auroc, bar_at, rates

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


def numbers_and_names_only(value: Any, path: str = "$") -> Any:
    """Refuse any string that is not a name, and any list holding a float (a vector)."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        if not NAME.match(value):
            raise ObservationToolError(f"{path}: a string that is not a name")
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not NAME.match(key):
                raise ObservationToolError(f"{path}: a key that is not a name")
            numbers_and_names_only(item, f"{path}.{key}")
        return value
    if isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(item, float):
                raise ObservationToolError(f"{path}[{i}]: a list of floats (a vector)")
            numbers_and_names_only(item, f"{path}[{i}]")
        return value
    raise ObservationToolError(f"{path}: a {type(value).__name__}")


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
