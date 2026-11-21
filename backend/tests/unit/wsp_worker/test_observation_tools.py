"""Observation tools (feature 016; FTASKS 8.11; FR-016.18's AC; WR-8, WR-14).

- an instrumented scorer records every practice row a tool touches: a subset of
  ``[0, practice.rows)``;
- every answer passes the schema: no string except names, rule ids and row ids, and no list of
  floats (a vector), checked here independently of the tools' own guard;
- a canary written into the bundle's row-id positions never reaches an answer as anything but an
  integer id; no answer holds an activation row;
- a tool's request entry must match the catalogue's version and cost (the cost the adapter charges;
  cost accounting through the library itself is in ``test_adapter_scripted.py``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.support.wsp_fixtures import make_bundle
from wsp_worker import observation_tools as ot
from wsp_worker.scoring import Scorer, bar_at

NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


class RecordingScorer(Scorer):
    """Records every row whose activations are read. ``read_rows`` is the scorer's one read of
    the map (``test_worker_scoring.py`` replaces it and proves every score flows through it), so
    a row no call recorded was never read."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.touched: set[int] = set()

    def read_rows(self, first: int, stop: int) -> np.ndarray:
        self.touched.update(range(int(first), int(stop)))
        return super().read_rows(first, stop)


def _ctx(tmp_path: Path, rule: str = "mean") -> tuple[ot.Context, Any, RecordingScorer]:
    fx = make_bundle(tmp_path / "b", rule=rule)
    validation = np.zeros(fx.manifest["practice"]["rows"], dtype=bool)
    validation[:6] = True
    truncated = np.zeros_like(validation)
    truncated[::5] = True
    scorer = RecordingScorer(fx.manifest["function"], fx.activations, fx.offsets)
    view = ot.PracticeView(scorer, fx.labels, validation, truncated, target_fpr=0.1)
    parent = {n: t.values64() for n, t in fx.tensors.items()}
    edited = {**parent, "weight": parent["weight"] + 0.25}
    return ot.Context(view, lambda: dict(edited), parent, seed=7), fx, scorer


CALLS = [
    ("practice_metrics", {"split": "train"}),
    ("practice_metrics", {"split": "validation", "rule": "max"}),
    ("stratified_false_alarms", {"by": "length_band"}),
    ("stratified_false_alarms", {"by": "truncation"}),
    ("direction_cosines", {"names": ["weight"]}),
    ("firing_positions", {"top_k": 5}),
    ("compare_to_parent", {"split": "validation"}),
]


def _tools(ctx: ot.Context) -> dict[str, ot.ToolSpec]:
    requested = [
        {"name": n, "version": v, "cost": ot.CATALOGUE[(n, v)].cost} for n, v in ot.CURRENT.items()
    ]
    return {s.name: s for s in ot.build(requested, ctx)}


def _walk(value: Any, path: str = "$") -> None:
    """The schema, restated: numbers, booleans, null, names; lists never of floats."""
    if value is None or isinstance(value, bool | int | float):
        return
    if isinstance(value, str):
        assert NAME.match(value), f"{path}: {value!r}"
        return
    if isinstance(value, dict):
        for k, v in value.items():
            assert NAME.match(k), f"{path}: key {k!r}"
            _walk(v, f"{path}.{k}")
        return
    assert isinstance(value, list), f"{path}: {type(value)}"
    for i, item in enumerate(value):
        assert not isinstance(item, float), f"{path}[{i}] is a float in a list"
        _walk(item, f"{path}[{i}]")


@pytest.mark.parametrize(
    ("name", "args"), CALLS, ids=[f"{n}-{i}" for i, (n, _) in enumerate(CALLS)]
)
def test_each_tool_touches_only_practice_rows_and_answers_numbers_and_names(
    tmp_path: Path, name: str, args: dict[str, Any]
) -> None:
    ctx, fx, scorer = _ctx(tmp_path)
    answer = _tools(ctx)[name].fn(**args)
    assert answer["tool"] == name
    json.dumps(answer)  # plain JSON
    _walk(answer)
    assert scorer.touched <= set(range(fx.manifest["practice"]["rows"]))
    # No answer holds an activation row, as a list or as a dict's values.
    flat: list[float] = []

    def numbers(v: Any) -> None:
        if isinstance(v, dict):
            for x in v.values():
                numbers(x)
        elif isinstance(v, list):
            for x in v:
                numbers(x)
        elif isinstance(v, float):
            flat.append(v)

    numbers(answer)
    d_model = fx.manifest["practice"]["d_model"]
    for r in range(min(4, fx.manifest["practice"]["rows"])):
        row = fx.activations[int(fx.offsets[r])].astype(np.float64)
        assert not all(any(abs(f - x) < 1e-6 for f in flat) for x in row[:d_model])


def test_the_tools_together_touch_rows_and_row_ids_pass_only_as_integers(tmp_path: Path) -> None:
    ctx, fx, scorer = _ctx(tmp_path)
    tools = _tools(ctx)
    answers = [tools[n].fn(**a) for n, a in CALLS]
    assert scorer.touched and scorer.touched <= set(range(fx.manifest["practice"]["rows"]))
    text = json.dumps(answers)
    # The canary: a row-id view name and any practice text never appear (the bundle carries none).
    assert fx.manifest["practice"]["view_id"] not in text
    for answer in answers:
        for item in answer.get("top", []):
            assert isinstance(item["row"], int)
        assert all(isinstance(r, int) for r in answer.get("flipped_rows", []))


def test_the_guard_refuses_a_vector_and_free_text() -> None:
    with pytest.raises(ot.ObservationToolError):
        ot.numbers_and_names_only({"activation": [0.1, 0.2]})
    with pytest.raises(ot.ObservationToolError):
        ot.numbers_and_names_only({"note": "the user said hello"})
    assert ot.numbers_and_names_only({"rows": [1, 2], "rule": "max", "x": 0.5}) is not None


def test_a_request_must_match_the_catalogue_version_and_cost(tmp_path: Path) -> None:
    ctx, _, _ = _ctx(tmp_path)
    for bad in (
        {"name": "practice_metrics", "version": "3", "cost": 1},
        {"name": "practice_metrics", "version": "2", "cost": 0},
        {"name": "read_practice_text", "version": "1", "cost": 1},
    ):
        with pytest.raises(ot.ObservationToolError) as caught:
            ot.build([bad], ctx)
        assert caught.value.code == "worker_observation_tool_unknown"
    built = ot.build([{"name": "compare_to_parent", "version": "2", "cost": 2}], ctx)
    assert [(s.name, s.cost) for s in built] == [("compare_to_parent", 2)]


def test_bad_arguments_are_refused(tmp_path: Path) -> None:
    ctx, _, _ = _ctx(tmp_path)
    tools = _tools(ctx)
    for name, args in (
        ("practice_metrics", {"split": "heldout"}),
        ("practice_metrics", {"rule": "median"}),
        ("stratified_false_alarms", {"by": "template_position"}),
        ("direction_cosines", {"names": ["not_a_tensor"]}),
        ("firing_positions", {"top_k": 10_000}),
    ):
        with pytest.raises(ot.ObservationToolError):
            tools[name].fn(**args)


def test_the_backend_catalogue_equals_the_workers() -> None:
    """``freeze`` refuses an unknown tool before a run is queued; both lists are one list."""
    from wsp.freeze import CURRENT_OBSERVATION_TOOLS, OBSERVATION_TOOLS

    assert OBSERVATION_TOOLS == {key: entry.cost for key, entry in ot.CATALOGUE.items()}
    assert CURRENT_OBSERVATION_TOOLS == ot.CURRENT


# ------------------------------------------------------------------ review 016-P2 (18.4, r016p2)


class _RowScorer:
    """Scores each row by its index, and every validation row at 1000: a bar that saw a
    validation row would sit at 1000."""

    def __init__(self, rows: int, validation: np.ndarray) -> None:
        self.rows = rows
        self.validation = validation

    def score(self, tensors: Any, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        return np.where(self.validation[rows], 1000.0, rows.astype(np.float64))


def test_the_splits_and_the_bar_keep_the_validation_rows_apart() -> None:
    """Held-out isolation inside the worker (controls I2, I3): ``validation`` is exactly the
    validation rows and ``train`` exactly the rest, and the practice-cut bar the editor's tools
    report is cut on the TRAIN split only, so no validation score moves it."""
    rows = 40
    validation = np.zeros(rows, dtype=bool)
    validation[[3, 9, 17, 33]] = True
    labels = np.zeros(rows, dtype=np.int64)
    labels[::4] = 1
    view = ot.PracticeView(
        _RowScorer(rows, validation),  # type: ignore[arg-type]
        labels,
        validation,
        np.zeros(rows, dtype=bool),
        target_fpr=0.1,
    )
    assert view.split("validation").tolist() == [3, 9, 17, 33]
    assert view.split("train").tolist() == [r for r in range(rows) if r not in (3, 9, 17, 33)]
    assert view.split("all").tolist() == list(range(rows))
    train = view.split("train")
    expected = bar_at(train.astype(np.float64), labels[train], 0.1)
    assert view.bar({}) == expected and expected is not None and expected < 1000.0


def test_a_repeated_tool_call_on_unchanged_weights_reads_no_activation(tmp_path: Path) -> None:
    """Fix 2026-11-21: the scorer caches each head's token scores, so the second call of a tool
    (and the parent's half of ``compare_to_parent`` after the first) reads nothing; an edit makes
    the next call read again."""
    ctx, fx, scorer = _ctx(tmp_path)
    tools = _tools(ctx)
    first = tools["practice_metrics"].fn(split="validation")
    assert scorer.touched == set(range(fx.manifest["practice"]["rows"]))
    scorer.touched.clear()
    assert tools["practice_metrics"].fn(split="validation") == first
    tools["firing_positions"].fn(top_k=3)
    assert scorer.touched == set()
    tools["compare_to_parent"].fn(split="validation")  # the parent is a new head: read once
    assert scorer.touched
    scorer.touched.clear()
    tools["compare_to_parent"].fn(split="validation")
    assert scorer.touched == set()


#: After a bias edit of +0.3 on the fixture (computed independently with numpy: the parent's train
#: scores plus 0.3, against 0.0 with >= and against the parent's cut with >).
GOLDEN_SHIPPED = {"bar": 0.0, "source": "shipped", "recall": 0.875, "false_alarm_rate": 0.6}
GOLDEN_CUT = {
    "bar": 0.9454358816146851,
    "source": "parent_practice_cut",
    "recall": 0.125,
    "false_alarm_rate": 0.4,
}


# ------------------- 2026-11-21: the rates at the parent's bar, held fixed (run wsp_855a74f820ff)


def _fixed_ctx(
    tmp_path: Path, edit: Any, *, shipped: bool
) -> tuple[ot.Context, dict[str, np.ndarray], Any]:
    """The fixture's practice view with the probe's shipped bar (``operating_point.threshold``)
    or without one, and ``edit(parent)`` as the current tensors."""
    fx = make_bundle(tmp_path / "b", rule="mean")
    validation = np.zeros(fx.manifest["practice"]["rows"], dtype=bool)
    validation[:6] = True
    scorer = Scorer(fx.manifest["function"], fx.activations, fx.offsets)
    parent = {n: t.values64() for n, t in fx.tensors.items()}
    view = ot.PracticeView(
        scorer,
        fx.labels,
        validation,
        np.zeros_like(validation),
        target_fpr=0.1,
        shipped_bar=ot.shipped_bar_of(fx.manifest) if shipped else None,
    )
    edited = edit(parent)
    return ot.Context(view, lambda: dict(edited), parent, seed=7), parent, fx


def _bias(delta: float) -> Any:
    return lambda p: {**p, "bias": p["bias"] + delta}


def _scale(k: float) -> Any:
    return lambda p: {**p, "weight": p["weight"] * k, "bias": p["bias"] * k}


def test_the_fixture_carries_a_shipped_bar_and_an_sae_bundle_none() -> None:
    assert ot.shipped_bar_of({"operating_point": {"threshold": 0.25}}) == 0.25
    assert ot.shipped_bar_of({"operating_point": None}) is None
    assert ot.shipped_bar_of({"operating_point": {"threshold": True}}) is None
    assert ot.shipped_bar_of({}) is None


@pytest.mark.parametrize("shipped", [True, False], ids=["shipped", "parent-practice-cut"])
def test_a_bias_edit_moves_the_fixed_bar_rates_and_not_the_recut_ones(
    tmp_path: Path, shipped: bool
) -> None:
    """Run wsp_855a74f820ff: the re-cut bar moves with a bias edit, so its false-alarm rate read
    the target in every measurement and the editor's bias edits were invisible. At the parent's
    bar, held fixed, a raised bias raises the false alarms."""
    before, _, _ = _fixed_ctx(tmp_path / "a", lambda p: p, shipped=shipped)
    after, _, _ = _fixed_ctx(tmp_path / "b", _bias(5.0), shipped=shipped)
    tool = ot.CATALOGUE[("practice_metrics", "2")].fn
    was, now = tool(before, split="train"), tool(after, split="train")
    assert (now["recall"], now["false_alarm_rate"]) == (was["recall"], was["false_alarm_rate"])
    assert now["false_alarm_rate"] <= 0.1
    fixed_was, fixed_now = was["at_parent_bar"], now["at_parent_bar"]
    assert fixed_now["bar"] == fixed_was["bar"]
    assert fixed_now["source"] == ("shipped" if shipped else "parent_practice_cut")
    assert fixed_now["false_alarm_rate"] > fixed_was["false_alarm_rate"]
    assert fixed_now["recall"] >= fixed_was["recall"]
    # compare_to_parent holds the same bar for both sides.
    compared = ot.CATALOGUE[("compare_to_parent", "2")].fn(after, split="train")
    assert compared["parent"]["at_parent_bar"] == fixed_was
    assert compared["now"]["at_parent_bar"] == fixed_now


@pytest.mark.parametrize("shipped", [True, False], ids=["shipped", "parent-practice-cut"])
def test_a_scale_edit_leaves_the_recut_rates_and_moves_the_fixed_bar_as_its_math_says(
    tmp_path: Path, shipped: bool
) -> None:
    """Scaling the weight and the bias by k > 0 scales every score by k: the ranking, the AUROC
    and the re-cut rates are unchanged. The rates at a fixed bar t are then the parent's at t/k;
    a nonzero bar moves (the fixture's shipped bar is 0.0, where a scale cannot show)."""
    before, _, _ = _fixed_ctx(tmp_path / "a", lambda p: p, shipped=shipped)
    after, _, _ = _fixed_ctx(tmp_path / "b", _scale(3.0), shipped=shipped)
    tool = ot.CATALOGUE[("practice_metrics", "2")].fn
    was, now = tool(before, split="train"), tool(after, split="train")
    for key in ("auroc", "recall", "false_alarm_rate"):
        assert now[key] == pytest.approx(was[key])
    bar = was["at_parent_bar"]["bar"]
    rows = before.view.split("train")
    scores = before.view.scorer.score(before.parent, rows)
    labels = before.view.labels[rows]
    fires = scores * 3.0 >= bar if shipped else scores * 3.0 > bar
    assert now["at_parent_bar"]["false_alarm_rate"] == pytest.approx(
        fires[~labels.astype(bool)].mean()
    )
    if bar != 0.0:
        assert now["at_parent_bar"] != was["at_parent_bar"]


def test_the_fixed_bar_rates_on_the_fixture_golden(tmp_path: Path) -> None:
    """Golden numbers (make_bundle, rule mean, target 0.1, the train split: 8 positives and 10
    negatives, the parent's scores -2.40 to 2.96), after a bias edit of +0.3."""
    ctx, _, fx = _fixed_ctx(tmp_path, _bias(0.3), shipped=True)
    answer = ot.CATALOGUE[("practice_metrics", "2")].fn(ctx, split="train")
    assert fx.manifest["operating_point"]["threshold"] == 0.0
    assert answer["at_parent_bar"] == GOLDEN_SHIPPED
    ctx, _, _ = _fixed_ctx(tmp_path / "x", _bias(0.3), shipped=False)
    cut = ot.CATALOGUE[("practice_metrics", "2")].fn(ctx, split="train")["at_parent_bar"]
    assert cut == GOLDEN_CUT


def test_another_rule_is_held_at_the_parents_own_cut_at_that_rule(tmp_path: Path) -> None:
    """The shipped bar is for the probe's own rule; a practice_metrics call at another rule holds
    the parent's practice cut at that rule (cut once per rule)."""
    ctx, parent, _ = _fixed_ctx(tmp_path, _bias(5.0), shipped=True)
    answer = ot.CATALOGUE[("practice_metrics", "2")].fn(ctx, split="train", rule="max")
    assert answer["at_parent_bar"]["source"] == "parent_practice_cut"
    ctx.view.scorer.rule = "max"
    assert answer["at_parent_bar"]["bar"] == ctx.view.bar(parent)


def test_the_engine_view_carries_the_bundles_shipped_bar(tmp_path: Path) -> None:
    """Wiring: the engine's observation context reads operating_point.threshold."""
    from tests.support.wsp_engine import line_for
    from wsp_worker.miwsp_engine import session as session_mod

    fx = make_bundle(tmp_path / "b", rule="mean")
    s = session_mod.Session(line_for(tmp_path / "run", fx, "repair", None, python=False))
    view = session_mod.observation_context(s.bundle, s.store).view
    assert view.shipped_bar is not None
    assert view.shipped_bar == fx.manifest["operating_point"]["threshold"]


def test_the_shipped_bar_fires_at_or_above_and_the_practice_cut_strictly_above() -> None:
    """The evaluator's rule for the shipped bar (``score >= bar``) and the practice cut's
    (``score > bar``, :func:`wsp_worker.scoring.bar_at`): a score exactly at the bar."""
    scores, labels = np.array([0.5, 0.5]), np.array([1, 0])
    shipped = ot.FixedBar(0.5, "shipped", inclusive=True).rates(scores, labels)
    cut = ot.FixedBar(0.5, "parent_practice_cut", inclusive=False).rates(scores, labels)
    assert (shipped["recall"], shipped["false_alarm_rate"]) == (1.0, 1.0)
    assert (cut["recall"], cut["false_alarm_rate"]) == (0.0, 0.0)


def test_compare_to_parent_called_first_holds_the_parents_cut_for_both_sides(
    tmp_path: Path,
) -> None:
    """The parent's practice cut is computed once per rule, by whichever call comes first: a
    session whose first tool call is compare_to_parent must cut it from the PARENT's tensors."""
    ctx, parent, _ = _fixed_ctx(tmp_path, _bias(0.3), shipped=False)
    compared = ot.CATALOGUE[("compare_to_parent", "2")].fn(ctx, split="train")
    expected = ctx.view.bar(parent)
    assert compared["now"]["at_parent_bar"]["bar"] == expected
    assert compared["parent"]["at_parent_bar"]["bar"] == expected
    assert compared["now"]["at_parent_bar"] == GOLDEN_CUT


#: Version 1's answers on ``_ctx`` (weight + 0.25), computed with the catalogue as it was at
#: ``908f003``, before version 2: a replay of a recorded run must get exactly these.
V1_GOLDEN = [
    (
        "practice_metrics",
        {"split": "train"},
        {
            "split": "train",
            "rule": "mean",
            "rows": 18,
            "auroc": 0.45,
            "bar": 0.7005069851875305,
            "recall": 0.125,
            "false_alarm_rate": 0.1,
        },
    ),
    (
        "practice_metrics",
        {"split": "validation", "rule": "max"},
        {
            "split": "validation",
            "rule": "max",
            "rows": 6,
            "auroc": 0.0,
            "bar": 4.163820743560791,
            "recall": 0.0,
            "false_alarm_rate": 0.0,
        },
    ),
    (
        "compare_to_parent",
        {"split": "validation"},
        {
            "split": "validation",
            "now": {
                "rows": 6,
                "auroc": 1.0,
                "bar": 0.7005069851875305,
                "recall": 1.0,
                "false_alarm_rate": 0.0,
            },
            "parent": {
                "rows": 6,
                "auroc": 1.0,
                "bar": 0.9454358816146851,
                "recall": 0.0,
                "false_alarm_rate": 0.0,
            },
            "flipped": 1,
            "flipped_rows": [5],
        },
    ),
]


@pytest.mark.parametrize(("name", "args", "golden"), V1_GOLDEN, ids=["pm-train", "pm-max", "ctp"])
def test_version_1_answers_exactly_as_it_did_before_version_2(
    tmp_path: Path, name: str, args: dict[str, Any], golden: dict[str, Any]
) -> None:
    ctx, _, _ = _ctx(tmp_path)
    (spec,) = ot.build(
        [{"name": name, "version": "1", "cost": ot.CATALOGUE[(name, "1")].cost}], ctx
    )
    answer = spec.fn(**args)
    assert answer == {"tool": name, **golden}
    # The same keys in the same order: the transcript records the JSON text.
    assert json.dumps(answer) == json.dumps({"tool": name, **golden})
    assert spec.version == "1"
