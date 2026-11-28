"""A practice gain is shown with its uncertainty (plan A1, 2026-11-22; D128 round 9).

Run ``wsp_54b6c1e5f0c9`` kept an edit for AUROC 0.99033 -> 0.99053 on the 4,982 practice
validation rows. The paired-bootstrap interval of that difference is [-0.00012, +0.00053]: noise.
Held out it was significantly worse on two of three primary sets. 51 of that run's measurements
were ``test_program``, so program style's tool carries the interval as well as
``compare_to_parent``; and ``wsp_4bd06d60acf0`` measured on train only, ``practice_metrics``'
default, the rows the parent was fitted on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.support.wsp_engine import line_for
from tests.support.wsp_fixtures import make_bundle
from tests.unit.wsp_worker.test_judged_measure import _ctx, _turn, _view
from wsp import verdict
from wsp_worker import judging, practice_interval, program
from wsp_worker import observation_tools as ot
from wsp_worker.miwsp_engine import paging
from wsp_worker.miwsp_engine import session as session_mod

V4 = [
    {"name": "practice_metrics", "version": "4", "cost": 1},
    {"name": "compare_to_parent", "version": "4", "cost": 2},
]
V3 = [{**t, "version": "3"} for t in V4]
FLIP = "def edit(base, support):\n    return {'weight': -np.array(base['weight'])}\n"


def _scores(seed: int, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.random(400) < 0.2
    before = (rng.normal(size=400) + 1.2 * labels).astype(np.float32)
    after = (before + rng.normal(scale=0.2, size=400) + shift * labels).astype(np.float32)
    after[:40] = after[0]  # ties
    return after, before, labels


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_the_interval_is_the_verdicts_own(seed: int) -> None:
    after, before, labels = _scores(seed)
    got = practice_interval.against_parent(after, before, labels, seed)
    assert got is not None
    pos, neg = np.flatnonzero(labels), np.flatnonzero(~labels)
    rng, deltas = np.random.default_rng(seed), []
    a, b = verdict._Ranked(after, labels), verdict._Ranked(before, labels)
    for _ in range(practice_interval.RESAMPLES // practice_interval.BATCH):
        counts = verdict._counts(rng, pos, neg, labels.size, practice_interval.BATCH)
        deltas.append(a.auroc(counts) - b.auroc(counts))
    low, high = verdict._interval(np.concatenate(deltas))
    assert got["low"] == pytest.approx(low, abs=1e-12)
    assert got["high"] == pytest.approx(high, abs=1e-12)


def test_its_constants_are_the_verdicts() -> None:
    from wsp.settings import WSPSettings

    assert (practice_interval.CI_LOW, practice_interval.CI_HIGH) == (
        verdict.CI_LOW,
        verdict.CI_HIGH,
    )
    assert practice_interval.RESAMPLES == WSPSettings().wsp_bootstrap_b


def test_noise_and_real_differences_are_told_apart() -> None:
    _, before, labels = _scores(3)
    same = practice_interval.against_parent(before, before, labels, 1)
    assert same is not None and same["within_noise"] is True
    assert (same["delta"], same["low"], same["high"]) == (0.0, 0.0, 0.0)
    better = practice_interval.against_parent(before + 2.0 * labels, before, labels, 1)
    worse = practice_interval.against_parent(before - 2.0 * labels, before, labels, 1)
    assert better is not None and better["within_noise"] is False and better["low"] > 0
    assert worse is not None and worse["within_noise"] is False and worse["high"] < 0


def test_it_is_deterministic_and_absent_without_both_classes() -> None:
    after, before, labels = _scores(2)
    first = practice_interval.against_parent(after, before, labels, 11)
    assert first == practice_interval.against_parent(after, before, labels, 11)
    assert first != practice_interval.against_parent(after, before, labels, 12)
    assert practice_interval.against_parent(after, before, np.zeros(400, bool), 1) is None


def test_compare_to_parent_4_carries_the_interval(tmp_path: Path) -> None:
    view, fx = _view(tmp_path)
    ctx = _ctx(view, fx, _turn)
    answer = ot.CATALOGUE[("compare_to_parent", "4")].fn(ctx)
    rows = view.split("validation")
    expected = practice_interval.against_parent(
        view.scorer.score(ctx.current(), rows),
        view.scorer.score(ctx.parent, rows),
        view.labels[rows],
        ctx.seed,
    )
    assert expected is not None and answer["auroc_against_parent"] == expected
    assert answer["delta"]["auroc"] == pytest.approx(expected["delta"], abs=1e-12)
    ot.numbers_and_names_only({"tool": "compare_to_parent", **answer})


def test_a_difference_within_noise_says_so_in_words(tmp_path: Path) -> None:
    view, fx = _view(tmp_path)
    answer = ot.CATALOGUE[("compare_to_parent", "4")].fn(_ctx(view, fx, lambda p: dict(p)))
    assert answer["auroc_against_parent"]["within_noise"] is True
    assert answer["noise_note"] == judging.WITHIN_NOISE
    assert judging.WITHIN_NOISE in judging.FIXED_TEXTS
    ot.numbers_and_names_only({"tool": "compare_to_parent", **answer})


def test_version_3_answers_as_it_did(tmp_path: Path) -> None:
    view, fx = _view(tmp_path)
    ctx = _ctx(view, fx, _turn)
    assert "auroc_against_parent" not in ot.CATALOGUE[("compare_to_parent", "3")].fn(ctx)
    assert ot.CATALOGUE[("practice_metrics", "3")].fn(ctx)["split"] == "train"


def test_practice_metrics_4_measures_on_validation_unless_asked(tmp_path: Path) -> None:
    view, fx = _view(tmp_path)
    tool = ot.CATALOGUE[("practice_metrics", "4")].fn
    assert tool(_ctx(view, fx, _turn))["split"] == "validation"
    assert tool(_ctx(view, fx, _turn), split="train")["split"] == "train"


def test_version_4_is_what_a_start_body_gets() -> None:
    from wsp.freeze import CURRENT_OBSERVATION_TOOLS, OBSERVATION_TOOLS

    for name, cost in (("practice_metrics", 1), ("compare_to_parent", 2)):
        assert ot.CURRENT[name] == CURRENT_OBSERVATION_TOOLS[name] == "4"
        assert OBSERVATION_TOOLS[(name, "4")] == cost == ot.CATALOGUE[(name, "4")].cost


def _programs(tmp: Path, tools: list[dict[str, Any]]) -> tuple[Any, Any]:
    fx = make_bundle(tmp / "b", rule="mean")
    line = line_for(
        tmp / "run", fx, "repair", None, python=False, mode="program", observation_tools=tools
    )
    s = session_mod.Session(line)
    assert s.programs is not None
    return s.programs, fx


def test_test_program_carries_the_interval(tmp_path: Path) -> None:
    programs, fx = _programs(tmp_path, V4)
    answer = programs.test_program(FLIP)
    got = answer["auroc_against_parent"]
    assert got["delta"] == pytest.approx(answer["auroc"] - answer["parent"]["auroc"], abs=1e-12)
    parent = {n: t.values64() for n, t in fx.tensors.items()}
    rows = programs._validation_rows
    expected = practice_interval.against_parent(
        programs._scorer.score({**parent, "weight": -parent["weight"]}, rows),
        programs._scorer.score(parent, rows),
        programs._labels[rows],
        program.seed_for(fx.manifest["bundle_sha256"]),
    )
    assert got == expected


def test_test_program_says_noise_in_words(tmp_path: Path) -> None:
    programs, _ = _programs(tmp_path, V4)
    same = "def edit(base, support):\n    return {'weight': np.array(base['weight'])}\n"
    answer = programs.test_program(same)
    assert answer["auroc_against_parent"]["within_noise"] is True
    assert answer["noise_note"] == judging.WITHIN_NOISE


def test_test_program_of_an_older_run_answers_as_it_did(tmp_path: Path) -> None:
    programs, _ = _programs(tmp_path, V3)
    assert "auroc_against_parent" not in programs.test_program(FLIP)


def test_the_section_says_how_to_choose_only_for_version_4(tmp_path: Path) -> None:
    fx = make_bundle(tmp_path / "b", rule="mean")
    request = {"kind": "repair", "primary_sets": ["eval"]}
    new = judging.section({**request, "observation_tools": V4}, fx.manifest)
    old = judging.section({**request, "observation_tools": V3}, fx.manifest)
    assert new is not None and old is not None
    assert judging.CHOOSE_ON_VALIDATION in new and judging.CHOOSE_ON_VALIDATION not in old
    assert old == judging.section(request, fx.manifest)


@pytest.mark.parametrize(
    ("low", "high", "word"), [(-0.1, 0.1, "noise"), (0.01, 0.1, "above"), (-0.1, -0.01, "below")]
)
def test_the_ledger_keeps_the_answer_to_is_it_real(low: float, high: float, word: str) -> None:
    interval = {"delta": 0.0, "low": low, "high": high, "within_noise": word == "noise"}
    nested = {"now": {"auroc": 0.9}, "auroc_against_parent": interval}
    flat = {"auroc": 0.9, "auroc_against_parent": interval}
    for answer in (nested, flat):
        assert ("vs_parent", word) in paging.numbers_of(json.dumps(answer))


def test_test_program_seeds_from_the_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture's interval is the same at every seed, so the seed is asserted at the call."""
    programs, fx = _programs(tmp_path, V4)
    seeds: list[Any] = []
    monkeypatch.setattr(practice_interval, "against_parent", lambda *a: seeds.append(a[3]))
    programs.test_program(FLIP)
    assert seeds == [program.seed_for(fx.manifest["bundle_sha256"])]
