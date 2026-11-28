"""What the submitted weights measure, beside what the editor claims (plan A2, 2026-11-22).

Run ``wsp_4bd06d60acf0`` claimed AUROC 0.9904 with recall 0.229; its best measured AUROC was
0.990304 at recall 0.2109 (0.990434 was a cosine in its own output). Nothing measured what it
submitted. The worker now appends ``submitted_measured`` to the transcript (in both providers until
018-P4 retired ``weightspace``; its three cases went with its adapter).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.wsp_engine import line_for, transcript
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from tests.unit.wsp_worker.miwsp_engine.test_engine_scripted import call, session
from wsp_worker import submitted_measure
from wsp_worker.miwsp_engine import session as session_mod
from wsp_worker.miwsp_engine.transcript import ScriptedClient

V4 = [
    {"name": "practice_metrics", "version": "4", "cost": 1},
    {"name": "compare_to_parent", "version": "4", "cost": 2},
]
V3 = [{**t, "version": "3"} for t in V4]
PROGRAM = (
    "def edit(base, support):\n"
    "    w = np.array(base['weight'])\n"
    "    w[0] += np.float32(1.0)\n"
    "    return {'weight': w}\n"
)
LIE = "AUROC 0.9999 with recall 0.999"


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


def _events(run: Path) -> list[dict[str, Any]]:
    return [e for e in transcript(run) if e.get("event") == submitted_measure.EVENT]


def _turned(fx: FixtureBundle) -> dict[str, Any]:
    w = fx.tensors["weight"].values64().astype(float).tolist()
    w[0] += 1.0
    return {"calls": [("write_tensor", {"name": "weight", "values": w})]}


def test_false_claims_sit_beside_the_true_figures(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [_turned(fx), call("submit", answer=LIE)]
    _, run, _ = session(tmp_path, fx, "repair", steps, observation_tools=V4)
    (event,) = _events(run)
    assert event["measured"] is True and event["changed"] is True
    assert event["split"] == "validation"
    for side in ("now", "parent"):
        assert {"auroc", "recall", "false_alarms"} <= set(event[side])
        assert event[side]["auroc"] != 0.9999 and event[side]["recall"] != 0.999
    assert event["auroc_against_parent"]["delta"] == pytest.approx(
        event["now"]["auroc"] - event["parent"]["auroc"], abs=1e-12
    )
    assert transcript(run)[-1]["event"] == submitted_measure.EVENT  # last: nothing is fed back


def test_an_unchanged_submit_says_so(tmp_path: Path, fx: FixtureBundle) -> None:
    _, run, _ = session(tmp_path, fx, "repair", [call("submit", answer=LIE)], observation_tools=V4)
    (event,) = _events(run)
    assert event["changed"] is False and event["now"] == event["parent"]


def test_program_style_measures_the_programs_weights(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [call("submit_program", code=PROGRAM), call("submit", answer=LIE)]
    _, run, _ = session(tmp_path, fx, "repair", steps, mode="program", observation_tools=V4)
    (event,) = _events(run)
    assert event["measured"] is True and event["changed"] is True


def test_it_spends_none_of_the_editors_budget(tmp_path: Path, fx: FixtureBundle) -> None:
    _, run, _ = session(
        tmp_path, fx, "repair", [call("submit", answer="x")], observation_tools=V4, forward_passes=0
    )
    assert _events(run)[0]["measured"] is True
    assert json.loads((run / "out" / "result.json").read_text())["forward_used"] == 0


@pytest.mark.parametrize(("kind", "tools"), [("repair", V3), ("inspect", V4)])
def test_an_older_run_and_an_inspect_get_none(
    tmp_path: Path, fx: FixtureBundle, kind: str, tools: list[dict[str, Any]]
) -> None:
    _, run, _ = session(tmp_path, fx, kind, [call("submit", answer="x")], observation_tools=tools)
    assert _events(run) == []


def _in_process(tmp: Path, fx: FixtureBundle) -> session_mod.Session:
    line = line_for(tmp / "run", fx, "repair", None, python=False, observation_tools=V4)
    submit = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c", "name": "submit", "arguments": {"answer": LIE}}],
    }
    s = session_mod.Session(line, client=ScriptedClient([submit]))
    # Only the ending is under test: no kernel, no protocol check.
    s.replaying = True
    s._isolate_and_spawn = lambda: None  # type: ignore[method-assign]
    s._loop = lambda: None  # type: ignore[method-assign]
    return s


def test_the_outputs_are_written_before_it_measures(
    tmp_path: Path, fx: FixtureBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _in_process(tmp_path, fx)
    seen: list[bool] = []

    def measure(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append((s.out / "result.json").exists())
        return {"event": submitted_measure.EVENT}

    monkeypatch.setattr(submitted_measure, "event", measure)
    s.run()
    assert seen == [True]


def test_without_time_it_does_not_measure(tmp_path: Path, fx: FixtureBundle) -> None:
    s = _in_process(tmp_path, fx)
    s._time_left = lambda: 1.0  # type: ignore[method-assign]
    s.run()
    (event,) = _events(s.out.parent)
    # FU-2: the session offered submit's figures; this submit stated none.
    assert event == {
        "event": submitted_measure.EVENT,
        "measured": False,
        "reason": "no_time",
        "claims": {
            "stated": False,
            "stated_split": None,
            "compared": False,
            "differs": None,
            "checks": [],
        },
    }


def test_a_failed_measurement_never_fails_the_run() -> None:
    def broken() -> Any:
        raise MemoryError

    got = submitted_measure.event(broken, {}, left_s=600.0, estimate_s=1.0)
    assert got == {
        "event": submitted_measure.EVENT,
        "measured": False,
        "reason": "failed",
        "error": "MemoryError",
    }


def test_now_is_the_submitted_head_and_parent_the_parents(tmp_path: Path) -> None:
    """A small turn moves none of the tiny fixture's figures, so this one flips the weight and
    expects exactly what ``compare_to_parent`` 4 says of that head."""
    from tests.unit.wsp_worker.test_judged_measure import _ctx, _view
    from wsp_worker import observation_tools

    view, fx = _view(tmp_path)
    parent = {n: t.values64() for n, t in fx.tensors.items()}
    flipped = _ctx(view, fx, lambda p: {**p, "weight": -p["weight"]})
    expected = observation_tools.CATALOGUE[("compare_to_parent", "4")].fn(flipped)
    assert expected["now"]["auroc"] != expected["parent"]["auroc"]  # or this test proves nothing
    unedited = _ctx(view, fx, lambda p: dict(p))
    got = submitted_measure.event(
        lambda: unedited, {"weight": -parent["weight"]}, left_s=600.0, estimate_s=1.0
    )
    assert got["now"]["auroc"] == expected["now"]["auroc"]
    assert got["parent"]["auroc"] == expected["parent"]["auroc"]
