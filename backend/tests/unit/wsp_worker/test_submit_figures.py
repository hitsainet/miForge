"""018 FU-2 (operator, 2026-11-25): ``submit``'s optional claimed ``figures``, compared with the
measurement of the submitted weights (plan A2's ``submitted_measured`` line), so the run page and
the run document can say where the editor's figures differ. Prose is never parsed.

Each session runs for real through the child preamble against a scripted endpoint that records
what it RECEIVED (``test_engine_scripted.session``):

- ``submit`` offers ``figures`` exactly where the worker measures what was submitted (a repair or
  create with the version-4 tools), never on an inspect or a run with older tools;
- the transcript header says so (``figures``), and a replay offers them as its original did, so a
  run recorded before FU-2 replays with its own tool list and its own line;
- false figures (run ``wsp_4bd06d60acf0``'s: AUROC 0.9904 where 0.990304 was measured) differ;
  the measured figures, stated back, match; another split is not compared;
- the rule itself: a figure matches when the measurement rounds to it at the claim's own
  precision; ``false_alarms`` exactly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.wsp_engine import line_for, result, run_main, transcript
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from tests.unit.wsp_worker.miwsp_engine.test_engine_scripted import call, session
from wsp_worker import submitted_measure
from wsp_worker.miwsp_engine import prompts
from wsp_worker.miwsp_engine import tools as tools_mod

V4 = [
    {"name": "practice_metrics", "version": "4", "cost": 1},
    {"name": "compare_to_parent", "version": "4", "cost": 2},
]
V3 = [{**t, "version": "3"} for t in V4]

#: The figures schema, pinned (a change to what the editor reads is a switch-comparison change;
#: FU-2's comparison re-run is pending, in FU-3's GPU window).
FIGURES_SCHEMA_SHA256 = "3c901e0a1c6365ad"


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


def _events(run: Path) -> list[dict[str, Any]]:
    return [e for e in transcript(run) if e.get("event") == submitted_measure.EVENT]


def _turned(fx: FixtureBundle) -> dict[str, Any]:
    w = fx.tensors["weight"].values64().astype(float).tolist()
    w[0] += 1.0
    return {"calls": [("write_tensor", {"name": "weight", "values": w})]}


def _submit_parameters(ep: Any, index: int = 0) -> dict[str, Any]:
    """``submit``'s parameters in request ``index``, as the endpoint received them."""
    for tool in ep.requests[index]["body"]["tools"]:
        if tool["function"]["name"] == "submit":
            parameters: dict[str, Any] = tool["function"]["parameters"]
            return parameters
    raise AssertionError("no submit tool was offered")


# ----------------------------------------------------------------------------- offered


def test_submit_offers_figures_where_the_submitted_weights_are_measured(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    ep, run, _ = session(tmp_path, fx, "repair", [call("submit", answer="a")], observation_tools=V4)
    parameters = _submit_parameters(ep)
    assert parameters["properties"]["figures"] == tools_mod.FIGURES
    assert parameters["required"] == ["answer"]  # optional: a submit without figures is valid
    assert parameters["properties"]["figures"]["description"] == prompts.SUBMIT_FIGURES
    assert transcript(run)[0]["figures"] == submitted_measure.FIGURES_VERSION


@pytest.mark.parametrize(("kind", "tools"), [("repair", V3), ("inspect", V4)])
def test_no_figures_where_nothing_is_measured_after_submit(
    tmp_path: Path, fx: FixtureBundle, kind: str, tools: list[dict[str, Any]]
) -> None:
    ep, run, _ = session(tmp_path, fx, kind, [call("submit", answer="a")], observation_tools=tools)
    assert "figures" not in _submit_parameters(ep)["properties"]
    assert "figures" not in transcript(run)[0]
    assert _events(run) == []


def test_the_figures_schema_is_pinned() -> None:
    text = json.dumps(tools_mod.FIGURES, sort_keys=True)
    assert hashlib.sha256(text.encode()).hexdigest()[:16] == FIGURES_SCHEMA_SHA256, (
        "submit's figures changed: re-run the switch comparison (018_switch.md) or record why "
        "not, then update the pin"
    )


def test_a_submit_without_figures_offers_the_old_spec_exactly() -> None:
    def fn(**_: Any) -> str:
        return ""

    old = tools_mod.submit_spec(fn)
    assert set(old.parameters["properties"]) == {"answer", "confidence"}
    assert tools_mod.submit_spec(fn, figures=True).parameters["properties"]["figures"] == (
        tools_mod.FIGURES
    )


# ----------------------------------------------------------------------------- compared


def test_false_figures_are_flagged(tmp_path: Path, fx: FixtureBundle) -> None:
    """``wsp_4bd06d60acf0``'s shape: figures that were never measured on the submitted weights."""
    claimed = {"split": "validation", "auroc": 0.9999, "recall": 0.999, "false_alarms": 0}
    steps = [_turned(fx), call("submit", answer="better", figures=claimed)]
    _, run, _ = session(tmp_path, fx, "repair", steps, observation_tools=V4)
    (event,) = _events(run)
    claims = event["claims"]
    assert claims["stated"] is True and claims["compared"] is True
    assert claims["differs"] is True
    assert [c["figure"] for c in claims["checks"]] == ["auroc", "recall", "false_alarms"]
    for check in claims["checks"]:
        assert check["claimed"] == claimed[check["figure"]]
        assert check["measured"] == event["now"][check["figure"]]
    assert [c["matches"] for c in claims["checks"]][:2] == [False, False]


def test_the_measured_figures_stated_back_match(tmp_path: Path, fx: FixtureBundle) -> None:
    _, first, _ = session(
        tmp_path / "a",
        fx,
        "repair",
        [_turned(fx), call("submit", answer="x")],
        observation_tools=V4,
    )
    now = _events(first)[0]["now"]
    claimed = {"split": "validation", **{k: now[k] for k in ("auroc", "recall", "false_alarms")}}
    _, run, _ = session(
        tmp_path / "b",
        fx,
        "repair",
        [_turned(fx), call("submit", answer="x", figures=claimed)],
        observation_tools=V4,
    )
    claims = _events(run)[0]["claims"]
    assert claims["compared"] is True and claims["differs"] is False
    assert all(c["matches"] is True for c in claims["checks"])


def test_figures_on_another_split_are_not_compared(tmp_path: Path, fx: FixtureBundle) -> None:
    claimed = {"split": "train", "auroc": 0.5}
    steps = [_turned(fx), call("submit", answer="x", figures=claimed)]
    _, run, _ = session(tmp_path, fx, "repair", steps, observation_tools=V4)
    claims = _events(run)[0]["claims"]
    assert claims == {
        "stated": True,
        "stated_split": "train",
        "compared": False,
        "differs": None,
        "checks": [{"figure": "auroc", "claimed": 0.5, "measured": None, "matches": None}],
    }


def test_no_figures_stated_reads_so(tmp_path: Path, fx: FixtureBundle) -> None:
    _, run, _ = session(
        tmp_path, fx, "repair", [_turned(fx), call("submit", answer="x")], observation_tools=V4
    )
    assert _events(run)[0]["claims"] == {
        "stated": False,
        "stated_split": None,
        "compared": False,
        "differs": None,
        "checks": [],
    }


def test_a_recall_given_as_a_percentage_is_refused_to_the_editor(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = [
        _turned(fx),
        call("submit", answer="x", figures={"split": "validation", "recall": 18.35}),
        call("submit", answer="x", figures={"split": "validation", "recall": 0.1835}),
    ]
    _, run, _ = session(tmp_path, fx, "repair", steps, observation_tools=V4)
    outputs = [m["content"] for m in transcript(run) if m.get("role") == "tool"]
    assert any("invalid arguments for submit" in str(o) and "above 1" in str(o) for o in outputs)
    assert _events(run)[0]["claims"]["checks"][0]["claimed"] == 0.1835


# ----------------------------------------------------------------------------- the rule


@pytest.mark.parametrize(
    ("value", "places"),
    [(0.9904, 4), (1.0, 0), (1, 0), (1e-05, 5), (0.229, 3), (0.25, 2), (0.990304, 6)],
)
def test_places_read_the_claims_own_precision(value: float, places: int) -> None:
    assert submitted_measure.places(value) == places


@pytest.mark.parametrize(
    ("figure", "claimed", "measured", "expected"),
    [
        ("auroc", 0.9904, 0.990304, False),  # wsp_4bd06d60acf0
        ("auroc", 0.9903, 0.990304, True),
        ("auroc", 0.990, 0.990304, True),
        ("auroc", 0.99031, 0.990304, False),  # 0.990304 rounds to 0.99030
        ("auroc", 0.99030, 0.990304, True),
        ("recall", 0.229, 0.2109, False),  # wsp_4bd06d60acf0
        ("recall", 0.1835, 0.18349, True),
        ("recall", 0.1835, 0.18344, False),
        ("false_alarms", 3, 3, True),
        ("false_alarms", 3, 4, False),
    ],
)
def test_a_figure_matches_at_its_own_precision(
    figure: str, claimed: float, measured: float, expected: bool
) -> None:
    assert submitted_measure.matches(figure, claimed, measured) is expected


def test_a_line_without_claims_offered_is_the_line_before_fu2() -> None:
    def broken() -> Any:
        raise MemoryError

    line = submitted_measure.event(broken, {}, left_s=600.0, estimate_s=1.0)
    assert "claims" not in line
    offered = submitted_measure.event(broken, {}, left_s=600.0, estimate_s=1.0, claims=True)
    assert offered["claims"]["compared"] is False and offered["claims"]["stated"] is False


# ----------------------------------------------------------------------------- replay


def _replay(tmp: Path, fx: FixtureBundle, transcript_path: Path) -> Path:
    run = tmp / "replay"
    line = line_for(
        run,
        fx,
        "repair",
        None,
        mode="replay",
        replay_of_mode="session",
        observation_tools=V4,
        extra=[("transcript.jsonl", "transcript", transcript_path)],
    )
    seen = run_main(run, line, mode="replay")
    assert seen["exit"] == 0, seen["stderr"][-3000:]
    return run


def test_a_replay_states_and_compares_the_same_figures(tmp_path: Path, fx: FixtureBundle) -> None:
    claimed = {"split": "validation", "auroc": 0.9999}
    steps = [_turned(fx), call("submit", answer="x", figures=claimed)]
    _, first, _ = session(tmp_path, fx, "repair", steps, observation_tools=V4)
    run = _replay(tmp_path, fx, first / "out" / "transcript.jsonl")
    assert result(run)["replay_divergence"] is None
    assert transcript(run)[0]["figures"] == submitted_measure.FIGURES_VERSION
    assert _events(run)[0]["claims"] == _events(first)[0]["claims"]


def test_a_run_recorded_before_figures_replays_as_it_ran(tmp_path: Path, fx: FixtureBundle) -> None:
    """A transcript from before FU-2: no ``figures`` in its header and a line without ``claims``.
    Its replay offers ``submit`` as it was, reproduces, and writes no ``claims``."""
    steps = [_turned(fx), call("submit", answer="x")]
    _, first, _ = session(tmp_path, fx, "repair", steps, observation_tools=V4)
    lines = [json.loads(x) for x in (first / "out" / "transcript.jsonl").read_text().splitlines()]
    lines[0] = {k: v for k, v in lines[0].items() if k != "figures"}
    lines = [
        (
            {k: v for k, v in o.items() if k != "claims"}
            if o.get("event") == "submitted_measured"
            else o
        )
        for o in lines
    ]
    old = tmp_path / "old.jsonl"
    old.write_text("".join(json.dumps(o) + "\n" for o in lines))
    run = _replay(tmp_path, fx, old)
    assert result(run)["replay_divergence"] is None
    assert "figures" not in transcript(run)[0]
    (event,) = _events(run)
    assert "claims" not in event
