"""An edit already measured and reverted says so, and counts (2026-11-28, step 5c).

Run ``wsp_0d4a2f916c9c`` (phase-2 E1r, Qwen3.6-27B editor, session style): at turn 80 the editor
ran a ``python`` edit (code ``ecf23989``) that left the weights ``da70b7da``; turn 81 measured them
(``forward_compare_to_parent``: AUROC 0.9902, within noise of the parent) and turn 82 reverted
them. From turn 83 to turn 100 it made the same weights eight more times (seven with the same
code, one with ``3d98ee85``), each followed by a revert: about 20 of its 100 steps. The run ended
at ``max_steps`` with ``no_change``, and the engine never said it was looping (its repeat guard
watches measurements and reads, and each repeat measurement was answered, but the edits ran).

Now an edit that CHANGES the weights to ones a forward pass measured and a ``revert`` or
``restore`` then left ends its answer with ``prompts.EDIT_REPEATED`` (the measurement's turn,
tool and figures, and the revert's turn), counts in ``repeated_turns`` and writes
``{event: edit_repeated}``. It follows the transcript header (``edit_repeats: 1``), so a run
recorded before (E1r among them) replays bit for bit.

The first tests run the REAL child entry point (``engine_cases.py``: the session's own
``_isolate_and_spawn`` and kernel grandchild); the replay tests run in process with a real
kernel attached (``test_reserved_names._run``), so the recorded-before engine can be built.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.wsp_engine import line_for, result, tool_outputs, transcript
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from tests.unit.wsp_worker.miwsp_engine.test_engine_scripted import call, session
from tests.unit.wsp_worker.miwsp_engine.test_reserved_names import _replies, _run, _tool
from wsp_worker.miwsp_engine import paging, prompts
from wsp_worker.miwsp_engine import session as session_mod
from wsp_worker.miwsp_engine.transcript import ScriptedClient, read_script

TOOLS = [{"name": "practice_metrics", "version": "4", "cost": 1}]
EDIT_A = "W['weight'][0] += 1.0"
EDIT_B = "W['weight'][1] += 1.0"
#: The notice's fixed opening, to look for it in an answer.
OPENING = prompts.EDIT_REPEATED.split("{", 1)[0]


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


def _events(lines: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [m for m in lines if m.get("event") == name]


def _figures(measured: str) -> str:
    """The measurement's figures as the notice names them, from its own JSON answer."""
    answer = json.loads(measured.split(paging.FORWARD_SUFFIX, 1)[0])
    cells = {**answer, **answer["at_own_bar"]}
    named = (("auroc", "auroc"), ("recall", "recall"), ("false_alarm_rate", "far"), ("bar", "bar"))
    figures = " ".join(
        f"{column}={paging.fmt(cells[key])}" for key, column in named if cells.get(key) is not None
    )
    assert figures.startswith("auroc=")
    assert " ".join(f"{k}={v}" for k, v in paging.numbers_of(measured)) == figures
    return figures


# ----------------------------------------------------------------------------- the real child


def test_an_edit_measured_and_reverted_says_so_from_the_second_try_on(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """E1r's loop, through the real engine and kernel: edit, measure, revert, edit, revert, edit.
    The first edit is new; each later one names the measurement (turn 2, its tool and figures)
    and the revert (turn 3), and ``repeated_turns`` counts both, beside one event each."""
    steps = [
        call("python", code=EDIT_A),  # 1
        call("forward_practice_metrics", split="validation"),  # 2
        call("revert", name="weight"),  # 3
        call("python", code=EDIT_A),  # 4: tried and reverted
        call("revert", name="weight"),  # 5
        call("python", code=EDIT_A),  # 6: again
        call("submit", answer="kept the last try"),  # 7
    ]
    _ep, run, _ = session(
        tmp_path, fx, "repair", steps, forward_passes=5, observation_tools=TOOLS, max_steps=20
    )
    lines = transcript(run)
    assert lines[0]["edit_repeats"] == session_mod.EDIT_REPEATS_VERSION
    first, measured, _rev, second, _rev2, third, _submitted = tool_outputs(run)
    assert OPENING not in first, first
    events = _events(lines, "edit_repeated")
    assert [(e["turn"], e["n"]) for e in events] == [(4, 1), (6, 2)]
    weights = events[0]["weights"]
    assert len(weights) == 8 and all(e["weights"] == weights for e in events)
    assert all(
        (e["tool"], e["measured_turn"], e["reverted_turn"]) == ("python", 2, 3) for e in events
    )
    notice = "\n" + prompts.EDIT_REPEATED.format(
        weights=weights,
        turn=2,
        tool="forward_practice_metrics",
        figures=_figures(measured),
        reverted=3,
    )
    for answer in (second, third):
        assert answer.endswith(notice), answer
        assert answer.count(OPENING) == 1
    assert "measured at turn 2" in notice and "reverted at turn 3" in notice
    engine = result(run)["engine"]
    assert engine["repeated_turns"] == 2
    assert result(run)["forward_used"] == 1 and result(run)["stop_reason"] == "submitted"


def test_different_weights_and_unmeasured_ones_get_no_notice(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """The hash is the test: after A was measured and reverted, B is new (another entry), B again
    after an unmeasured revert is new too (nothing measured B), and A on top of B is other
    weights again. None says it was tried; nothing is counted."""
    steps = [
        call("python", code=EDIT_A),  # 1: A
        call("forward_practice_metrics", split="validation"),  # 2: A measured
        call("revert", name="weight"),  # 3: A reverted
        call("python", code=EDIT_B),  # 4: B, never measured
        call("revert", name="weight"),  # 5
        call("python", code=EDIT_B),  # 6: B again, unmeasured
        call("python", code=EDIT_A),  # 7: A + B
        call("submit", answer="b"),  # 8
    ]
    _ep, run, _ = session(
        tmp_path, fx, "repair", steps, forward_passes=5, observation_tools=TOOLS, max_steps=20
    )
    assert not any(OPENING in out for out in tool_outputs(run))
    assert _events(transcript(run), "edit_repeated") == []
    assert result(run)["engine"]["repeated_turns"] == 0


def test_a_write_tensor_and_a_restore_are_watched_too(tmp_path: Path, fx: FixtureBundle) -> None:
    """The other edit tool, and a checkpoint's restore as the revert, with Python off."""
    write = call("write_tensor", name="weight", index="0", values=7.0)
    steps = [
        call("checkpoint", label="start"),  # 1
        write,  # 2
        call("forward_practice_metrics", split="validation"),  # 3
        call("restore", label="start"),  # 4
        write,  # 5: tried and restored away
        call("submit", answer="w"),  # 6
    ]
    _ep, run, _ = session(
        tmp_path,
        fx,
        "repair",
        steps,
        forward_passes=5,
        observation_tools=TOOLS,
        max_steps=20,
        python=False,
    )
    outputs = tool_outputs(run)
    assert OPENING not in outputs[1]
    [event] = _events(transcript(run), "edit_repeated")
    assert (event["turn"], event["tool"], event["measured_turn"], event["reverted_turn"]) == (
        5,
        "write_tensor",
        3,
        4,
    )
    assert outputs[4].endswith(
        "\n"
        + prompts.EDIT_REPEATED.format(
            weights=event["weights"],
            turn=3,
            tool="forward_practice_metrics",
            figures=_figures(outputs[2]),
            reverted=4,
        )
    )
    assert result(run)["engine"]["repeated_turns"] == 1


def test_an_inspect_session_has_no_edit_repeats(tmp_path: Path, fx: FixtureBundle) -> None:
    line = line_for(tmp_path / "run", fx, "inspect", None, max_steps=5, python=False)
    client = ScriptedClient(_replies([("submit", {"answer": "a"})]))
    s = session_mod.Session(line, client=client)
    assert not s.edit_repeats and "edit_repeats" not in s.header()


# ----------------------------------------------------------------------------- replays

STEPS = [
    ("python", {"code": EDIT_A}),
    ("forward_practice_metrics", {"split": "validation"}),
    ("revert", {"name": "weight"}),
    ("python", {"code": EDIT_A}),
    ("revert", {"name": "weight"}),
    ("python", {"code": EDIT_A}),
    ("submit", {"answer": "done"}),
]


def _in_process(tmp: Path, fx: FixtureBundle, client: Any, **over: Any) -> session_mod.Session:
    line = line_for(
        tmp,
        fx,
        "repair",
        None,
        max_steps=20,
        python=True,
        forward_passes=5,
        observation_tools=TOOLS,
        **over,
    )
    return session_mod.Session(line, client=client)


@pytest.mark.parametrize("recorded_before", [True, False])
def test_a_replay_follows_the_header_it_was_recorded_with(
    tmp_path: Path, fx: FixtureBundle, recorded_before: bool
) -> None:
    """THE replay guarantee. A run recorded before (no ``edit_repeats`` in its header: E1r's)
    replays with the bare answers and no count, bit for bit; one recorded after replays its
    notices and its count."""
    s = _in_process(tmp_path / "run", fx, ScriptedClient(_replies(STEPS)))
    if recorded_before:
        s.edit_repeats = False  # the engine as it was before 2026-11-28
    lines = _run(s)
    assert ("edit_repeats" in lines[0]) is not recorded_before
    noticed = [OPENING in out for out in _tool(lines)]
    assert noticed == [False, False, False, not recorded_before, False, not recorded_before, False]
    assert s.repeated_turns == (0 if recorded_before else 2)
    original = s.out / "transcript.jsonl"
    again = line_for(
        tmp_path / "replay",
        fx,
        "repair",
        None,
        max_steps=20,
        python=True,
        forward_passes=5,
        observation_tools=TOOLS,
        mode="replay",
        replay_of_mode="session",
        extra=[("transcript.jsonl", "transcript", original)],
    )
    r = session_mod.Session(again, client=read_script(original))
    assert r.edit_repeats is not recorded_before
    replayed = _run(r)
    assert r.replay_divergence is None
    assert _tool(replayed) == _tool(lines)
    assert r.repeated_turns == s.repeated_turns
    assert r.store.sha256() == s.store.sha256()


def test_an_old_header_replays_without_the_notice(tmp_path: Path, fx: FixtureBundle) -> None:
    """E1r's own header shape (``write_noise: 1`` and no ``edit_repeats``): the replay is off."""
    s = _in_process(tmp_path / "run", fx, ScriptedClient(_replies(STEPS)))
    s.edit_repeats = False
    _run(s)
    original = s.out / "transcript.jsonl"
    header = json.loads(original.read_text().splitlines()[0])
    assert header.get("write_noise") == 1 and "edit_repeats" not in header
    again = line_for(
        tmp_path / "replay",
        fx,
        "repair",
        None,
        max_steps=20,
        python=True,
        forward_passes=5,
        observation_tools=TOOLS,
        mode="replay",
        replay_of_mode="session",
        extra=[("transcript.jsonl", "transcript", original)],
    )
    r = session_mod.Session(again, client=read_script(original))
    assert r.edit_repeats is False
    replayed = _run(r)
    assert r.replay_divergence is None and r.repeated_turns == 0
    assert not any(OPENING in out for out in _tool(replayed))
    assert _events(replayed, "edit_repeated") == []


def test_a_paging_session_notices_too(tmp_path: Path, fx: FixtureBundle) -> None:
    """E1r paged (the production default): the notice does not depend on the ledger."""
    s = _in_process(
        tmp_path / "run",
        fx,
        ScriptedClient(_replies(STEPS)),
        paging="on",
        editor={"context_window_tokens": 32_768},
    )
    assert s.paging.keeps_pages
    lines = _run(s)
    assert [e["turn"] for e in _events(lines, "edit_repeated")] == [4, 6]
    assert s.repeated_turns == 2
