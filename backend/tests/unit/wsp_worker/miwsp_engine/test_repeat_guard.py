"""The repeat guard (fix 2026-11-21, live round 5 on main ``df8c9c6``).

Run ``wsp_7767fb91254b`` called ``forward_practice_metrics {"rule": "max"}`` 13 times in a row on
unchanged weights and ran out of steps; ``wsp_554d3932d108`` 10 times. An action identical to
one that already ran (same tool, same arguments) on the same weights is not run and not charged;
the editor is told so, with the turn that ran it; ``repeated_turns`` counts it; after three the
session moves to the submit-only turn. A changed weight, or other arguments, runs it again. A
replay makes the same decisions from the transcript.

In-process on a scripted client (``Session._loop``, as ``test_graceful_end.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support.wsp_engine import line_for
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from wsp_worker.miwsp_engine import paging, prompts
from wsp_worker.miwsp_engine import session as session_mod
from wsp_worker.miwsp_engine.transcript import ScriptedClient, TranscriptWriter, read_script

TOOLS = [{"name": "practice_metrics", "version": "2", "cost": 1}]
FORWARD = ("forward_practice_metrics", {"split": "train"})


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


def _replies(steps: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"c{i}", "name": n, "arguments": a}],
        }
        for i, (n, a) in enumerate(steps)
    ]


def _session(
    tmp: Path, fx: FixtureBundle, steps: list[tuple[str, dict[str, Any]]], **over: Any
) -> tuple[session_mod.Session, ScriptedClient]:
    line = line_for(
        tmp / "run",
        fx,
        "repair",
        None,
        python=False,
        forward_passes=50,
        observation_tools=TOOLS,
        max_steps=over.pop("max_steps", 30),
        **over,
    )
    client = ScriptedClient(_replies(steps))
    s = session_mod.Session(line, client=client)
    s.writer = TranscriptWriter(s.out / "transcript.jsonl", {"x": 1})
    return s, client


def _tool_texts(s: session_mod.Session) -> list[str]:
    import json

    lines = [json.loads(x) for x in (s.out / "transcript.jsonl").read_text().splitlines()]
    return [str(m["content"]) for m in lines if m.get("role") == "tool"]


def test_a_repeat_on_the_same_weights_is_not_run_or_charged(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    s, _ = _session(tmp_path, fx, [FORWARD, FORWARD, ("submit", {"answer": "a"})])
    s._loop()
    s.writer.close()  # type: ignore[union-attr]
    texts = _tool_texts(s)
    assert texts[1] == prompts.REPEATED.format(turn=1)
    assert s.forward.used == 1 and s.repeated_turns == 1
    assert s.result()["engine"]["repeated_turns"] == 1 and s.stop == "submitted"


def test_three_repeats_move_to_the_submit_only_turn(tmp_path: Path, fx: FixtureBundle) -> None:
    """The live loop: the same pass again and again. After the third repeat, the next request
    offers ``submit`` alone, long before ``max_steps``."""
    steps = [FORWARD] * 4 + [("submit", {"answer": "done"})]
    s, client = _session(tmp_path, fx, steps)
    s._loop()
    assert (s.repeated_turns, s.forward.used) == (3, 1)
    assert client.calls == 5 and s.answer == ("done", None) and s.stop == "submitted"
    assert s.steps == 4  # the fifth request is the submit-only turn


def test_a_changed_weight_or_other_arguments_run_it_again(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    w = fx.tensors["weight"].values64().astype(float).tolist()
    w[0] += 1.0
    steps = [
        FORWARD,
        ("forward_practice_metrics", {"split": "validation"}),  # other arguments: runs
        ("write_tensor", {"name": "weight", "values": w}),
        FORWARD,  # the weights changed: runs
        ("submit", {"answer": "a"}),
    ]
    s, _ = _session(tmp_path, fx, steps)
    s._loop()
    assert s.forward.used == 3 and s.repeated_turns == 0


def test_a_repeated_read_is_answered_from_the_guard_too(tmp_path: Path, fx: FixtureBundle) -> None:
    read = ("read_tensor", {"name": "weight"})
    s, _ = _session(tmp_path, fx, [read, read, ("submit", {"answer": "a"})])
    s._loop()
    assert s.repeated_turns == 1


def test_a_replay_makes_the_same_decisions(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [FORWARD] * 4 + [("submit", {"answer": "done"})]
    s, _ = _session(tmp_path, fx, steps)
    s._loop()
    s.writer.close()  # type: ignore[union-attr]
    script = read_script(s.out / "transcript.jsonl")
    again = line_for(
        tmp_path / "b",
        fx,
        "repair",
        None,
        python=False,
        forward_passes=50,
        observation_tools=TOOLS,
        max_steps=30,
    )
    r = session_mod.Session(again, client=script)
    r.writer = None
    r.replaying = True
    r._loop()
    assert (r.repeated_turns, r.forward.used, r.replay_divergence) == (3, 1, None)
    import json

    lines = [json.loads(x) for x in (s.out / "transcript.jsonl").read_text().splitlines()]
    events = [m for m in lines if m.get("event") == "repeated"]
    assert [e["n"] for e in events] == [1, 2, 3] and all(e["first_turn"] == 1 for e in events)


# ------------------------------------------------------- fix 2026-11-22, D128 round 9
# Run wsp_4bd06d60acf0 repeated at turns 5, 47 and 89, with 30 measurements between, and was
# moved to the submit-only turn at 90 of its 100 steps. Its 26 restores all changed the weights
# and all 26 told the editor "no weights changed".


def _bumped(fx: FixtureBundle, by: float) -> tuple[str, dict[str, Any]]:
    w = fx.tensors["weight"].values64().astype(float).tolist()
    w[0] += by
    return ("write_tensor", {"name": "weight", "values": w})


REVERT = ("revert", {})
SUBMIT = ("submit", {"answer": "a"})


def _spread(fx: FixtureBundle) -> list[tuple[str, dict[str, Any]]]:
    """The round-9 shape: three repeats on the original weights, a new measurement between each."""
    steps = [FORWARD, FORWARD]
    for by in (1.0, 2.0):
        steps += [_bumped(fx, by), FORWARD, REVERT, FORWARD]
    return [*steps, _bumped(fx, 3.0), FORWARD, SUBMIT]


def test_repeats_with_new_measurements_between_do_not_end_the_session(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = _spread(fx)
    s, _ = _session(tmp_path, fx, steps)
    s._loop()
    assert (s.repeated_turns, s.forward.used) == (3, 4)
    assert s.steps == len(steps) and s.stop == "submitted"  # every turn was an ordinary one


def test_a_loop_without_a_new_measurement_still_ends(tmp_path: Path, fx: FixtureBundle) -> None:
    """Edits alone are not progress: measure, revert, measure is still the loop."""
    steps = [FORWARD, REVERT, FORWARD, REVERT, FORWARD, REVERT, FORWARD, SUBMIT]
    s, _ = _session(tmp_path, fx, steps)
    s._loop()
    assert (s.repeated_turns, s.forward.used, s.steps) == (3, 1, 7)


def test_the_limit_does_not_claim_it_was_one_action(tmp_path: Path, fx: FixtureBundle) -> None:
    other = ("forward_practice_metrics", {"split": "validation"})
    s, _ = _session(tmp_path, fx, [FORWARD, other, FORWARD, other, FORWARD, SUBMIT])
    s._loop()
    s.writer.close()  # type: ignore[union-attr]
    text = (s.out / "transcript.jsonl").read_text()
    assert prompts.REPEAT_LIMIT_IN_A_ROW.format(n=3) in text
    assert "The same action" not in text


def test_restore_revert_and_write_say_what_they_changed(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [
        ("checkpoint", {"label": "original"}),
        _bumped(fx, 1.0),
        ("restore", {"label": "original"}),
        _bumped(fx, 1.0),
        REVERT,
        _bumped(fx, 1.0),
        _bumped(fx, 0.0),  # a write back to the start
        REVERT,  # nothing to change
        SUBMIT,
    ]
    s, _ = _session(tmp_path, fx, steps)
    s._loop()
    s.writer.close()  # type: ignore[union-attr]
    texts = _tool_texts(s)
    for i in (2, 4, 6):
        assert "this action changed 1 of " in texts[i], texts[i]
        assert texts[i].endswith("against the start: no weights changed")
    assert "this action changed 0 of " in texts[7]


def test_a_repeat_of_a_pageable_result_names_its_page(tmp_path: Path, fx: FixtureBundle) -> None:
    s, _ = _session(tmp_path, fx, [FORWARD, FORWARD, SUBMIT], paging="on")
    s.paging = s.pager.config = paging.Config(True, 32_768, threshold=0, ledgered_threshold=0)
    s._loop()
    s.writer.close()  # type: ignore[union-attr]
    assert prompts.REPEATED_PAGE.format(page="p1") in _tool_texts(s)[1]


def test_the_header_records_the_feedback_version(tmp_path: Path, fx: FixtureBundle) -> None:
    s, _ = _session(tmp_path, fx, [SUBMIT])
    assert s.header()["feedback"] == session_mod.FEEDBACK_VERSION


@pytest.mark.parametrize("recorded_before_fix", [True, False])
def test_a_replay_follows_the_feedback_it_was_recorded_with(
    tmp_path: Path, fx: FixtureBundle, recorded_before_fix: bool
) -> None:
    """THE replay guarantee: rounds 8 and 9 stopped on the cumulative count and were told the
    old texts, so their replays must do both."""
    s, _ = _session(tmp_path, fx, _spread(fx))
    if recorded_before_fix:
        s.feedback = False
    assert s.writer is not None
    s.writer.close()
    original = s.out / "transcript.jsonl"
    s.writer = TranscriptWriter(original, s.header())
    s._loop()
    s.writer.close()
    again = line_for(
        tmp_path / "replay",
        fx,
        "repair",
        None,
        python=False,
        forward_passes=50,
        observation_tools=TOOLS,
        max_steps=30,
        mode="replay",
        replay_of_mode="session",
        extra=[("transcript.jsonl", "transcript", original)],
    )
    r = session_mod.Session(again, client=read_script(original))
    r._loop()
    assert r.feedback is not recorded_before_fix
    assert r.replay_divergence is None
    assert (r.steps, r.repeated_turns, r.forward.used) == (s.steps, 3, s.forward.used)
    assert ("feedback" in r.header()) is not recorded_before_fix
    assert (r.forward.used < 4) is recorded_before_fix  # the old count stopped it early


def test_a_run_from_before_the_fix_gets_the_texts_it_recorded(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """Literal texts, as rounds 8 and 9 recorded them: a test that recorded and replayed with this
    code would agree with itself whatever the old texts became."""
    s, _ = _session(tmp_path, fx, _spread(fx), paging="on")
    s.feedback = False
    s.paging = s.pager.config = paging.Config(True, 32_768, threshold=0, ledgered_threshold=0)
    s._loop()
    s.writer.close()  # type: ignore[union-attr]
    texts = _tool_texts(s)
    assert texts[1] == (
        "You already ran this on these weights; the result is unchanged (see turn 1). Edit the "
        "weights, try a different measurement, or call `submit`. Nothing was run or charged."
    )
    assert "reverted all tensors\nno weights changed" in texts
    assert (
        "The same action was asked for on unchanged weights 3 times. Call `submit` now."
        in (s.out / "transcript.jsonl").read_text()
    )
