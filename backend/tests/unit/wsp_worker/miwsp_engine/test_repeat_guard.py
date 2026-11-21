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
from wsp_worker.miwsp_engine import prompts
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
