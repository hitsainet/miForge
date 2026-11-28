"""H-10: a refusal ends the session (feature 020; FR-020.17, FR-020.18; W3-D8, F20-D14; TD-7).

Each session runs for real under the worker's interpreter against the scripted endpoint, which
records what it RECEIVED. A refusal on turn 4 ends the session after exactly four requests, with
no nudge; the transcript holds one ``refusal`` event with the category (or null); the result names
``failure.code: editor_refused``; a replay ends the same way; and the neutral ``refusal`` key is in
no request body. The OpenAI-compatible half reads ``message.refusal``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.wsp_engine import line_for, result, run_main, transcript
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from tests.unit.wsp_worker.miwsp_engine.test_engine_scripted import call, session
from wsp_worker.miwsp_engine import prompts

HOSTED = {"locality": "external", "history_mode": "append_only", "window_tokens": 200_000}


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


def steps_refused_on_turn_4(category: Any) -> list[dict[str, Any]]:
    return [
        call("list_tensors"),
        call("describe_tensor", name="weight"),
        call("python", code="W['weight'][0] += 0.5"),  # an accepted edit before the refusal
        {"refusal": category},
        # Reached only if the refusal were read as an empty reply: a nudge, then a fifth request.
        call("submit", answer="done"),
    ]


def run_refused(tmp: Path, fx: FixtureBundle, kind: str, category: Any) -> tuple[Any, Path]:
    ep, run, _ = session(
        tmp,
        fx,
        "create",
        steps_refused_on_turn_4(category),
        endpoint_kind=kind,
        serving="generic",
        editor=HOSTED,
        max_steps=10,
    )
    return ep, run


@pytest.mark.parametrize("category", ["cyber", None])
def test_an_anthropic_refusal_ends_the_session_after_exactly_four_requests(
    tmp_path: Path, fx: FixtureBundle, category: Any
) -> None:
    ep, run = run_refused(tmp_path, fx, "anthropic", category)
    assert len(ep.requests) == 4  # no nudge, no fifth request
    out = result(run)
    assert out["stop_reason"] == "error"
    assert out["failure"]["code"] == "editor_refused"
    assert (category or "none given") in out["failure"]["message"]
    lines = transcript(run)
    events = [e for e in lines if e.get("event") == "refusal"]
    assert events == [{"event": "refusal", "turn": 4, "category": category}]
    assert not any(m.get("content") == prompts.NUDGE for m in lines if m.get("role") == "user")
    # The refused reply is recorded as it came, with its neutral mark, and never sent back.
    for request in ep.requests:
        assert "refusal" not in json.dumps(request["body"])


def test_an_openai_compatible_refusal_ends_the_session_the_same_way(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    ep, run = run_refused(tmp_path, fx, "openai_compatible", "ignored")
    assert len(ep.requests) == 4
    out = result(run)
    assert out["failure"]["code"] == "editor_refused"
    [event] = [e for e in transcript(run) if e.get("event") == "refusal"]
    assert event == {"event": "refusal", "turn": 4, "category": None}
    [refused_line] = [m for m in transcript(run) if m.get("refusal")]
    assert refused_line["refusal"]["text"] == "I can't help with that."


def test_a_replay_of_a_refused_session_ends_the_same_way(tmp_path: Path, fx: FixtureBundle) -> None:
    _ep, run = run_refused(tmp_path, fx, "anthropic", "cyber")
    again = tmp_path / "replay"
    line = line_for(
        again,
        fx,
        "create",
        None,
        mode="replay",
        replay_of_mode="session",
        max_steps=10,
        extra=[("transcript.jsonl", "transcript", run / "out" / "transcript.jsonl")],
    )
    assert run_main(again, line, mode="replay")["exit"] == 0
    replayed = result(again)
    assert replayed["failure"]["code"] == "editor_refused"
    assert replayed["steps"] == result(run)["steps"] == 4
    assert replayed["replay_divergence"] is None
    assert [e for e in transcript(again) if e.get("event") == "refusal"] == [
        {"event": "refusal", "turn": 4, "category": "cyber"}
    ]


def test_a_refusal_on_the_final_turn_ends_it_by_name(tmp_path: Path, fx: FixtureBundle) -> None:
    """The step limit's final turn reads a refusal as a refusal too, not as a submit that could
    not be read."""
    steps = [call("list_tensors"), {"refusal": "cyber"}, call("submit", answer="x")]
    ep, run, _ = session(
        tmp_path,
        fx,
        "create",
        steps,
        endpoint_kind="anthropic",
        serving="generic",
        editor=HOSTED,
        max_steps=1,
    )
    assert len(ep.requests) == 2
    assert result(run)["failure"]["code"] == "editor_refused"
