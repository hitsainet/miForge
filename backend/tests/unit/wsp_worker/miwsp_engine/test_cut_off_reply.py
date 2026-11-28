"""A reply cut off at the limit for one reply is refused as cut off, not as "not JSON" (fix
2026-11-23, D128 round 10, run ``wsp_05c71b5fb432``).

The editor wrote all 4,096 probe weights inline in one ``write_tensor`` call; the reply ran to
13,561 tokens, the gateway recorded ``finish_reason: length``, and the engine answered "the reply
is not JSON", which named a symptom. Now:

- ``client.py`` marks a reply that stopped at the limit (OpenAI ``finish_reason: length``,
  Anthropic ``stop_reason: max_tokens``) in its recorded message, and only then;
- ``protocol.one_action`` refuses such a reply as before (same kind, so every count and the
  submit retry are unchanged) with a text that says it was cut off, and, when the editor can write
  weights from ``python``, how to write many values;
- a replay reads the mark from the transcript, so a run from before the fix replays as it ran.

In-process on a scripted client, as ``test_read_and_nudges.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.wsp_engine import line_for
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from wsp_worker.miwsp_engine import prompts
from wsp_worker.miwsp_engine import protocol as proto
from wsp_worker.miwsp_engine import session as session_mod
from wsp_worker.miwsp_engine.client import (
    anthropic_messages,
    anthropic_reply,
    openai_messages,
    openai_reply,
)
from wsp_worker.miwsp_engine.tools import ToolSpec
from wsp_worker.miwsp_engine.transcript import ScriptedClient, TranscriptWriter, read_script

CUT = prompts.CUT_OFF
CUT_W = prompts.CUT_OFF + prompts.CUT_OFF_WRITE
NOT_JSON = prompts.UNPARSEABLE.format(detail="the reply is not JSON")


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


def _openai(finish: str | None, content: str = "x") -> dict[str, Any]:
    choice: dict[str, Any] = {"message": {"role": "assistant", "content": content}}
    if finish is not None:
        choice["finish_reason"] = finish
    return {"choices": [choice], "usage": {"prompt_tokens": 5, "completion_tokens": 7}}


def _anthropic(stop: str | None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "content": [{"type": "text", "text": "x"}],
        "usage": {"input_tokens": 5, "output_tokens": 7},
    }
    if stop is not None:
        data["stop_reason"] = stop
    return data


# ----------------------------------------------------------------------------- the client's mark


@pytest.mark.parametrize(
    ("finish", "cut"), [("length", True), ("stop", False), ("tool_calls", False), (None, False)]
)
def test_an_openai_reply_is_marked_only_when_it_stopped_at_the_length_limit(
    finish: str | None, cut: bool
) -> None:
    reply = openai_reply(_openai(finish))
    assert reply.cut_off is cut
    # Only when true: every other turn's record is byte-for-byte as before the fix.
    assert (proto.CUT_OFF in reply.message) is cut


@pytest.mark.parametrize(
    ("stop", "cut"), [("max_tokens", True), ("end_turn", False), ("tool_use", False), (None, False)]
)
def test_an_anthropic_reply_is_marked_only_when_it_stopped_at_max_tokens(
    stop: str | None, cut: bool
) -> None:
    reply = anthropic_reply(_anthropic(stop))
    assert reply.cut_off is cut
    assert (proto.CUT_OFF in reply.message) is cut


def test_the_mark_is_recorded_but_never_sent_back_to_the_endpoint() -> None:
    o = openai_reply(_openai("length")).message
    a = anthropic_reply(_anthropic("max_tokens")).message
    assert o[proto.CUT_OFF] is True and a[proto.CUT_OFF] is True
    assert proto.CUT_OFF not in json.dumps(openai_messages([o]))
    assert proto.CUT_OFF not in json.dumps(anthropic_messages([a]))


# ----------------------------------------------------------------------------- the parser


def _spec(name: str) -> ToolSpec:
    return ToolSpec(name, name, {"type": "object", "properties": {}, "required": []}, lambda: "")


def _reply(text: str | None, cut: bool, calls: list[proto.ToolCall] | None = None) -> proto.Reply:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if cut:
        message[proto.CUT_OFF] = True
    return proto.Reply(text, calls or [], None, message)


EDIT_SPECS = {n: _spec(n) for n in ("python", "write_tensor", "submit")}
READ_SPECS = {n: _spec(n) for n in ("python", "submit")}


@pytest.mark.parametrize(
    ("specs", "text"),
    [(EDIT_SPECS, CUT_W), (READ_SPECS, CUT), ({"write_tensor": _spec("write_tensor")}, CUT)],
    ids=["can-write-from-python", "cannot-write", "no-python"],
)
def test_a_cut_off_unreadable_reply_says_it_was_cut_off(
    specs: dict[str, ToolSpec], text: str
) -> None:
    decision = proto.one_action(
        _reply('{"tool": "write_tensor", "arguments": {"values": [0.0, 0', True),
        proto.STRUCTURED,
        specs,
    )
    assert isinstance(decision, proto.Refusal)
    assert (decision.kind, decision.n, decision.cut_off) == ("unparseable", 1, True)
    assert decision.text() == text


def test_a_cut_off_native_call_whose_arguments_are_not_json_says_so_too() -> None:
    call = proto.ToolCall("c1", "write_tensor", '{"name": "weight", "values": [0.0, 0')
    decision = proto.one_action(_reply(None, True, [call]), proto.TOOL_CALLS, EDIT_SPECS)
    assert isinstance(decision, proto.Refusal)
    assert (decision.kind, decision.text()) == ("unparseable", CUT_W)


def test_a_cut_off_reply_with_no_action_says_so_and_is_still_a_no_action() -> None:
    decision = proto.one_action(_reply("", True), proto.STRUCTURED, EDIT_SPECS)
    assert isinstance(decision, proto.Refusal)
    assert (decision.kind, decision.text()) == ("no_action", CUT_W)


def test_a_reply_that_was_not_cut_off_is_refused_as_before() -> None:
    decision = proto.one_action(_reply("{not json", False), proto.STRUCTURED, EDIT_SPECS)
    assert isinstance(decision, proto.Refusal)
    assert (decision.cut_off, decision.text()) == (False, NOT_JSON)
    nothing = proto.one_action(_reply("", False), proto.STRUCTURED, EDIT_SPECS)
    assert isinstance(nothing, proto.Refusal)
    assert nothing.text() == prompts.NUDGE


def test_a_cut_off_reply_that_still_reads_as_one_action_runs() -> None:
    decision = proto.one_action(
        _reply('{"tool": "submit", "arguments": {}}', True), proto.STRUCTURED, EDIT_SPECS
    )
    assert decision == proto.Action("submit", {}, None)


# ----------------------------------------------------------------------------- the session


def _message(content: str | None, cut: bool) -> dict[str, Any]:
    m: dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": []}
    if cut:
        m[proto.CUT_OFF] = True
    return m


TRUNCATED = '{"tool": "write_tensor", "arguments": {"name": "weight", "values": [0.0, 0.0, 0'
SUBMIT = '{"tool": "submit", "arguments": {"answer": "done"}}'


def _line(tmp: Path, fx: FixtureBundle, **over: Any) -> dict[str, Any]:
    return line_for(
        tmp,
        fx,
        "repair",
        None,
        python=False,
        max_steps=30,
        editor={"action_protocol": proto.STRUCTURED},
        **over,
    )


def _run(s: session_mod.Session) -> list[dict[str, Any]]:
    s.writer = TranscriptWriter(s.out / "transcript.jsonl", s.header())
    s._loop()
    s.writer.close()
    return [json.loads(x) for x in (s.out / "transcript.jsonl").read_text().splitlines()]


def _answers(lines: list[dict[str, Any]]) -> list[str]:
    """What the engine said back after each assistant turn (the structured protocol's user
    turns after the first task message)."""
    out, seen = [], False
    for m in lines:
        if m.get("role") == "assistant":
            seen = True
        elif seen and m.get("role") in ("user", "tool"):
            out.append(str(m["content"]))
    return out


@pytest.mark.parametrize("cut", [True, False], ids=["cut-off", "not-cut-off"])
def test_the_session_tells_the_editor_its_reply_was_cut_off(
    tmp_path: Path, fx: FixtureBundle, cut: bool
) -> None:
    s = session_mod.Session(
        _line(tmp_path / "run", fx),
        client=ScriptedClient([_message(TRUNCATED, cut), _message(SUBMIT, False)]),
    )
    lines = _run(s)
    assert (s.stop, s.refused_turns) == ("submitted", 1)
    # python is off here, so the text names no ``W``.
    assert _answers(lines)[0] == (CUT if cut else NOT_JSON)
    refused = [e for e in lines if e.get("event") == "refused"]
    assert refused == [
        {"event": "refused", "kind": "unparseable", "n": 1, **({"cut_off": True} if cut else {})}
    ]


def test_three_cut_off_empty_replies_in_a_row_still_end_the_session(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    empty = _message("", True)
    s = session_mod.Session(
        _line(tmp_path / "run", fx),
        client=ScriptedClient([empty, empty, empty, _message(SUBMIT, False)]),
    )
    lines = _run(s)
    assert (s.stop, s.steps) == ("answered", 3)
    assert _answers(lines)[:2] == [CUT, CUT]


@pytest.mark.parametrize("cut", [True, False], ids=["after-the-fix", "before-the-fix"])
def test_a_replay_says_what_its_original_said(tmp_path: Path, fx: FixtureBundle, cut: bool) -> None:
    """A run recorded after the fix carries the mark and replays its text; one from before has
    none and replays "not JSON". ``replay_divergence`` compares each answer with the record."""
    s = session_mod.Session(
        _line(tmp_path / "run", fx),
        client=ScriptedClient([_message(TRUNCATED, cut), _message(SUBMIT, False)]),
    )
    _run(s)
    original = s.out / "transcript.jsonl"
    r = session_mod.Session(
        _line(
            tmp_path / "replay",
            fx,
            mode="replay",
            replay_of_mode="session",
            extra=[("transcript.jsonl", "transcript", original)],
        ),
        client=read_script(original),
    )
    r._loop()
    assert (r.stop, r.replay_divergence) == ("submitted", None)


def test_a_replay_of_a_record_without_the_mark_does_not_invent_one(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """Negative control for the replay rule: strip the mark from a run recorded after the fix,
    and the replay must diverge at that turn (it now says "not JSON" where the record says "cut
    off"), so the mark, not anything else, is what the replay reads."""
    s = session_mod.Session(
        _line(tmp_path / "run", fx),
        client=ScriptedClient([_message(TRUNCATED, True), _message(SUBMIT, False)]),
    )
    _run(s)
    original = s.out / "transcript.jsonl"
    stripped = tmp_path / "stripped.jsonl"
    rows = [json.loads(x) for x in original.read_text().splitlines()]
    for row in rows:
        row.pop(proto.CUT_OFF, None)
    stripped.write_text("".join(json.dumps(row) + "\n" for row in rows))
    r = session_mod.Session(
        _line(
            tmp_path / "replay",
            fx,
            mode="replay",
            replay_of_mode="session",
            extra=[("transcript.jsonl", "transcript", stripped)],
        ),
        client=read_script(stripped),
    )
    r._loop()
    assert r.replay_divergence == 1
