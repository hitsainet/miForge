"""A cut-off reply lowers an uncapped session's output ceiling (2026-11-28, step 5d).

Run ``wsp_3417a18bc85d`` (phase-2 E7, create session, output cap off, Qwen3.6-27B on miServer):
four replies ran to the 32,000-token output ceiling, about 870 s each, 58 of its 95 minutes; the
run ended ``time_budget`` with ``no_change``. The engine refused each as cut off and sent the next
request uncapped again, so the gateway sized it to the ceiling again. Now the first cut-off reply
in a session with the cap off lowers the session's own ceiling to ``CUT_OFF_CAP_TOKENS`` (through
``context.output_cap``'s bounds, never above the pushed ceiling): every later request names that
cap, the refusal says so (``prompts.CUT_OFF_CAPPED``), the transcript gets
``{event: ceiling_lowered}`` and ``result.json``'s engine block ``output_ceiling_lowered_to``,
which the intake stores as the run's ``output_ceiling_tokens`` with ``output_ceiling_lowered``.

In process on a scripted client that records each request's ``max_tokens``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.wsp_engine import line_for
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from wsp import runs
from wsp_worker.miwsp_engine import context, prompts
from wsp_worker.miwsp_engine import protocol as proto
from wsp_worker.miwsp_engine import session as session_mod
from wsp_worker.miwsp_engine.transcript import ScriptedClient, TranscriptWriter, read_script

#: E7's window (miServer reported 126,976) and the default ceiling (32,000).
WINDOW = 126_976
TRUNCATED = '{"tool": "write_tensor", "arguments": {"name": "weight", "values": [0.0, 0.0, 0'
LIST = '{"tool": "list_tensors", "arguments": {}}'
SUBMIT = '{"tool": "submit", "arguments": {"answer": "done"}}'


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


def _message(content: str, cut: bool = False) -> dict[str, Any]:
    m: dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": []}
    if cut:
        m[proto.CUT_OFF] = True
    return m


SCRIPT = [_message(TRUNCATED, True), _message(LIST), _message(TRUNCATED, True), _message(SUBMIT)]


def _line(tmp: Path, fx: FixtureBundle, *, cap: str = "off", **over: Any) -> dict[str, Any]:
    line = line_for(
        tmp,
        fx,
        "repair",
        None,
        python=False,
        max_steps=30,
        editor={"action_protocol": proto.STRUCTURED, "context_window_tokens": WINDOW},
        **over,
    )
    line["spec"]["request"]["output_cap"] = cap
    return line


class Recording(ScriptedClient):
    """A scripted client that keeps each request's ``max_tokens``."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        super().__init__(replies)
        self.max_tokens: list[int | None] = []

    def complete(self, messages: Any, *, fields: Any, max_tokens: int | None) -> Any:
        self.max_tokens.append(max_tokens)
        return super().complete(messages, fields=fields, max_tokens=max_tokens)


def _run(s: session_mod.Session) -> list[dict[str, Any]]:
    s.writer = TranscriptWriter(s.out / "transcript.jsonl", s.header())
    s._loop()
    s.writer.close()
    return [json.loads(x) for x in (s.out / "transcript.jsonl").read_text().splitlines()]


def _answers(lines: list[dict[str, Any]]) -> list[str]:
    out, seen = [], False
    for m in lines:
        if m.get("role") == "assistant":
            seen = True
        elif seen and m.get("role") in ("user", "tool"):
            out.append(str(m["content"]))
    return out


def test_a_cut_off_reply_caps_every_later_request_and_says_so(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    client = Recording(SCRIPT)
    s = session_mod.Session(_line(tmp_path / "run", fx), client=client)
    assert not s.capped and s.ceiling == 32_000
    lines = _run(s)
    cap = session_mod.CUT_OFF_CAP_TOKENS
    assert cap == min(context.output_cap(WINDOW, cap), 32_000) == 4_096
    # The first request is uncapped (the gateway sizes it); every one after the cut-off names it.
    assert client.max_tokens == [None, cap, cap, cap]
    assert s.lowered_cap == cap and s.stop == "submitted" and s.refused_turns == 2
    assert lines[0]["cut_off_cap"] == session_mod.CUT_OFF_CAP_VERSION
    assert [e for e in lines if e.get("event") == "ceiling_lowered"] == [
        {"event": "ceiling_lowered", "turn": 1, "from": 32_000, "to": cap, "why": "cut_off"}
    ]
    answers = _answers(lines)
    assert answers[0] == prompts.CUT_OFF + prompts.CUT_OFF_CAPPED.format(cap=cap)
    assert answers[2] == prompts.CUT_OFF  # said once
    assert s.result()["engine"]["output_ceiling_lowered_to"] == cap


def test_the_lowered_cap_is_never_above_the_pushed_ceiling(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    line = _line(tmp_path / "run", fx)
    line["spec"]["editor"]["output_ceiling_tokens"] = 1_000
    client = Recording(SCRIPT)
    s = session_mod.Session(line, client=client)
    _run(s)
    assert client.max_tokens == [None, 1_000, 1_000, 1_000]


def test_a_capped_session_already_names_its_cap_and_is_unchanged(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    client = Recording(SCRIPT)
    s = session_mod.Session(_line(tmp_path / "run", fx, cap="on"), client=client)
    lines = _run(s)
    assert s.capped and len(set(client.max_tokens)) == 1 and client.max_tokens[0] == s.cap
    assert s.lowered_cap is None
    assert not [e for e in lines if e.get("event") == "ceiling_lowered"]
    assert _answers(lines)[0] == prompts.CUT_OFF
    assert "output_ceiling_lowered_to" not in s.result()["engine"]


def test_a_session_with_no_cut_off_stays_uncapped(tmp_path: Path, fx: FixtureBundle) -> None:
    client = Recording([_message(LIST), _message(SUBMIT)])
    s = session_mod.Session(_line(tmp_path / "run", fx), client=client)
    _run(s)
    assert client.max_tokens == [None, None]
    assert "output_ceiling_lowered_to" not in s.result()["engine"]


@pytest.mark.parametrize("recorded_before", [True, False])
def test_a_replay_follows_the_header_it_was_recorded_with(
    tmp_path: Path, fx: FixtureBundle, recorded_before: bool
) -> None:
    """A run recorded before (no ``cut_off_cap``: E7's) replays its bare cut-off answers and never
    lowers; one recorded after replays the lowering and its text, bit for bit."""
    s = session_mod.Session(_line(tmp_path / "run", fx), client=ScriptedClient(SCRIPT))
    if recorded_before:
        s.cut_off_cap = False  # the engine as it was before 2026-11-28
    lines = _run(s)
    assert ("cut_off_cap" in lines[0]) is not recorded_before
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
    assert r.cut_off_cap is not recorded_before
    replayed = _run(r)
    assert r.replay_divergence is None
    assert _answers(replayed) == _answers(lines)
    assert (r.lowered_cap is None) is recorded_before
    if recorded_before:
        assert _answers(lines)[0] == prompts.CUT_OFF


# ----------------------------------------------------------------------------- the intake

BLOCK = {"action_protocol": "structured", "handoffs": 0, "refused_turns": 1, "kernel_restarts": 0}


def test_the_intake_stores_the_lowered_ceiling_and_leaves_the_push_figures_otherwise() -> None:
    lowered = runs.result_counters({"engine": {**BLOCK, "output_ceiling_lowered_to": 4_096}})
    assert (lowered["output_ceiling_tokens"], lowered["output_ceiling_lowered"]) == (4_096, True)
    assert "output_ceiling_lowered_to" not in lowered
    # Absent: neither column is in the update, so the push's figures stand (never nulled).
    plain = runs.result_counters({"engine": BLOCK})
    assert "output_ceiling_tokens" not in plain and "output_ceiling_lowered" not in plain


@pytest.mark.parametrize("value", [0, -1, "4096", True])
def test_a_bad_lowered_ceiling_is_refused(value: Any) -> None:
    from core.errors import Refusal

    with pytest.raises(Refusal) as caught:
        runs.result_counters({"engine": {**BLOCK, "output_ceiling_lowered_to": value}})
    assert caught.value.code == "wsp_result_invalid"
