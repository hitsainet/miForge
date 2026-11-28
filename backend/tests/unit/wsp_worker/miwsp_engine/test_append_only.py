"""H-7 and H-8: the engine's append-only session (feature 020; FR-020.6 to FR-020.11, FR-020.14;
F20-D2, D3, D15, D19; TD-4, TD-5).

In process, as ``test_paging.py``'s session tests run, with :class:`Wire`: a client that builds
each request with the REAL ``GatewayClient.body`` (Anthropic kind, cache markers on, as the
session's own client is built for a hosted Anthropic endpoint) and records exactly the body
``complete_body`` is handed, which is the body the engine checked (F20-D15). Replies are real
Anthropic answers read by the real ``client.anthropic_reply``.

The byte-prefix test (FR-020.7's AC) checks every recorded body against the one before it with
``miwsp_engine.prefix``, except across a handoff, the one declared boundary.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import itertools
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.support.wsp_engine import line_for
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from wsp_worker.miwsp_engine import client as client_mod
from wsp_worker.miwsp_engine import paging, prefix, prompts
from wsp_worker.miwsp_engine import session as session_mod
from wsp_worker.miwsp_engine import tools as tools_mod
from wsp_worker.miwsp_engine.client import ClientError, GatewayClient
from wsp_worker.miwsp_engine.protocol import Reply
from wsp_worker.miwsp_engine.transcript import TranscriptWriter

TOOLS = [
    {"name": "practice_metrics", "version": "1", "cost": 1},
    {"name": "compare_to_parent", "version": "1", "cost": 2},
]
HOSTED = {
    "locality": "external",
    "history_mode": "append_only",
    "window_tokens": 200_000,
    "serving": "generic",
}
#: 30,000 characters whose every offset is recognisable: ``[000000][000001]...``.
LONG = "".join(f"[{i:06d}]" for i in range(3_750))
GATEWAY = {"ip": "127.0.0.1", "port": 9, "token": "t"}


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


class Wire:
    """Builds the real wire body and answers with scripted Anthropic replies, recording every
    body it was handed to send. ``thinking`` adds a signed thinking block to each reply."""

    def __init__(self, steps: Sequence[tuple[str, dict[str, Any]]], *, thinking: bool = True):
        self.steps = list(steps)
        self.thinking = thinking
        self.bodies: list[dict[str, Any]] = []
        #: Requests that came through ``complete`` (a paging session's path), not ``complete_body``.
        self.completes = 0
        self.inner = GatewayClient("anthropic", "claude-x", GATEWAY, cache=True)

    def body(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        fields: Mapping[str, Any],
        max_tokens: int | None,
    ) -> dict[str, Any]:
        return self.inner.body(messages, fields=fields, max_tokens=max_tokens)

    def complete_body(self, body: Mapping[str, Any]) -> Reply:
        sent = json.loads(json.dumps(body))  # what would cross the wire, not a live reference
        self.bodies.append(sent)
        n = len(self.bodies) - 1
        name, args = self.steps[n] if n < len(self.steps) else ("submit", {"answer": "end"})
        blocks: list[dict[str, Any]] = []
        if self.thinking:
            blocks.append({"type": "thinking", "thinking": f"plan {n}", "signature": f"s{n}"})
        blocks.append({"type": "tool_use", "id": f"t{n}", "name": name, "input": args})
        return client_mod.anthropic_reply(
            {
                "content": blocks,
                "stop_reason": "tool_use",
                "usage": {"input_tokens": len(json.dumps(sent)) // 3, "output_tokens": 20},
            }
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        fields: Mapping[str, Any],
        max_tokens: int | None,
    ) -> Reply:
        self.completes += 1
        return self.complete_body(self.body(messages, fields=fields, max_tokens=max_tokens))


def make(
    tmp: Path, fx: FixtureBundle, steps: Sequence[tuple[str, dict[str, Any]]], **over: Any
) -> tuple[session_mod.Session, Wire]:
    wire = over.pop("wire", None) or Wire(steps)
    line = line_for(
        tmp,
        fx,
        over.pop("kind", "repair"),
        None,
        python=False,
        forward_passes=200,
        observation_tools=TOOLS,
        max_steps=over.pop("max_steps", 40),
        tokens=10_000_000,
        editor_kind="anthropic",
        serving="generic",
        editor={**HOSTED, **over.pop("editor", {})},
        **over,
    )
    s = session_mod.Session(line, client=wire)
    s.writer = TranscriptWriter(s.out / "transcript.jsonl", s.header())
    return s, wire


def run(s: session_mod.Session) -> list[dict[str, Any]]:
    with contextlib.redirect_stdout(io.StringIO()):
        s._loop()
    assert s.writer is not None
    s.writer.close()
    return [json.loads(x) for x in (s.out / "transcript.jsonl").read_text().splitlines()]


def long_tool(s: session_mod.Session) -> None:
    """``list_tensors`` answers :data:`LONG`: a result above the tool bound, as a full read of a
    large tensor or a long forward answer would be."""
    spec = s.toolbox.specs["list_tensors"]
    s.toolbox.specs["list_tensors"] = dataclasses.replace(spec, fn=lambda pattern=None: LONG)


def measurements(n: int) -> list[tuple[str, dict[str, Any]]]:
    splits = ("train", "validation", "all")
    rules = ("mean", "max", "last", "softmax", "rolling_mean_max")
    combos = [("forward_practice_metrics", {"split": a, "rule": r}) for a in splits for r in rules]
    return combos[:n]


def tool_result(body: dict[str, Any], tool_use_id: str) -> str:
    for message in body["messages"]:
        content = message["content"]
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result" and block["tool_use_id"] == tool_use_id:
                    return str(block["content"])
    raise AssertionError(f"no result for {tool_use_id}")


def assert_append_only(bodies: list[dict[str, Any]], boundaries: set[int] = frozenset()) -> None:
    """FR-020.7's byte-prefix rule on every request, except the first after a handoff."""
    for n in range(1, len(bodies)):
        if n in boundaries:
            continue
        assert prefix.check(bodies[n - 1], bodies[n]) is None, (
            n,
            prefix.check(bodies[n - 1], bodies[n]),
        )


# ------------------------------------------------------------------ H-7: the append-only session


def test_an_append_only_session_pages_nothing_out_and_every_request_keeps_the_prefix(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = [("list_tensors", {}), *measurements(10), ("submit", {"answer": "a"})]
    s, wire = make(tmp_path, fx, steps)
    long_tool(s)
    lines = run(s)
    assert s.paging.history == "append_only" and not s.paging.pages_out
    assert [e for e in lines if e.get("event") == "page_out"] == []
    assert s.pager.pages_out == 0
    assert_append_only(wire.bodies)
    # The system text carries the append-only rule, not paging's; the header records the mode.
    system = wire.bodies[0]["system"][0]["text"]
    assert prompts.APPEND_ONLY in system and prompts.PAGING not in system
    assert lines[0]["history_mode"] == "append_only"
    # recall is offered with an offset (TD-4), under the same name.
    [recall] = [t for t in wire.bodies[0]["tools"] if t["name"] == "recall"]
    assert recall["input_schema"]["properties"]["offset"] == {"type": "integer", "minimum": 0}


def test_the_ledger_is_appended_once_per_batch_and_each_row_once(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = [
        *measurements(10),
        ("write_tensor", {"name": "bias", "values": 0.25}),
        ("submit", {"answer": "a"}),
    ]
    s, wire = make(tmp_path, fx, steps)
    lines = run(s)
    appended = [e for e in lines if e.get("event") == "ledger_appended"]
    assert len(appended) >= 2
    # On the paging schedule: at most one batch every PAGE_EVERY_TURNS turns.
    turns = [e["turn"] for e in appended]
    assert all(b - a >= paging.PAGE_EVERY_TURNS for a, b in itertools.pairwise(turns))
    ledger_messages = [
        m
        for m in lines
        if m.get("role") == "user" and paging.LEDGER_APPENDED_HEADING in str(m.get("content"))
    ]
    assert len(ledger_messages) == len(appended)
    shown = [
        int(row.split(" | ")[0])
        for m in ledger_messages
        for row in str(m["content"]).splitlines()
        if row[:1].isdigit()
    ]
    assert len(shown) == len(set(shown)), shown  # each row is sent once
    assert sorted(shown) == shown
    assert sum(e["rows"] for e in appended) == len(shown)
    assert_append_only(wire.bodies)


def test_a_long_result_is_cut_at_first_send_and_recalled_from_offsets(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = [
        ("list_tensors", {}),
        ("recall", {"page": "p1"}),
        ("recall", {"page": "p1", "offset": 12_000}),
        ("recall", {"page": "p1", "offset": 24_000}),
        ("submit", {"answer": "a"}),
    ]
    s, wire = make(tmp_path, fx, steps)
    long_tool(s)
    run(s)
    first = tool_result(wire.bodies[1], "t0")
    assert first.startswith(LONG[:12_000])
    assert prompts.TRUNCATED.format(n=f"{30_000 - 12_000:,}") in first
    assert first.endswith(prompts.RECALL_HINT.format(page="p1", offset=12_000))
    # The store holds the FULL text (Toolbox.last_full), not the 12,000 the editor was sent.
    assert s.pager.pages["p1"].text == LONG
    parts = [tool_result(wire.bodies[n], f"t{n - 1}") for n in (2, 3, 4)]
    assert (
        parts[0] == LONG[:12_000] + '\n[18,000 characters after this; recall("p1", offset=12000)]'
    )
    assert (
        parts[1]
        == LONG[12_000:24_000] + '\n[6,000 characters after this; recall("p1", offset=24000)]'
    )
    assert parts[2] == LONG[24_000:] + "\n[end of page p1]"
    assert s.pager.recalls == 3
    # The first send was never changed afterwards (FR-020.8: cut once, before it is first sent).
    assert all(tool_result(b, "t0") == first for b in wire.bodies[1:])
    assert_append_only(wire.bodies)


def test_a_page_is_cut_at_a_million_characters(tmp_path: Path) -> None:
    pager = paging.Pager(paging.Config(False, 200_000, history=paging.APPEND_ONLY))
    pager.offer(1, "read", {}, "x" * 1_200_000, {}, ledgered=False)
    assert pager.pages["p1"].text == "x" * paging.MAX_PAGE_CHARS + paging.PAGE_CUT_NOTE


def test_the_store_drops_its_oldest_pages_past_sixteen_million_and_says_so() -> None:
    def fill() -> paging.Pager:
        pager = paging.Pager(paging.Config(False, 200_000, history=paging.APPEND_ONLY))
        for turn in range(1, 19):
            pager.offer(turn, "read", {}, str(turn % 10) * 999_000, {}, ledgered=False)
        return pager

    pager = fill()
    assert pager.stored_chars <= paging.MAX_STORE_CHARS
    assert pager.dropped == {"p1", "p2"}  # the oldest, in order
    assert "p18" in pager.pages and "p3" in pager.pages
    assert (
        pager.recall_from("p1") == "refused: page p1 was dropped to bound memory. Nothing was run."
    )
    assert pager.recall_from("p99").startswith("refused: no page 'p99'")
    # A function of page sizes and order only: a replay drops the same pages.
    assert fill().dropped == pager.dropped


def test_a_page_from_before_a_handoff_is_recalled_after_it(
    tmp_path: Path, fx: FixtureBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    steps = [
        ("list_tensors", {}),
        *measurements(3),
        ("recall", {"page": "p1", "offset": 24_000}),
        ("submit", {"answer": "a"}),
    ]
    s, wire = make(tmp_path, fx, steps)
    long_tool(s)
    real = s.guard.must_hand_off
    calls = {"n": 0}

    def hand_off_once(*a: Any, **kw: Any) -> bool:
        calls["n"] += 1
        return calls["n"] == 4 or real(*a, **kw)  # the fourth request starts a new conversation

    monkeypatch.setattr(s.guard, "must_hand_off", hand_off_once)
    lines = run(s)
    [handoff] = [e for e in lines if e.get("event") == "handoff"]
    assert handoff["conversation"] == 2
    restarted = wire.bodies[3]["messages"]
    assert len(restarted) == 1 and restarted[0]["role"] == "user"  # the task and the summary
    assert "thinking" not in json.dumps(restarted)  # no thinking block is carried (F20-D3)
    summary = restarted[0]["content"]
    summary = summary if isinstance(summary, str) else summary[0]["text"]
    assert "Pages you can recall:" in summary and "p1" in summary
    assert paging.LEDGER_HEADING in summary
    assert tool_result(wire.bodies[5], "t4") == LONG[24_000:] + "\n[end of page p1]"
    assert_append_only(wire.bodies, boundaries={3})


def test_the_handoff_summary_is_the_engines_own_and_the_same_every_time(
    tmp_path: Path, fx: FixtureBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The summary is written by the engine from its own state (notes, diff, budget, ledger,
    pages), never by the editor: the same session gives the same bytes. (A replay does not hand
    off at all: the engine's guard runs on live sessions only, so this AC of FR-020.10 is held
    by determinism; ``0xcc/reviews/020_P1_controls.md`` records it.)"""
    steps = [("list_tensors", {}), *measurements(3), ("submit", {"answer": "a"})]

    def summary(where: Path) -> Any:
        s, _wire = make(where, fx, steps)
        long_tool(s)
        n = {"calls": 0}
        real = s.guard.must_hand_off

        def once(*a: Any, **kw: Any) -> bool:
            n["calls"] += 1
            return n["calls"] == 4 or real(*a, **kw)

        monkeypatch.setattr(s.guard, "must_hand_off", once)
        [event] = [e for e in run(s) if e.get("event") == "handoff"]
        return event["messages"]

    first, second = summary(tmp_path / "a"), summary(tmp_path / "b")
    # The budget line carries the tokens used, which the scripted usage makes equal too.
    assert first == second


def test_the_final_turn_keeps_every_tool_and_refuses_another_with_one_retry(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = [
        *measurements(2),
        ("read_tensor", {"name": "weight"}),  # the final turn (the step limit is 2)
        ("submit", {"answer": "done"}),  # its one retry
    ]
    s, wire = make(tmp_path, fx, steps, max_steps=2)
    original = s.toolbox.call
    ran: list[str] = []

    def spy(name: str, args: Mapping[str, Any]) -> str:
        ran.append(name)
        return original(name, args)

    s.toolbox.call = spy  # type: ignore[method-assign]
    lines = run(s)
    assert len(wire.bodies) == 4
    names = [[t["name"] for t in b["tools"]] for b in wire.bodies]
    assert names[2] == names[3] == names[0]  # the full list on the final turn and its retry
    # read_tensor on the final turn ran nothing: only the measurements and the submit ran.
    assert ran == ["forward_practice_metrics", "forward_practice_metrics", "submit"]
    assert tool_result(wire.bodies[3], "t2") == prompts.FINAL_SUBMIT_ONLY
    assert s.answer == ("done", None) and s.stop == "max_steps"
    assert [e for e in lines if e.get("event") == "submit_retry"] == [
        {"event": "submit_retry", "kind": "not_offered"}
    ]
    assert s.forward.used == 2 and wire.completes == 0
    # The thinking configuration is the same on every request, the retry included (FTDD §0 2).
    assert {json.dumps(b.get("thinking")) for b in wire.bodies} == {
        json.dumps({"type": "adaptive", "display": "summarized"})
    }
    assert_append_only(wire.bodies)


def test_one_timing_event_per_request(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [*measurements(3), ("submit", {"answer": "a"})]
    s, wire = make(tmp_path, fx, steps)
    timing = [e for e in run(s) if e.get("event") == "timing"]
    assert len(timing) == len(wire.bodies) == 4
    assert timing[0]["gap_s"] is None  # a conversation's first request has none
    assert all(e["gap_s"] >= 0 and e["duration_s"] >= 0 for e in timing[1:])
    assert [e["turn"] for e in timing] == [1, 2, 3, 4]


class SlowWire(Wire):
    """:class:`Wire` that holds each reply for its scripted delay, so a request's duration is
    known before the session measures it."""

    def __init__(self, steps: Sequence[tuple[str, dict[str, Any]]], delays: Sequence[float]):
        super().__init__(steps)
        self.delays = list(delays)

    def complete_body(self, body: Mapping[str, Any]) -> Reply:
        time.sleep(self.delays[len(self.bodies)])
        return super().complete_body(body)


def test_each_requests_duration_is_its_scripted_delay_within_half_a_second(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """FR-020.14's AC: the recorded durations equal the upstream's delays within 0.5 s. The
    delays differ by more than the tolerance, and the slow ones are not first or last, so a
    duration that is constant, zero, or taken from another request fails."""
    delays = [0.0, 1.2, 0.0, 0.6]
    steps = [*measurements(3), ("submit", {"answer": "a"})]
    s, wire = make(tmp_path, fx, steps, wire=SlowWire(steps, delays))
    timing = [e for e in run(s) if e.get("event") == "timing"]
    assert len(timing) == len(wire.bodies) == len(delays)
    for event, delay in zip(timing, delays, strict=True):
        assert abs(event["duration_s"] - delay) <= 0.5, (event, delay)
    # Start to start: a gap holds at least the previous request's duration.
    for before, after in itertools.pairwise(timing):
        assert after["gap_s"] >= before["duration_s"] - 0.05, (before, after)


def test_a_paging_session_keeps_its_tools_and_final_turn(tmp_path: Path, fx: FixtureBundle) -> None:
    """TD-4 and FR-020.11's last AC: a ``paging`` session's ``recall`` has no offset and its final
    turn still offers ``submit`` alone; its header names no history mode."""
    steps = [*measurements(2), ("submit", {"answer": "a"})]
    s, wire = make(
        tmp_path,
        fx,
        steps,
        max_steps=2,
        paging="on",
        editor={"history_mode": "paging", "locality": "local", "window_tokens": 32_768},
    )
    lines = run(s)
    assert s.paging.pages_out and "history_mode" not in lines[0]
    recall = s.toolbox.specs["recall"]
    assert "offset" not in recall.parameters["properties"]
    assert recall.fn == s.pager.recall
    # A paging session sends through complete(), and its final turn offers submit alone.
    assert wire.completes == len(wire.bodies) == 3
    assert [t["name"] for t in wire.bodies[-1]["tools"]] == ["submit"]


# ---------------------------------------------------------------- H-8: the engine's own check


def test_an_edited_history_ends_the_session_before_it_is_sent(
    tmp_path: Path, fx: FixtureBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    steps = [*measurements(4), ("submit", {"answer": "a"})]
    s, wire = make(tmp_path, fx, steps)
    real = s._append_ledger_if_due

    def edit_then_append(messages: list[dict[str, Any]]) -> None:
        if s.steps == 3:
            messages[3]["content"] = "[page p1: stubbed]"  # what a page-out would have done
        real(messages)

    monkeypatch.setattr(s, "_append_ledger_if_due", edit_then_append)
    with contextlib.redirect_stdout(io.StringIO()) as out, pytest.raises(ClientError) as caught:
        s._loop()
    assert caught.value.code == "editor_history_edited"
    assert "messages." in caught.value.message
    assert len(wire.bodies) == 2  # nothing was sent after the edit
    assert '"error": "wsp_history_edited"' in out.getvalue()
    assert "stubbed" not in out.getvalue()  # the log line names the path, never the content


def test_the_checked_body_is_the_body_sent(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [*measurements(2), ("submit", {"answer": "a"})]
    s, wire = make(tmp_path, fx, steps)
    run(s)
    assert s._last_wire == wire.bodies[-1]


def test_tools_bound_their_own_recall_answer() -> None:
    """The append-only recall bounds its own answer, so the toolbox does not cut its tail."""
    spec = tools_mod.recall_from_spec(lambda page, offset=0: "y" * 12_050)
    box = tools_mod.Toolbox([spec], tools_mod.ForwardBudget(0))
    assert box.call("recall", {"page": "p1"}) == "y" * 12_050
    assert box.last_full == "y" * 12_050


def test_a_replay_runs_append_only_from_the_header_whatever_the_endpoint_says(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """FR-020.6's replay AC (review survivor A6): the replay reads the mode from the original's
    transcript header, never from the endpoint (here a replay line naming ``paging``), so its
    ``recall`` takes the offset again and the replay reproduces the original."""
    from wsp_worker.miwsp_engine.transcript import read_script

    steps = [
        ("list_tensors", {}),
        ("recall", {"page": "p1", "offset": 12_000}),
        ("submit", {"answer": "a"}),
    ]
    s, _wire = make(tmp_path / "s", fx, steps)
    long_tool(s)
    run(s)
    original = s.out / "transcript.jsonl"
    line = line_for(
        tmp_path / "r",
        fx,
        "repair",
        None,
        mode="replay",
        replay_of_mode="session",
        python=False,
        forward_passes=200,
        observation_tools=TOOLS,
        max_steps=40,
        tokens=10_000_000,
        editor={"history_mode": "paging", "locality": "local"},
        extra=[("transcript.jsonl", "transcript", original)],
    )
    r = session_mod.Session(line, client=read_script(original))
    r.writer = TranscriptWriter(r.out / "transcript.jsonl", r.header())
    long_tool(r)
    run(r)
    assert r.paging.history == paging.APPEND_ONLY
    assert "offset" in r.toolbox.specs["recall"].parameters["properties"]
    assert r.replay_divergence is None
    assert r.pager.recalls == 1
