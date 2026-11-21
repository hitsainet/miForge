"""An uncapped thinking turn never sends a useless reasoning allowance (fix 2026-11-21, run
``wsp_855a74f820ff``: from about turn 17 every call had ``max_tokens`` 256 and an allowance of 1,
and six actions, the final ``submit`` among them, were cut mid-JSON).

- The engine sizes the cap the gateway will (``context.gateway_cap``, pinned case by case to
  ``wsp_gateway.meter.uncapped_default``) from the gateway's own prompt estimate, and sends the
  reasoning share the gateway would leave (``context.reasoning_share``, pinned to
  ``allowlist.insert_cap``).
- A share under 256 sends the turn with thinking off, recorded ``{event: thinking_skipped}``.
- A room that cannot hold one action is not sent: the conversation is handed off, and a task
  that still leaves no room ends the session ``editor_context_window_too_small``.
- A submit the submit-only turn cannot read is asked for once more, with thinking off.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.support.wsp_engine import line_for
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from wsp_gateway import allowlist, meter
from wsp_worker.miwsp_engine import context, prompts
from wsp_worker.miwsp_engine import protocol as proto
from wsp_worker.miwsp_engine import session as session_mod
from wsp_worker.miwsp_engine.client import ClientError, openai_reply

CTK = "chat_template_kwargs"
FIELD = proto.REASONING_ALLOWANCE_FIELD
WINDOW = 32_768


class Writer:
    def __init__(self) -> None:
        self.lines: list[dict[str, Any]] = []

    def write(self, obj: Mapping[str, Any]) -> None:
        self.lines.append(dict(obj))

    def events(self, name: str) -> list[dict[str, Any]]:
        return [line for line in self.lines if line.get("event") == name]


class CountingClient:
    """Answers each request with the next scripted message, reporting ``prompt_tokens`` from
    ``prompts`` (the last one repeats), and records what was asked."""

    def __init__(self, messages: list[dict[str, Any]], prompts_seen: list[int]) -> None:
        self.messages, self.prompts_seen = messages, prompts_seen
        self.asked: list[dict[str, Any]] = []

    def complete(
        self, messages: Sequence[Mapping[str, Any]], *, fields: Mapping[str, Any], max_tokens: Any
    ) -> proto.Reply:
        n = len(self.asked)
        self.asked.append({"n": len(messages), "fields": dict(fields), "max": max_tokens})
        message = self.messages[min(n, len(self.messages) - 1)]
        tokens = self.prompts_seen[min(n, len(self.prompts_seen) - 1)]
        return openai_reply(
            {
                "choices": [{"message": message}],
                "usage": {"prompt_tokens": tokens, "completion_tokens": 50},
            }
        )

    def body(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {}


def structured(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": json.dumps({"tool": tool, "arguments": args})}


SUBMIT = structured("submit", {"answer": "done"})
LIST = structured("list_tensors", {})


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


def _session(
    tmp: Path, fx: FixtureBundle, client: Any, *, cap: str = "off", **over: Any
) -> tuple[session_mod.Session, Writer]:
    over.setdefault("thinking", "on")
    over.setdefault(
        "editor", {"action_protocol": proto.STRUCTURED, "context_window_tokens": WINDOW}
    )
    line = line_for(tmp / "run", fx, "repair", None, python=False, **over)
    line["spec"]["request"]["output_cap"] = cap
    s = session_mod.Session(line, client=client)
    writer = Writer()
    s.writer = writer  # type: ignore[assignment]
    s._set_thinking()
    return s, writer


# ------------------------------------------------------------------ one sizing, two copies


@pytest.mark.parametrize("prompt", [0, 1_000, 19_177, 27_000, 28_400, 31_000, 32_168, 32_700])
@pytest.mark.parametrize("window", [8_192, 32_768, 131_072])
def test_the_engines_cap_is_the_gateways(window: int, prompt: int) -> None:
    assert context.gateway_cap(window, prompt) == meter.uncapped_default(
        window, prompt_tokens=prompt
    )


@pytest.mark.parametrize("cap", [256, 300, 511, 512, 767, 1_024, 9_874, 32_000])
@pytest.mark.parametrize("allowance", [1, 256, 4_096, 20_000])
def test_the_engines_share_is_what_the_gateway_leaves(cap: int, allowance: int) -> None:
    kept = allowlist.insert_cap(
        "openai_compatible", {FIELD: allowance}, "max_tokens", cap, uncapped=True
    )[FIELD]
    assert context.reasoning_share(allowance, cap) == kept


def test_the_mirrored_figures_equal_the_gateways() -> None:
    assert context.GATEWAY_FIRST_TOKENS_PER_BYTE == meter.FIRST_TOKENS_PER_BYTE
    assert context.GATEWAY_MARGIN_PERCENT == round(meter.ESTIMATE_MARGIN * 100)
    assert context.GATEWAY_TEMPLATE_TOKENS == meter.TEMPLATE_TOKENS
    assert context.GATEWAY_UNCAPPED_FLOOR_TOKENS == meter.UNCAPPED_FLOOR_TOKENS
    assert context.GATEWAY_MAX_OUTPUT_TOKENS == meter.MAX_OUTPUT_TOKENS
    assert context.GATEWAY_REASONING_DIVISOR == allowlist.UNCAPPED_REASONING_DIVISOR
    assert context.MIN_ACTION_TOKENS == allowlist.ACTION_FLOOR_TOKENS


# ------------------------------------------------------------------ the thinking decision


def test_a_roomy_uncapped_turn_sends_the_share_the_gateway_leaves(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    client = CountingClient([LIST, SUBMIT], [2_000])
    s, writer = _session(tmp_path, fx, client)
    s._loop()
    second = client.asked[1]["fields"]
    assert second[CTK] == {"enable_thinking": True}
    # The second turn's prompt is the first's count plus what was added: far from the window.
    assert second[FIELD] == min(s.allowance, 16_000) == 4_096
    assert writer.events("thinking_skipped") == []


def _dumped(s: session_mod.Session, counted: int, prompt: int) -> list[dict[str, Any]]:
    """A conversation whose last count is ``counted`` and which then took a dense tool answer (a
    tensor dump: about a token per byte) bringing the gateway's estimate to ``prompt``.

    Since the paging branch (2026-11-21) the context guard counts new bytes as the gateway does,
    a token each, so it hands off such a conversation itself, long before the room rule. These
    tests give the guard an endless window so they reach the room rule behind it: the second line
    of defence, still needed (the guard hands off at 85%, the room rule at one action's room)."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": "the task"}]
    s.guard.reported(counted, len(messages))
    shell = context.message_bytes([{"role": "user", "content": ""}])
    messages.append({"role": "user", "content": "7" * (prompt - counted - shell)})
    assert s.guard.gateway_prompt(messages) == prompt
    assert s.guard.must_hand_off(messages)  # the guard's own rule now catches the dump
    s.guard.window = 10**9
    assert not s.guard.must_hand_off(messages)
    return messages


def test_a_turn_whose_share_is_under_256_is_sent_with_thinking_off(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """Counted at 15,000, then a 17,500-byte dump: the gateway estimates 32,500 of 32,768 and
    would size the cap at 268, whose reasoning share (12) cannot hold a thought. The turn goes
    with thinking off and no allowance, and the transcript says so."""
    client = CountingClient([SUBMIT], [100])
    s, writer = _session(tmp_path, fx, client)
    messages = _dumped(s, 15_000, 32_500)
    assert context.gateway_cap(WINDOW, 32_500) == 268
    assert context.reasoning_share(s.allowance, 268) == 12
    s._complete(messages, s.toolbox.specs, "intro")
    (asked,) = client.asked
    assert asked["fields"][CTK] == {"enable_thinking": False} and FIELD not in asked["fields"]
    assert asked["n"] == 2 and s.handoffs == 0
    assert writer.events("thinking_skipped") == [
        {"event": "thinking_skipped", "turn": 0, "why": "room", "cap": 268, "share": 12}
    ]


def test_a_turn_with_room_for_a_thought_keeps_thinking_at_the_gateways_share(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    client = CountingClient([SUBMIT], [100])
    s, writer = _session(tmp_path, fx, client)
    messages = _dumped(s, 15_000, 30_000)
    s._complete(messages, s.toolbox.specs, "intro")
    share = context.reasoning_share(s.allowance, context.gateway_cap(WINDOW, 30_000))
    assert client.asked[0]["fields"][CTK] == {"enable_thinking": True}
    assert client.asked[0]["fields"][FIELD] == share >= 256
    assert writer.events("thinking_skipped") == []


def test_a_dump_that_leaves_no_room_for_an_action_is_handed_off_first(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """The gateway's estimate at 32,600 of 32,768: the turn could only be cut off, so the
    conversation is handed off before anything is sent."""
    client = CountingClient([SUBMIT], [100])
    s, _ = _session(tmp_path, fx, client)
    messages = [{"role": "system", "content": "sys"}, *_dumped(s, 15_000, 32_600)]
    s.guard.reported(15_000, 2)
    s._complete(messages, s.toolbox.specs, "intro")
    assert s.handoffs == 1 and client.asked[0]["n"] == 2


def test_a_turn_with_no_room_for_an_action_is_handed_off_first(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """Counted at 32,600 of 32,768, the next turn could only be cut off: the conversation is
    handed off before anything is sent, and the turn then goes with thinking on."""
    client = CountingClient([LIST, SUBMIT], [32_600, 2_000])
    s, writer = _session(tmp_path, fx, client)
    s._loop()
    assert s.handoffs == 1 and len(client.asked) == 2
    assert client.asked[1]["n"] == 2  # the system prompt and the handoff's one message
    assert client.asked[1]["fields"][CTK] == {"enable_thinking": True}
    assert writer.events("thinking_skipped") == []


def test_a_task_that_leaves_no_room_after_a_handoff_ends_by_name(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """Thinking off, cap off, a 6,000-token window: the context guard (a third of a byte per
    token) lets a 10,000-byte task through, but the gateway's estimate (0.75 per byte) leaves
    the answer under 256 tokens, before and after a handoff."""
    client = CountingClient([SUBMIT], [100])
    s, _ = _session(
        tmp_path,
        fx,
        client,
        thinking="off",
        editor={"action_protocol": proto.STRUCTURED, "context_window_tokens": 6_000},
    )
    messages = [{"role": "user", "content": "x" * 10_000}]
    assert not s.guard.must_hand_off(messages)
    with pytest.raises(ClientError) as caught:
        s._complete(messages, s.toolbox.specs, "intro")
    assert caught.value.code == "editor_context_window_too_small"
    assert "for an answer" in str(caught.value) and client.asked == []


def test_a_capped_turn_is_unchanged(tmp_path: Path, fx: FixtureBundle) -> None:
    """With the cap on, the engine names its own cap plus the allowance, as before."""
    client = CountingClient([LIST, SUBMIT], [32_300])
    s, writer = _session(tmp_path, fx, client, cap="on", thinking_tokens=3_000)
    s._loop()
    assert [a["fields"].get(FIELD) for a in client.asked] == [3_000, 3_000]
    assert writer.events("thinking_skipped") == []


# ------------------------------------------------------------------ the submit retry


def _final_turn_session(
    tmp: Path, fx: FixtureBundle, replies: list[dict[str, Any]]
) -> tuple[session_mod.Session, CountingClient, Writer]:
    client = CountingClient(replies, [2_000])
    s, writer = _session(tmp, fx, client, cap="on", thinking_tokens=3_000, max_steps=1)
    s._loop()
    return s, client, writer


CUT = {"role": "assistant", "content": '{"tool": "submit", "arguments": {"answer": "the edit lo'}


def test_a_cut_off_submit_is_asked_for_once_more_with_thinking_off(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    s, client, writer = _final_turn_session(tmp_path, fx, [LIST, CUT, SUBMIT])
    assert len(client.asked) == 3
    retry = client.asked[2]
    assert retry["fields"][CTK] == {"enable_thinking": False} and FIELD not in retry["fields"]
    assert retry["max"] == s.cap  # the action's whole cap, no reasoning share
    assert client.asked[1]["max"] == s.cap + 3_000
    assert s.answer is not None and s.answer[0] == "done"
    assert writer.events("submit_retry") == [{"event": "submit_retry", "kind": "unparseable"}]
    assert [e["why"] for e in writer.events("thinking_skipped")] == ["submit_retry"]
    cut_prefix = prompts.SUBMIT_CUT.split("{")[0]
    told = [m for m in writer.lines if str(m.get("content", "")).startswith(cut_prefix)]
    assert len(told) == 1


def test_the_submit_retry_happens_once(tmp_path: Path, fx: FixtureBundle) -> None:
    s, client, _ = _final_turn_session(tmp_path, fx, [LIST, CUT, CUT, SUBMIT])
    assert len(client.asked) == 3 and s.refused_turns == 2
    assert s.answer is not None and s.answer[0] != "done"


def test_a_readable_final_submit_is_not_retried(tmp_path: Path, fx: FixtureBundle) -> None:
    s, client, writer = _final_turn_session(tmp_path, fx, [LIST, SUBMIT])
    assert len(client.asked) == 2 and writer.events("submit_retry") == []
    assert s.answer is not None and s.answer[0] == "done"


# ------------------------------------------------------- the gateway's own thinking-off


def test_the_engine_records_a_turn_the_gateway_sent_with_thinking_off(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """The gateway's header (its estimate left no useful reasoning share) reaches the reply, and
    the turn is recorded {event: thinking_skipped, why: gateway}."""
    import httpx2

    from wsp_gateway.app import THINKING_SKIPPED_HEADER as GATEWAY_HEADER
    from wsp_worker.miwsp_engine import client as client_mod

    assert client_mod.THINKING_SKIPPED_HEADER == GATEWAY_HEADER
    assert context.MIN_REASONING_TOKENS == meter.MIN_REASONING_TOKENS

    def answer(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "choices": [{"message": SUBMIT, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
            headers={GATEWAY_HEADER: "room"},
        )

    gateway = client_mod.GatewayClient(
        "openai_compatible",
        "m",
        {"ip": "127.0.0.1", "port": 9, "token": "t"},
        transport=httpx2.MockTransport(answer),
    )
    try:
        s, writer = _session(tmp_path, fx, gateway)
        s._loop()
    finally:
        gateway.close()
    assert writer.events("thinking_skipped") == [
        {"event": "thinking_skipped", "turn": 1, "why": "gateway"}
    ]
    assert s.answer is not None and s.answer[0] == "done"
