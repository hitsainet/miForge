"""The context guard (feature 018; FTASKS 4.3, 4.4, 4.5; FR-018.12; TD-4; F18-D12).

The unit half runs here; the long session runs the engine for real under the worker's
interpreter against a scripted endpoint that ENFORCES a 2,000-token window the way miServer does
(``400 context_length_exceeded``) and records every request's size.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tests.support.wsp_engine import line_for, result, run_case, transcript
from tests.support.wsp_fixtures import make_bundle
from tests.support.wsp_scripted_endpoint import ScriptedEndpoint
from wsp_worker.miwsp_engine import context


def test_the_default_window_is_8192_tokens() -> None:
    """4.5: OQ-18-2, answered by the operator 2026-11-18 (F18-D12)."""
    assert context.DEFAULT_WINDOW_TOKENS == 8_192
    assert context.window_tokens({}) == 8_192
    assert context.window_tokens(
        {"context_window_tokens": None, "served_context_tokens": None}
    ) == (8_192)


def test_the_endpoint_window_wins_then_the_served_one() -> None:
    both = {"context_window_tokens": 4_000, "served_context_tokens": 32_768}
    assert context.window_tokens(both) == 4_000
    assert context.window_tokens({"served_context_tokens": 32_768}) == 32_768
    for bad in (0, -5, True, "4000"):
        assert context.window_tokens({"context_window_tokens": bad}) == 8_192


def test_the_output_cap_leaves_the_prompt_most_of_the_window() -> None:
    assert context.output_cap(8_192, 2_048) == 1_024
    assert context.output_cap(65_536, 2_048) == 2_048
    assert context.output_cap(2_000, 2_048) == 256  # the gateway's floor


def test_the_estimate_is_the_gateways_rule_and_the_handoff_is_at_85_percent() -> None:
    """Amended 2026-11-21 (the paging branch): the last count plus the bytes added since at a
    token each, times 1.1, plus the answer; before any count a third of a token a byte (TD-4);
    after a rewrite every byte at the largest observed ratio (at most a token a byte)."""
    guard = context.Guard(window=10_000, cap=1_000)
    first = [{"role": "user", "content": "x" * 3_000}]
    size = context.message_bytes(first)
    assert guard.estimate(first) == math.ceil(math.ceil(size / 3) * 1.1) + 1_000
    guard.reported(prompt_tokens=1_800, messages_sent=1, sent_bytes=size)  # 0.6 a byte
    more = [*first, {"role": "tool", "content": "y" * 7_000}]
    added = context.message_bytes(more[1:])
    assert guard.prompt_estimate(more) == 1_800 + added
    assert guard.estimate(more) == math.ceil((1_800 + added) * 1.1) + 1_000
    assert guard.must_hand_off(more)  # (1800 + ~7040) * 1.1 + 1000 > 8500
    guard.reset()
    assert guard.prompt_estimate(more) == math.ceil(context.message_bytes(more) * 1_800 / size)
    guard.ratios = [5.0]  # not a tokenizer's ratio: kept at a token a byte
    assert guard.prompt_estimate(first) == size


def test_a_tensor_dump_hands_off_where_a_third_of_a_token_a_byte_did_not() -> None:
    """The coordinator's case (2026-11-21): counted at 15,000, then a 12,052-byte read (0.95 tokens
    a byte on Qwen3). A third a byte put it at about 20,400 of a 32,768 window, under the 85%
    hand-off; the gateway's rule puts it at 30,370 and hands off before the request overflows."""
    guard = context.Guard(window=32_768, cap=1_024)
    history = [{"role": "user", "content": "the task"}]
    guard.reported(prompt_tokens=15_000, messages_sent=1, sent_bytes=25_000)
    dumped = [*history, {"role": "tool", "content": "-0.01234 " * 1_339}]
    assert guard.prompt_estimate(dumped) == 15_000 + context.message_bytes(dumped[1:])
    old = math.ceil((15_000 + math.ceil(context.message_bytes(dumped[1:]) / 3)) * 1.1) + 1_024
    assert old < context.HANDOFF_SHARE * 32_768  # the old rule would have sent it
    assert guard.must_hand_off(dumped)


def test_the_guard_and_the_gateway_agree_after_the_first_count() -> None:
    """The guard's rule is the gateway's ``PromptMeter`` once a count exists: the same figure for a
    conversation that grew."""
    from wsp_gateway import meter

    assert context.NEW_TOKENS_PER_BYTE == meter.NEW_TOKENS_PER_BYTE
    assert context.RATIO_MEMORY == meter.RATIO_MEMORY
    assert round((context.FACTOR - 1) * 100) == round(meter.ESTIMATE_MARGIN * 100)
    guard = context.Guard(window=32_768, cap=1_024)
    guard.reported(prompt_tokens=9_000, messages_sent=2, sent_bytes=15_000)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    grown = [*messages, {"role": "tool", "content": "z" * 2_345}]
    assert (
        guard.prompt_estimate(grown)
        == guard.gateway_prompt(grown)
        == 9_000 + (context.message_bytes(grown[2:]))
    )


# ----------------------------------------------------------------------------- 4.4


def _long_session(tmp: Path, declared: int | None) -> tuple[ScriptedEndpoint, Path]:
    fx = make_bundle(tmp / "bundle", rule="mean")
    # 24 DIFFERENT reads (the repeat guard, 2026-11-21, answers a repeat without running it).
    reads = [
        {"calls": [("read_tensor", {"name": name, "precision": p})]}
        for name in ("weight", "bias")
        for p in range(4, 16)
    ]
    steps = [*reads, {"calls": [("submit", {"answer": "read it all"})]}]
    editor = {} if declared is None else {"context_window_tokens": declared}
    with ScriptedEndpoint("openai_compatible", steps, window=2_000) as ep:
        line = line_for(
            tmp / "s",
            fx,
            "inspect",
            ep.gateway,
            python=False,
            max_steps=30,
            editor={"context_window_tokens": 2_000, **editor},
        )
        run_case(tmp / "s", line)
    return ep, tmp / "s"


def test_a_long_session_never_exceeds_a_small_window_and_hands_off(tmp_path: Path) -> None:
    ep, run = _long_session(tmp_path, declared=None)
    assert max(ep.counts) <= 2_000, ep.counts
    out = result(run)
    assert out["stop_reason"] == "submitted" and out["failure"] is None
    assert out["engine"]["handoffs"] >= 1
    assert len(ep.requests) == 25
    assert all(r["body"]["max_tokens"] == 256 for r in ep.requests)
    handoffs = [m for m in transcript(run) if m.get("event") == "handoff"]
    assert len(handoffs) == out["engine"]["handoffs"]


@pytest.mark.parametrize("declared", [1_000_000])
def test_the_negative_control_a_window_declared_too_large_overflows(
    tmp_path: Path, declared: int
) -> None:
    """The same session, with the guard told a window the endpoint does not have: it never hands
    off, the endpoint refuses ``context_length_exceeded``, and the run ends ``error`` with that
    code (through miServer's gateway the code is ``editor_context_exceeded``)."""
    ep, run = _long_session(tmp_path, declared=declared)
    out = result(run)
    assert out["engine"]["handoffs"] == 0
    assert out["stop_reason"] == "error"
    assert out["failure"]["code"] == "context_length_exceeded"
    assert max(ep.counts) > 2_000


def test_the_guard_agrees_with_a_gateway_meter_seeded_from_persisted_evidence() -> None:
    """The uncapped branch's gateway (``31a2e0f``) seeds a restarted session's meter from the
    evidence its served calls persisted. Seeded or live, the meter's estimate for a conversation
    that grew is the last count plus the new bytes at a token each, and after a rewrite the
    body at the largest ratio: the guard's rule on the same figures."""
    import json as _json

    from wsp_gateway import meter

    first = {"model": "m", "messages": [{"role": "user", "content": "x" * 900}]}
    sent = len(_json.dumps(first).encode())
    live = meter.PromptMeter()
    seen = live.observe(first, sent, 540)
    assert seen is not None
    seeded = meter.PromptMeter()
    seeded.seed(
        [
            {
                "sent_bytes": seen.sent_bytes,
                "prompt_tokens": seen.prompt_tokens,
                "messages": seen.messages,
                "conversation_sha256": seen.key,
            }
        ]
    )
    grown = {**first, "messages": [*first["messages"], {"role": "tool", "content": "7" * 2_000}]}
    grown_bytes = len(_json.dumps(grown).encode())
    expected = 540 + math.ceil((grown_bytes - sent) * context.NEW_TOKENS_PER_BYTE)
    assert live.estimate(grown, grown_bytes) == seeded.estimate(grown, grown_bytes) == expected
    # The guard, given the same count and the same added bytes, says the same.
    guard = context.Guard(window=32_768, cap=1_024)
    guard.reported(540, 1, sent)
    added = [{"role": "user", "content": "x"}, {"role": "tool", "content": "7" * 2_000}]
    assert guard.prompt_estimate(added) == 540 + context.message_bytes(added[1:])
    # A rewritten conversation: both take the body at the largest ratio seen.
    rewritten = {"model": "m", "messages": [{"role": "user", "content": "y" * 3_000}]}
    body = len(_json.dumps(rewritten).encode())
    assert seeded.estimate(rewritten, body) == math.ceil(body * 540 / sent)
    guard.reset()
    assert guard.prompt_estimate([{"role": "user", "content": "y" * 3_000}], tools_bytes=0) == (
        math.ceil(context.message_bytes([{"role": "user", "content": "y" * 3_000}]) * 540 / sent)
    )
