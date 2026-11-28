"""H-17: each scripted hosted upstream holds the rules it imitates (feature 020; FR-020.23's AC).

An honest two-request conversation is accepted, with cache reads on the second; one with its first
message edited is refused with FTID §0's L3 text and HTTP 400; and so is an edit of the trailing
turn, which only the append-only rule (``prefix_ok``) can see. Bodies are built by the engine's own
client (``GatewayClient.body``) from neutral messages, as the gateway would forward them.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from tests.support.hosted_upstream import ANTHROPIC, OPENAI, HostedUpstream
from wsp_worker.miwsp_engine import protocol as proto
from wsp_worker.miwsp_engine.client import GatewayClient, anthropic_reply, openai_reply
from wsp_worker.miwsp_engine.tools import ForwardBudget, Toolbox, submit_spec

SPECS = Toolbox([submit_spec(lambda answer, confidence=None: "ok")], ForwardBudget(0)).specs
GATEWAY = {"ip": "127.0.0.1", "port": 9, "token": "t"}
HEADERS = {"anthropic-version": "2026-10-21", "content-type": "application/json"}


def kind_of(personality: str) -> str:
    return "anthropic" if personality == ANTHROPIC else "openai_compatible"


def build(personality: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    kind = kind_of(personality)
    client = GatewayClient(kind, "m", GATEWAY, cache=True)
    fields = proto.request_fields(
        "tool_calls", kind, "generic", SPECS, thinking="medium", locality="external"
    )
    return client.body(messages, fields=fields, max_tokens=1000)


def send(up: HostedUpstream, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    path = "/v1/messages" if up.personality == ANTHROPIC else "/v1/chat/completions"
    with up.client() as http:
        answer = http.post(f"https://upstream{path}", content=json.dumps(body), headers=HEADERS)
    return answer.status_code, answer.json()


def conversation(personality: str) -> tuple[HostedUpstream, list[dict[str, Any]], dict[str, Any]]:
    """The first request and its reply, appended to the neutral history."""
    up = HostedUpstream(personality, scripts=[{"calls": [("submit", {"answer": "x"})]}] * 3)
    history: list[dict[str, Any]] = [
        {"role": "system", "content": "You edit a probe's weights. " * 40},
        {"role": "user", "content": "Repair the probe."},
    ]
    status, data = send(up, build(personality, history))
    assert status == 200, data
    reply = anthropic_reply(data) if personality == ANTHROPIC else openai_reply(data)
    history.append(reply.message)
    call = reply.tool_calls[0]
    history.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": "ok"})
    return up, history, data


@pytest.mark.parametrize("personality", [ANTHROPIC, OPENAI])
def test_an_honest_two_request_conversation_is_accepted_with_cache_reads(
    personality: str,
) -> None:
    up, history, _first = conversation(personality)
    status, data = send(up, build(personality, history))
    assert status == 200, data
    first, second = up.requests
    assert first.conversation == second.conversation
    if personality == ANTHROPIC:
        assert first.usage["cache_read_input_tokens"] == 0
        assert first.usage["cache_creation_input_tokens"] > 0
        # The second reads at least everything the first request held.
        assert second.usage["cache_read_input_tokens"] >= sum(
            first.usage[k]
            for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        )
    else:
        assert second.usage["prompt_tokens_details"]["cached_tokens"] > 0


@pytest.mark.parametrize("personality", [ANTHROPIC, OPENAI])
def test_an_edited_first_message_is_refused_with_a_400(personality: str) -> None:
    up, history, _first = conversation(personality)
    edited = copy.deepcopy(history)
    edited[1]["content"] = "Repair the probe, differently."
    status, data = send(up, build(personality, edited))
    assert status == 400 and data["error"]["type"] == "invalid_request_error"
    if personality == ANTHROPIC:
        assert data["error"]["message"] == (
            "messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound "
            "to a different conversation. Remove the block, or set "
            '`thinking.block_binding.prefix_mismatch_behavior` to "drop_block". That setting '
            "requires the `thinking-binding-controls-2026-08-01` value in the `anthropic-beta` "
            "header."
        )
    else:
        # The first request had no assistant message, so its user turn is the trailing one.
        assert data["error"]["message"].startswith("messages.1.content.0:")


@pytest.mark.parametrize("personality", [ANTHROPIC, OPENAI])
def test_an_edited_trailing_turn_is_refused_by_the_append_only_rule(personality: str) -> None:
    up, history, _first = conversation(personality)
    assert send(up, build(personality, history))[0] == 200
    edited = copy.deepcopy(history)
    edited[-1]["content"] = "[page p1: stubbed]"
    status, data = send(up, build(personality, edited))
    assert status == 400, data
    assert "the request edits the conversation" in data["error"]["message"]


def test_the_sampling_fields_a_forced_choice_an_old_version_and_a_beta_are_refused() -> None:
    up, history, _first = conversation(ANTHROPIC)
    body = build(ANTHROPIC, history)
    for change in ({"temperature": 1.0}, {"tool_choice": {"type": "any"}}):
        assert send(up, {**body, **change})[0] == 400
    with up.client() as http:
        for headers in (
            {**HEADERS, "anthropic-version": "2023-06-01"},
            {**HEADERS, "anthropic-beta": "x"},
        ):
            answer = http.post(
                "https://upstream/v1/messages", content=json.dumps(body), headers=headers
            )
            assert answer.status_code == 400


def test_a_scripted_refusal_is_answered_with_its_category() -> None:
    up = HostedUpstream(ANTHROPIC, scripts=[{"refusal": "cyber"}])
    status, data = send(up, build(ANTHROPIC, [{"role": "user", "content": "x"}]))
    assert status == 200
    assert data["stop_reason"] == "refusal" and data["stop_details"]["category"] == "cyber"
    assert anthropic_reply(data).refusal == {"category": "cyber"}
