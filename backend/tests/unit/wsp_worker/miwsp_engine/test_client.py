"""The engine's model client against scripted servers of each kind (feature 018; FTASKS 4.1, 4.2,
4.6; FR-018.10).

A real HTTP round trip (``tests/support/wsp_scripted_endpoint.py``): what is asserted is what the
server RECEIVED, its headers included, and the usage it reported.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.support.wsp_scripted_endpoint import TOKEN, ScriptedEndpoint
from wsp_worker.miwsp_engine import protocol as proto
from wsp_worker.miwsp_engine.client import ClientError, GatewayClient
from wsp_worker.miwsp_engine.tools import ForwardBudget, Toolbox, submit_spec

MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "task"},
    {
        "role": "assistant",
        "content": "looking",
        "tool_calls": [{"id": "c0", "name": "submit", "arguments": {"answer": "a"}}],
    },
    {"role": "tool", "tool_call_id": "c0", "name": "submit", "content": "refused: x"},
    {"role": "user", "content": "Call `submit` now."},
]
SPECS = Toolbox([submit_spec(lambda answer, confidence=None: "ok")], ForwardBudget(0)).specs


def _complete(ep: ScriptedEndpoint, kind: str, serving: str = "generic") -> proto.Reply:
    client = GatewayClient(kind, "editor-model", ep.gateway)
    try:
        fields = proto.request_fields(proto.TOOL_CALLS, kind, serving, SPECS)
        return client.complete(MESSAGES, fields=fields, max_tokens=777)
    finally:
        client.close()


def test_openai_body_headers_and_usage() -> None:
    steps = [{"calls": [("submit", {"answer": "done"})], "usage": (1234, 56)}]
    with ScriptedEndpoint("openai_compatible", steps) as ep:
        reply = _complete(ep, "openai_compatible", "miserver")
    [seen] = ep.requests
    assert seen["path"] == "/v1/chat/completions"
    assert seen["headers"]["authorization"] == f"Bearer {TOKEN}"
    assert "x-api-key" not in {k.lower() for k in seen["headers"]}
    body = seen["body"]
    assert body["model"] == "editor-model" and body["max_tokens"] == 777
    assert body["parallel_tool_calls"] is False and body["tool_choice"] == "auto"
    assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "tool", "user"]
    assert body["messages"][2]["tool_calls"][0]["function"] == {
        "name": "submit",
        "arguments": '{"answer": "a"}',  # a string on the wire, as the gateway requires
    }
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "c0", "content": "refused: x"}
    assert reply.usage == proto.Usage(1234, 56, 0, 0)
    assert [(c.name, c.arguments) for c in reply.tool_calls] == [("submit", '{"answer": "done"}')]


def test_anthropic_body_headers_and_usage() -> None:
    steps = [{"text": "ok", "calls": [("submit", {"answer": "done"})], "usage": (900, 40)}]
    with ScriptedEndpoint("anthropic", steps) as ep:
        reply = _complete(ep, "anthropic")
    [seen] = ep.requests
    assert seen["path"] == "/v1/messages"
    headers = {k.lower(): v for k, v in seen["headers"].items()}
    assert headers["x-api-key"] == TOKEN and headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in headers
    body = seen["body"]
    assert body["system"] == "sys" and body["max_tokens"] == 777
    assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][1]["content"][1] == {
        "type": "tool_use",
        "id": "c0",
        "name": "submit",
        "input": {"answer": "a"},
    }
    # The tool result and the next user text share one user turn.
    assert body["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "c0", "content": "refused: x"},
        {"type": "text", "text": "Call `submit` now."},
    ]
    assert reply.usage == proto.Usage(900, 40, 0, 0) and reply.text == "ok"
    assert [(c.name, c.arguments) for c in reply.tool_calls] == [("submit", {"answer": "done"})]


def test_anthropic_cache_tokens_count_as_input_as_the_gateway_meters_them() -> None:
    from wsp_worker.miwsp_engine.client import anthropic_reply

    reply = anthropic_reply(
        {
            "content": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 7,
            },
        }
    )
    assert reply.usage == proto.Usage(117, 5, 100, 7)


@pytest.mark.parametrize("kind", ["openai_compatible", "anthropic"])
def test_a_gateway_409_ends_with_its_code_and_is_never_retried(kind: str) -> None:
    steps = [{"status": 409, "code": "editor_context_exceeded"}]
    with ScriptedEndpoint(kind, steps) as ep, pytest.raises(ClientError) as ended:
        _complete(ep, kind)
    assert (ended.value.status, ended.value.code) == (409, "editor_context_exceeded")
    assert len(ep.requests) == 1  # no retry


def test_a_generic_endpoints_error_passes_through_with_its_code() -> None:
    steps = [{"status": 400, "code": "invalid_value"}]
    with ScriptedEndpoint("openai_compatible", steps) as ep, pytest.raises(ClientError) as ended:
        _complete(ep, "openai_compatible")
    assert (ended.value.status, ended.value.code) == (400, "invalid_value")
    assert len(ep.requests) == 1


def test_an_unreachable_gateway_is_named() -> None:
    client = GatewayClient("openai_compatible", "m", {"ip": "127.0.0.1", "port": 1, "token": "t"})
    with pytest.raises(ClientError) as ended:
        client.complete(MESSAGES, fields={}, max_tokens=10)
    assert ended.value.code == "wsp_gateway_unreachable"
