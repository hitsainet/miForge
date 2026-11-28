"""The WSP MCP tools carry the editor's thinking (2026-11-21), in the LIVE registry.

``forge_wsp_register_endpoint`` declares ``thinking`` and ``thinking_tokens`` and sends them in
exactly one ``POST /api/v1/wsp/endpoints``; ``forge_wsp_start_run`` passes a start body's
``thinking`` and ``thinking_tokens`` through to ``POST /api/v1/wsp/runs`` unchanged, once. The
registry is ``build_server``'s, with only the network replaced.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server.config import MCPSettings
from mcp_server.health_gate import HealthGate
from mcp_server.server import build_server
from tests.support.mcp_harness import PENDING_BODY

LOCAL = "http://miserver-backend.miserver.svc:8000/v1"
#: The gate's real pending body (021: an answer the route can give).
PENDING = {
    **PENDING_BODY,
    "action": "wsp_session",
    "approval": {**PENDING_BODY["approval"], "action": "wsp_session"},
}


@pytest.fixture(autouse=True)
def _open_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unavailable(self: Any, *_: str) -> None:
        return None

    monkeypatch.setattr(HealthGate, "unavailable", unavailable)


def _server() -> tuple[Any, list[httpx.Request]]:
    mcp, client = build_server(MCPSettings(auth_token="x" * 32))
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(202, json=PENDING)

    client._http = httpx.AsyncClient(
        base_url="http://backend/api/v1", transport=httpx.MockTransport(handler)
    )
    return mcp, seen


def _tools(mcp: Any) -> dict[str, Any]:
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert len(tools) >= 80, "the registry is not the production one"
    return tools


def _call(mcp: Any, name: str, arguments: dict[str, Any]) -> Any:
    assert name in _tools(mcp), f"{name} is not in the live registry"
    return asyncio.run(mcp._tool_manager.call_tool(name, arguments, Context(mcp_server=mcp)))


def test_the_register_tool_declares_thinking_with_its_three_values() -> None:
    mcp, _ = _server()
    schema = _tools(mcp)["forge_wsp_register_endpoint"].parameters["properties"]
    values = {v for option in schema["thinking"]["anyOf"] for v in option.get("enum", [])}
    assert values == {"off", "on", "default"}
    assert "thinking_tokens" in schema


def test_the_register_tool_sends_thinking_once_with_exactly_its_payload() -> None:
    mcp, seen = _server()
    _call(
        mcp,
        "forge_wsp_register_endpoint",
        {
            "endpoint_id": "miserver-qwen36-27b",
            "kind": "openai_compatible",
            "base_url": LOCAL,
            "model": "qwen3.6-27b",
            "serving": "miserver",
            "action_protocol": "structured",
            "thinking": "on",
            "thinking_tokens": 6000,
            "output_cap": "off",
        },
    )
    assert [(r.method, r.url.path) for r in seen] == [("POST", "/api/v1/wsp/endpoints")]
    assert json.loads(seen[0].content) == {
        "id": "miserver-qwen36-27b",
        "kind": "openai_compatible",
        "base_url": LOCAL,
        "model": "qwen3.6-27b",
        "serving": "miserver",
        "action_protocol": "structured",
        "thinking": "on",
        "thinking_tokens": 6000,
        "output_cap": "off",
    }


def test_a_bad_thinking_value_is_refused_and_nothing_is_sent() -> None:
    mcp, seen = _server()
    with pytest.raises(ToolError):
        _call(
            mcp,
            "forge_wsp_register_endpoint",
            {
                "endpoint_id": "ep1",
                "kind": "openai_compatible",
                "base_url": LOCAL,
                "model": "m",
                "thinking": "high",
            },
        )
    assert seen == []


def test_the_start_tool_passes_the_run_override_through_once() -> None:
    mcp, seen = _server()
    body = {
        "bundle": {"owner": "miworkbench", "bundle_id": "wsb_abc123", "bundle_sha256": "a" * 64},
        "kind": "inspect",
        "mode": "session",
        "task": "describe",
        "endpoint_id": "miserver-qwen36-27b",
        "sensitivity": "not_sensitive",
        "primary_sets": ["eval"],
        "provider": {"id": "miwsp_engine"},
        "thinking": "on",
        "thinking_tokens": 4096,
    }
    _call(mcp, "forge_wsp_start_run", {"body": body})
    assert [(r.method, r.url.path) for r in seen] == [("POST", "/api/v1/wsp/runs")]
    assert json.loads(seen[0].content) == body
    description = _tools(mcp)["forge_wsp_start_run"].parameters["properties"]["body"]
    assert "thinking_tokens" in json.dumps(description)
