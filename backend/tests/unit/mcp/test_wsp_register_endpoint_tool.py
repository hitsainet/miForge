"""``forge_wsp_register_endpoint`` in the LIVE registry (2026-11-20).

Every test goes through ``build_server`` (the registry production serves), with only the network
replaced: the tool is present there, it declares no key, price or locality argument (so the MCP
layer refuses one by name and nothing reaches the backend), and it sends exactly one
``POST /api/v1/wsp/endpoints`` with exactly its payload. Removing the tool's registration, or the
``wsp`` category's module, turns these red; so does any change to the body it sends.
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
from mcp_server.context import GATED_TOOLS
from mcp_server.health_gate import HealthGate
from mcp_server.server import build_server

TOOL = "forge_wsp_register_endpoint"
LOCAL = "http://miserver-backend.miserver.svc:8000/v1"
PENDING = {
    "approval_id": "apr_x",
    "status": "pending",
    "action": "wsp_endpoint_register",
    "summary": {},
    "request_digest": "sha256:0",
    "expires_at": "2026-11-21T00:00:00+00:00",
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


def _call(mcp: Any, arguments: dict[str, Any]) -> Any:
    return asyncio.run(mcp._tool_manager.call_tool(TOOL, arguments, Context(mcp_server=mcp)))


def _tool(mcp: Any) -> Any:
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert len(tools) >= 80, "the registry is not the production one"
    assert TOOL in tools, f"{TOOL} is not in the live registry"
    return tools[TOOL]


def test_the_tool_is_live_and_declares_no_key_price_or_locality() -> None:
    mcp, seen = _server()
    properties = set(_tool(mcp).parameters["properties"])
    assert properties == {
        "endpoint_id",
        "kind",
        "base_url",
        "model",
        "serving",
        "action_protocol",
        "context_window_tokens",
        "thinking",
        "thinking_tokens",
        "output_cap",
        "paging",
        "output_cap_field",
    }
    assert seen == []
    assert GATED_TOOLS[TOOL] == "wsp_endpoint_register"


@pytest.mark.parametrize(
    "extra",
    [
        {"api_key": "sk-SENTINEL-tool-1234567890"},
        {"price_in_per_mtok": "3"},
        {"price_out_per_mtok": "15"},
        {"locality": "external"},
    ],
    ids=["api_key", "price_in", "price_out", "locality"],
)
def test_a_key_price_or_locality_argument_is_refused_and_nothing_is_sent(
    extra: dict[str, Any],
) -> None:
    mcp, seen = _server()
    arguments = {
        "endpoint_id": "ep1",
        "kind": "openai_compatible",
        "base_url": LOCAL,
        "model": "m",
        "serving": "miserver",
        **extra,
    }
    with pytest.raises(ToolError) as caught:
        _call(mcp, arguments)
    (name,) = extra
    assert name in str(caught.value)
    assert "sk-SENTINEL" not in str(caught.value)
    assert seen == []


def test_the_tool_sends_one_post_with_exactly_its_payload() -> None:
    mcp, seen = _server()
    result = _call(
        mcp,
        {
            "endpoint_id": "miserver-gemma4-26b",
            "kind": "openai_compatible",
            "base_url": LOCAL,
            "model": "gemma-4-26b",
            "serving": "miserver",
            "action_protocol": "structured",
            "context_window_tokens": 32768,
        },
    )
    assert [(r.method, r.url.path) for r in seen] == [("POST", "/api/v1/wsp/endpoints")]
    assert json.loads(seen[0].content) == {
        "id": "miserver-gemma4-26b",
        "kind": "openai_compatible",
        "base_url": LOCAL,
        "model": "gemma-4-26b",
        "serving": "miserver",
        "action_protocol": "structured",
        "context_window_tokens": 32768,
    }
    # The MCP server identifies itself as an agent, so the route gates the call.
    assert seen[0].headers["X-Miforge-Agent"].startswith("agent:")
    assert "wsp_endpoint_register" in json.dumps(result, default=str)


def test_an_unset_optional_field_is_not_sent() -> None:
    mcp, seen = _server()
    _call(
        mcp,
        {
            "endpoint_id": "ep1",
            "kind": "anthropic",
            "base_url": "http://10.0.0.5:8000",
            "model": "m",
        },
    )
    assert len(seen) == 1
    assert json.loads(seen[0].content) == {
        "id": "ep1",
        "kind": "anthropic",
        "base_url": "http://10.0.0.5:8000",
        "model": "m",
        "serving": "generic",
    }


def test_a_bad_id_is_refused_by_the_argument_model_and_nothing_is_sent() -> None:
    mcp, seen = _server()
    with pytest.raises(ToolError):
        _call(
            mcp,
            {"endpoint_id": "../runs", "kind": "anthropic", "base_url": LOCAL, "model": "m"},
        )
    assert seen == []
