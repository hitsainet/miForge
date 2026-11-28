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
from tests.support.mcp_harness import PENDING_BODY

TOOL = "forge_wsp_register_endpoint"
LOCAL = "http://miserver-backend.miserver.svc:8000/v1"
#: The gate's real pending body (021: an answer the route can give).
PENDING = {
    **PENDING_BODY,
    "action": "wsp_endpoint_register",
    "approval": {**PENDING_BODY["approval"], "action": "wsp_endpoint_register"},
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
        "history_mode",
        "output_ceiling_tokens",
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


def test_the_ceiling_is_sent_in_the_body_once() -> None:
    """W-11 (feature 019, FR-019.19): through the live registry, ``output_ceiling_tokens`` is sent
    in the one POST's body, exactly."""
    mcp, seen = _server()
    _call(
        mcp,
        {
            "endpoint_id": "ep1",
            "kind": "openai_compatible",
            "base_url": LOCAL,
            "model": "m",
            "serving": "miserver",
            "output_ceiling_tokens": 4096,
        },
    )
    assert [(r.method, r.url.path) for r in seen] == [("POST", "/api/v1/wsp/endpoints")]
    assert json.loads(seen[0].content) == {
        "id": "ep1",
        "kind": "openai_compatible",
        "base_url": LOCAL,
        "model": "m",
        "serving": "miserver",
        "output_ceiling_tokens": 4096,
    }


@pytest.mark.parametrize("value", [255, 10_000_001])
def test_a_ceiling_out_of_bounds_is_refused_and_nothing_is_sent(value: int) -> None:
    mcp, seen = _server()
    with pytest.raises(ToolError) as caught:
        _call(
            mcp,
            {
                "endpoint_id": "ep1",
                "kind": "openai_compatible",
                "base_url": LOCAL,
                "model": "m",
                "output_ceiling_tokens": value,
            },
        )
    assert "output_ceiling_tokens" in str(caught.value)
    assert seen == []


def test_the_window_description_names_the_readers() -> None:
    mcp, _seen = _server()
    text = _tool(mcp).parameters["properties"]["context_window_tokens"]["description"]
    for reader in ("/api/health/inference", "/v1/models", "max_model_len", "/props", "8192"):
        assert reader in text, reader


# ------------------------------------------------------------- feature 020 (FR-020.25, U-4)


@pytest.mark.parametrize("mode", ["append_only", "paging"])
def test_the_history_mode_is_sent_in_the_body_once(mode: str) -> None:
    """U-4: through the live registry, ``history_mode`` reaches the one POST's body, exactly."""
    mcp, seen = _server()
    _call(
        mcp,
        {
            "endpoint_id": "ep1",
            "kind": "openai_compatible",
            "base_url": LOCAL,
            "model": "m",
            "serving": "miserver",
            "history_mode": mode,
        },
    )
    assert [(r.method, r.url.path) for r in seen] == [("POST", "/api/v1/wsp/endpoints")]
    assert json.loads(seen[0].content) == {
        "id": "ep1",
        "kind": "openai_compatible",
        "base_url": LOCAL,
        "model": "m",
        "serving": "miserver",
        "history_mode": mode,
    }


def test_an_unknown_history_mode_is_refused_and_nothing_is_sent() -> None:
    mcp, seen = _server()
    with pytest.raises(ToolError) as caught:
        _call(
            mcp,
            {
                "endpoint_id": "ep1",
                "kind": "openai_compatible",
                "base_url": LOCAL,
                "model": "m",
                "history_mode": "rewrite",
            },
        )
    assert "history_mode" in str(caught.value)
    assert seen == []


def test_the_history_mode_is_a_closed_choice_of_the_backends_two() -> None:
    from wsp.models import HISTORY_MODES

    mcp, _seen = _server()
    schema = _tool(mcp).parameters["properties"]["history_mode"]
    values = {v for branch in schema["anyOf"] for v in branch.get("enum", [])}
    assert values == set(HISTORY_MODES)


def test_the_descriptions_name_the_hosted_paths_rules() -> None:
    """FR-020.25: the register tool says agents propose local endpoints only and that one may be
    append_only; the start tool names the window refusal, the history modes and editor_refused."""
    mcp, _seen = _server()
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    register = tools[TOOL].description or ""
    assert "Agents propose local endpoints only" in register
    assert "append_only" in register
    thinking = tools[TOOL].parameters["properties"]["thinking"]
    assert {v for b in thinking["anyOf"] for v in b.get("enum", [])} == {"off", "on", "default"}
    start = tools["forge_wsp_start_run"].description or ""
    for words in (
        "endpoint_window_required",
        "wsp_endpoint_cache_prices_required",
        "history_mode",
        "paging",
        "append_only",
        "editor_refused",
    ):
        assert words in start, words
