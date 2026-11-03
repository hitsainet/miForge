"""The production streamable-HTTP app, end to end over JSON-RPC (FR-012.1).

The app is taken from ``__main__.main`` itself (uvicorn stubbed), so this is the wiring that
ships: the registry served over HTTP equals the live registry, a backend refusal reaches the
agent verbatim, and the service host (``miforge-mcp:8765``) is not refused by the SDK's
loopback-only host check.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from mcp_server.client import ForgeClient, ForgeError
from mcp_server.health_gate import HealthGate

TOKEN = "t" * 40
SERVICE = "http://miforge-mcp:8765"
PROTOCOL = "2025-06-18"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
    "MCP-Protocol-Version": PROTOCOL,
}


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Any:
    from mcp_server import __main__ as entry

    captured: dict[str, Any] = {}

    def fake_run(app: Any, **_kwargs: Any) -> None:
        captured["app"] = app

    monkeypatch.setenv("MCP_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("MIFORGE_API_URL", "http://127.0.0.1:9")
    monkeypatch.setattr("uvicorn.run", fake_run)
    monkeypatch.setattr(entry.asyncio, "run", lambda coro: coro.close())
    assert entry.main([]) == 0
    return captured["app"]


@contextlib.asynccontextmanager
async def serving(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=SERVICE) as http:
            yield http


async def _rpc(http: httpx.AsyncClient, method: str, params: dict[str, Any]) -> httpx.Response:
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    return await http.post("/mcp", headers=HEADERS, content=json.dumps(body))


async def test_initialize_answers_on_the_service_host(app: Any) -> None:
    async with serving(app) as client:
        response = await _rpc(
            client,
            "initialize",
            {
                "protocolVersion": PROTOCOL,
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "1"},
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert result["serverInfo"]["name"] == "miforge"
        assert "Call `forge_howto` first." in result["instructions"]
        assert "90 tools across 10 categories" in result["instructions"]


async def test_tools_list_over_http_is_the_live_registry(app: Any) -> None:
    from mcp_server.config import MCPSettings
    from mcp_server.server import build_server

    async with serving(app) as client:
        response = await _rpc(client, "tools/list", {})
        assert response.status_code == 200, response.text
        served = {t["name"] for t in response.json()["result"]["tools"]}
        mcp, _ = build_server(MCPSettings(auth_token=TOKEN))
        assert served == {t.name for t in await mcp.list_tools()}
        assert len(served) == 90


async def test_a_backend_refusal_reaches_the_agent_verbatim(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def nothing_down(self: HealthGate, *deps: str) -> None:
        return None

    monkeypatch.setattr(HealthGate, "unavailable", nothing_down)
    calls: list[tuple[str, str]] = []

    async def refuse(self: ForgeClient, method: str, path: str, **_kw: Any) -> Any:
        calls.append((method, path))
        raise ForgeError(409, "gpu_lease_held", "another run holds the card", {"run_id": "r9"})

    monkeypatch.setattr(ForgeClient, "request", refuse)
    async with serving(app) as client:
        response = await _rpc(
            client, "tools/call", {"name": "forge_get_run", "arguments": {"run_id": "run-1"}}
        )
        result = response.json()["result"]
        assert result["isError"] is True
        assert "gpu_lease_held (409): another run holds the card" in result["content"][0]["text"]
        assert calls == [("GET", "/runs/run-1")]


async def test_the_token_is_required_on_the_served_app(app: Any) -> None:
    async with serving(app) as client:
        headers = {k: v for k, v in HEADERS.items() if k != "Authorization"}
        assert (await client.post("/mcp", headers=headers, content=b"{}")).status_code == 401


async def test_the_served_app_audits_every_tool_call(
    app: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """FR-012.23 through the wiring that ships: ``__main__`` wraps the registry it serves."""
    import logging

    async def nothing_down(self: HealthGate, *deps: str) -> None:
        return None

    async def answer(self: ForgeClient, method: str, path: str, **_kw: Any) -> Any:
        return {"approval_id": "apr_1", "status": "pending", "action": "gpu_start"}

    monkeypatch.setattr(HealthGate, "unavailable", nothing_down)
    monkeypatch.setattr(ForgeClient, "request", answer)
    with caplog.at_level(logging.INFO, logger="mcp_server.server"):
        async with serving(app) as client:
            response = await _rpc(
                client,
                "tools/call",
                {"name": "forge_define_run", "arguments": {"definition": {"run_id": "r"}}},
            )
            assert response.json()["result"]["isError"] is False
    lines = [r.getMessage() for r in caplog.records if "mcp_tool_call" in r.getMessage()]
    assert len(lines) == 1, lines
    assert "tool=forge_define_run" in lines[0] and "identity=agent:forge-mcp" in lines[0]
    assert "status=pending" in lines[0]
