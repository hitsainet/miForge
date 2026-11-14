"""FR-012.4: with the backend unreachable, EVERY registered tool returns
``{"unavailable": "backend", "reason"}`` and raises nothing."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from mcp_server.config import MCPSettings
from mcp_server.health_gate import HealthGate
from mcp_server.server import build_server
from tests.support.mcp_harness import call_tool
from tests.support.mcp_ledger import CALLER_ASSERTION_EXEMPT, EXPECTED_CALLS


def test_every_tool_reports_the_backend_unavailable_when_it_is_down() -> None:
    # A real gate against a port nothing listens on: the probe fails, nothing is mocked.
    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    asyncio.run(_point_gate_at_nothing(mcp))
    results = {tool: call_tool(mcp, tool, EXPECTED_CALLS[tool][2]) for tool in EXPECTED_CALLS}
    assert len(results) == 89
    for tool, result in results.items():
        assert isinstance(result, dict) and result.get("unavailable") == "backend", (tool, result)
        assert result["reason"], tool
    for tool in CALLER_ASSERTION_EXEMPT:
        assert "unavailable" not in call_tool(mcp, tool, {})


async def _point_gate_at_nothing(mcp: Any) -> None:
    del mcp


@pytest.fixture(autouse=True)
def _backend_on_a_dead_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIFORGE_API_URL", "http://127.0.0.1:9")


def _probe(handler: Any) -> HealthGate:
    gate = HealthGate("http://backend:8000")
    gate._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return gate


def test_a_degraded_503_is_the_backend_up_and_reporting() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={
                "status": "degraded",
                "dependencies": {
                    "postgres": {"status": "ok", "detail": None},
                    "redis": {"status": "unavailable", "detail": "refused"},
                },
            },
        )

    gate = _probe(handler)
    assert asyncio.run(gate.check("backend")) == (True, "ok")
    assert asyncio.run(gate.check("postgres")) == (True, "ok")
    ok, reason = asyncio.run(gate.check("redis"))
    assert not ok and "refused" in reason


def test_an_html_200_is_not_the_backend() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html/>")

    ok, _ = asyncio.run(_probe(handler).check("backend"))
    assert not ok


def test_a_tool_whose_dependency_is_down_issues_no_request() -> None:
    """The gate is checked BEFORE the call: a down backend costs the agent no request."""
    from tests.support.mcp_harness import FakeGate, build_harness

    mcp, client = build_harness(gate=FakeGate(closed="backend"))
    for tool, (_method, _path, arguments, _recorded, _class) in EXPECTED_CALLS.items():
        result = call_tool(mcp, tool, arguments)
        assert result == {"unavailable": "backend", "reason": "backend down: test"}, tool
    assert client.calls == [], f"{len(client.calls)} requests were sent past a closed gate"
