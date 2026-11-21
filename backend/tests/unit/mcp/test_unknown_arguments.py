"""Every tool REFUSES an argument name it does not declare (2026-11-11).

The SDK's argument models ignore extra keys: a misplaced or misspelt name was dropped and the call
ran on its defaults. miStudio's server changed GPU work that way on 2026-11-10; this server did the
same (``forge_list_runs`` with ``bogus_name`` sent ``GET /runs?limit=25&offset=0``, every run).
``server.refuse_unknown_arguments`` closes it for the whole registry in ``build_server``.

Every test goes through ``build_server`` (the registry production serves) with only the network
replaced, so removing the one wiring line there turns them red. The wire test drives the app
``__main__`` builds.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server.client import ForgeClient
from mcp_server.config import MCPSettings
from mcp_server.health_gate import HealthGate
from mcp_server.server import build_server
from tests.support.mcp_ledger import EXPECTED_CALLS
from tests.unit.mcp.test_http_transport import _rpc, serving
from tests.unit.mcp.test_http_transport import app as app  # the fixture

UNKNOWN = "not_an_argument_of_this_tool"


@pytest.fixture(autouse=True)
def _open_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unavailable(self: Any, *_: str) -> None:
        return None

    monkeypatch.setattr(HealthGate, "unavailable", unavailable)


def _server() -> tuple[Any, list[httpx.Request]]:
    """The production registry; every backend request is recorded and answered ``{}``."""
    mcp, client = build_server(MCPSettings(auth_token="x" * 32))
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    client._http = httpx.AsyncClient(  # the transport is the only stand-in
        base_url="http://backend/api/v1", transport=httpx.MockTransport(handler)
    )
    return mcp, seen


def _call(mcp: Any, name: str, arguments: dict[str, Any]) -> Any:
    return asyncio.run(mcp._tool_manager.call_tool(name, arguments, Context(mcp_server=mcp)))


def _tools(mcp: Any) -> list[Any]:
    return list(mcp._tool_manager.list_tools())


def test_every_registered_tool_refuses_an_unknown_name_and_runs_nothing() -> None:
    mcp, seen = _server()
    tools = _tools(mcp)
    assert len(tools) >= 80, "the registry is not the production one"
    ran: list[str] = []
    unnamed: list[str] = []
    for tool in tools:
        try:
            _call(mcp, tool.name, {UNKNOWN: 1})
        except ToolError as exc:
            text = str(exc)
            accepted = sorted(tool.parameters.get("properties", {}))
            if UNKNOWN not in text or any(name not in text for name in accepted):
                unnamed.append(f"{tool.name}: {text[:300]}")
        else:
            ran.append(tool.name)
    assert not ran, f"tools that ACCEPTED an unknown argument name: {ran}"
    assert not unnamed, "refusals that do not name the unknown and accepted names:\n" + "\n".join(
        unnamed
    )
    assert seen == [], f"a refused call reached the backend: {[str(r.url) for r in seen]}"


def test_the_refusal_names_the_unknown_and_the_accepted_names() -> None:
    mcp, seen = _server()
    with pytest.raises(ToolError) as caught:
        _call(mcp, "forge_list_runs", {"state": "running", "runType": "create"})
    text = str(caught.value)
    assert "unknown argument(s) runType, state" in text
    assert "forge_list_runs accepts include_archived, limit, offset, run_type, status" in text
    assert seen == []


def test_a_declared_name_still_runs_with_its_value() -> None:
    mcp, seen = _server()
    _call(mcp, "forge_list_runs", {"status": "running"})
    assert [(r.method, r.url.path, dict(r.url.params)) for r in seen] == [
        (
            "GET",
            "/api/v1/runs",
            {"limit": "25", "offset": "0", "status": "running", "include_archived": "false"},
        )
    ]


def test_every_documented_call_is_accepted_by_the_production_argument_models() -> None:
    """The ledger's arguments pass the strict models: no documented call carries a stray name
    (the caller harness registers modules directly, without this check)."""
    mcp, _ = _server()
    registered = {t.name: t for t in _tools(mcp)}
    refused: list[str] = []
    for tool, entry in sorted(EXPECTED_CALLS.items()):
        kwargs = entry[2]
        meta = registered[tool].fn_metadata
        try:
            meta.arg_model.model_validate(meta.pre_parse_json(kwargs))
        except ValueError as exc:
            refused.append(f"{tool}: {str(exc)[:300]}")
    assert not refused, "documented calls the production models refuse:\n" + "\n".join(refused)


def test_a_refusal_never_echoes_a_submitted_value() -> None:
    """No tool takes a secret today (``SECRET_PARAMS`` is empty); the day one does, its value must
    not come back in an error. Short on purpose: pydantic cuts a long repr in the middle."""
    mcp, seen = _server()
    value = "v4lu3Zq"
    with pytest.raises(ToolError) as caught:
        _call(mcp, "forge_list_runs", {"status": value, UNKNOWN: value})
    assert UNKNOWN in str(caught.value)
    assert value not in str(caught.value)
    assert "input_value" not in str(caught.value)
    with pytest.raises(ToolError) as caught:
        _call(mcp, "forge_list_runs", {"status": [value]})
    assert value not in str(caught.value)
    assert seen == []


def test_the_published_schema_says_no_other_names() -> None:
    mcp, _ = _server()
    loose = [
        t.name
        for t in asyncio.run(mcp.list_tools())
        if t.input_schema.get("additionalProperties") is not False
    ]
    assert not loose, f"input schemas that allow other names: {loose}"


async def test_over_the_wire_an_unknown_name_is_an_error_result(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    async def record(self: ForgeClient, method: str, path: str, **_kw: Any) -> Any:
        calls.append((method, path))
        return []

    monkeypatch.setattr(ForgeClient, "request", record)
    async with serving(app) as client:
        response = await _rpc(
            client,
            "tools/call",
            {"name": "forge_list_runs", "arguments": {"state": "running"}},
        )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["isError"] is True, json.dumps(result)[:500]
    assert "unknown argument(s) state" in result["content"][0]["text"]
    assert calls == []
