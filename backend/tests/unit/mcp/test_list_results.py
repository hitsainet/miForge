# Origin: miDataworks (Onegaishimas/miDataworks) backend/tests/unit/mcp/test_list_results.py
# @ b858ed3 - ported (017 FTID I7; docs/REUSE.md): miForge's client, ledger and tool names.
"""A list result is ONE content block, through the LIVE server (FR-017.21; survey gap 7).

The SDK renders a returned list as one block per item, so an empty list reached the client as
``content: []``: it could not be told from a dropped response. ``server.shape_results`` serialises
every list (and ``None``). These tests go through ``build_server`` (the registry production
serves) and the real ``ForgeClient`` over an ``httpx`` mock transport, so removing the
``shape_results`` line in ``build_server`` turns them red.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from mcp_server.config import MCPSettings
from mcp_server.health_gate import HealthGate
from mcp_server.server import build_server
from tests.support.mcp_ledger import CALLER_ASSERTION_EXEMPT, EXPECTED_CALLS


@pytest.fixture(autouse=True)
def _open_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unavailable(self: Any, *_: str) -> None:
        return None

    monkeypatch.setattr(HealthGate, "unavailable", unavailable)


def _server(answer: Any) -> tuple[Any, list[httpx.Request]]:
    mcp, client = build_server(MCPSettings(auth_token="x" * 32))
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=answer)

    client._http = httpx.AsyncClient(  # the transport is the only stand-in
        base_url="http://backend/api/v1",
        transport=httpx.MockTransport(handler),
        headers={"X-Miforge-Agent": client.identity},
    )
    return mcp, seen


def _blocks(result: Any) -> list[str]:
    return [c.text for c in result.content]


def test_an_empty_list_is_one_block_saying_empty() -> None:
    mcp, seen = _server([])
    result = asyncio.run(mcp.call_tool("forge_list_selectors", {}))
    assert _blocks(result) == ["[]"]
    assert [(r.method, r.url.path) for r in seen] == [("GET", "/api/v1/selectors")]


def test_a_list_of_two_is_one_block_holding_the_array() -> None:
    mcp, _ = _server([{"name": "rule"}, {"name": "judge"}])
    result = asyncio.run(mcp.call_tool("forge_list_selectors", {}))
    blocks = _blocks(result)
    assert len(blocks) == 1
    assert json.loads(blocks[0]) == [{"name": "rule"}, {"name": "judge"}]


def test_a_dict_result_is_unchanged() -> None:
    mcp, _ = _server({"runs": [], "total": 0})
    result = asyncio.run(mcp.call_tool("forge_list_runs", {}))
    blocks = _blocks(result)
    assert len(blocks) == 1 and json.loads(blocks[0]) == {"runs": [], "total": 0}


@pytest.mark.parametrize("tool", sorted(EXPECTED_CALLS))
def test_every_tool_answers_an_empty_list_with_exactly_one_block(tool: str) -> None:
    """Any tool whose route answers ``[]`` gives the client one block, never zero."""
    _, _, args, _, _ = EXPECTED_CALLS[tool]
    mcp, seen = _server([])
    result = asyncio.run(mcp.call_tool(tool, dict(args)))
    assert len(result.content) == 1, (tool, result.content)
    assert len(seen) >= 1


def test_every_ledger_tool_was_covered() -> None:
    """The parametrisation read the whole registry, not an empty ledger."""
    mcp, _ = _server([])
    live = {t.name for t in asyncio.run(mcp.list_tools())}
    assert set(EXPECTED_CALLS) | set(CALLER_ASSERTION_EXEMPT) == live
    assert len(EXPECTED_CALLS) >= 89
