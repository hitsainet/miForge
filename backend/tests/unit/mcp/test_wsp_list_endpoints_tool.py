"""``forge_wsp_list_endpoints`` (2026-11-25, ENH-010 finish plan track A): a read of the editor
endpoints through the LIVE server (``build_server``, the registry production serves) and the real
``ForgeClient`` over an ``httpx`` mock transport, so removing the tool's registration, its call or
its filter turns these red.

- registered in the ``wsp`` category, read-only, ungated;
- exactly one ``GET /api/v1/wsp/endpoints``, with no query and no body;
- every field of the route's answer, except the four prices and the key's fingerprint.
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
from mcp_server.tools import wsp as wsp_tools
from tests.support.answer_examples import example_of

#: One endpoint as ``GET /wsp/endpoints`` serves it (``wsp.endpoints.public_document``).
ROW = {
    "id": "ep-ext",
    "kind": "anthropic",
    "base_url": "https://api.example.test",
    "model": "m",
    "locality": "external",
    "serving": "generic",
    "output_cap_field": "max_tokens",
    "price_in_per_mtok": "3",
    "price_out_per_mtok": "15",
    "price_cache_write_per_mtok": "3.75",
    "price_cache_read_per_mtok": "0.3",
    "has_key": True,
    "key_fingerprint": "sha256:abcdef",
    "created_at": "2026-11-25T00:00:00+00:00",
    "created_by": "operator",
    "action_protocol": "tool_calls",
    "context_window_tokens": 200000,
    "thinking": "default",
    "thinking_tokens": None,
    "output_cap": "on",
    "paging": "off",
    "output_ceiling_tokens": None,
    "history_mode": "append_only",
    "runs_refused": None,
    "output_ceiling_effective": 32000,
    "registration_window": {"tokens": None, "read_at": None},
    "latest_run_window": None,
    "window_now": {"tokens": 200000, "source": "stated", "run_id": None, "at": None},
    "window_if_read_fails": {"tokens": 200000, "source": "stated", "no_answer": "stated"},
    "window_warnings": [],
    # 021: a protocol test in its full shape (the contract's example; this test checks only that
    # the row passes through).
    "protocol_test": example_of("WspEndpointProtocolTest"),
}
WITHHELD = {
    "price_in_per_mtok",
    "price_out_per_mtok",
    "price_cache_write_per_mtok",
    "price_cache_read_per_mtok",
    "key_fingerprint",
}


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


def _structured(result: Any) -> Any:
    return json.loads(result.content[0].text)


def test_it_is_registered_read_only_in_the_wsp_category() -> None:
    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    tool = tools["forge_wsp_list_endpoints"]
    assert tool.annotations is not None and tool.annotations.read_only_hint is True
    assert tool.annotations.destructive_hint is False
    assert tool.input_schema.get("properties", {}) == {}
    off, _ = build_server(MCPSettings(auth_token="x" * 32, tool_categories="core"))
    assert "forge_wsp_list_endpoints" not in {t.name for t in asyncio.run(off.list_tools())}


def test_one_get_and_every_field_but_the_prices_and_the_key() -> None:
    mcp, seen = _server({"endpoints": [ROW, {**ROW, "id": "ep-local", "has_key": False}]})
    result = _structured(asyncio.run(mcp.call_tool("forge_wsp_list_endpoints", {})))
    assert [(r.method, r.url.path, r.url.query, r.content) for r in seen] == [
        ("GET", "/api/v1/wsp/endpoints", b"", b"")
    ]
    rows = result["endpoints"]
    assert [r["id"] for r in rows] == ["ep-ext", "ep-local"]
    for row in rows:
        assert set(row) == set(ROW) - WITHHELD
        assert {k: row[k] for k in row if k != "id" and k != "has_key"} == {
            k: v for k, v in ROW.items() if k not in WITHHELD | {"id", "has_key"}
        }
    assert "3.75" not in json.dumps(result) and "abcdef" not in json.dumps(result)


def test_the_fixture_row_is_the_routes_row() -> None:
    """The row above carries exactly the keys the route serves (020's ``wsp_list_keys``, pinned
    from the route's real answer), so a key the route adds reaches this test."""
    from tests.support.wsp_list_keys import ENDPOINT_ROW_KEYS

    assert set(ROW) == ENDPOINT_ROW_KEYS


def test_the_withheld_list_is_the_routes_price_fields_and_the_fingerprint() -> None:
    from wsp.endpoints import PRICE_FIELDS

    assert set(wsp_tools.WITHHELD) == set(PRICE_FIELDS) | {"key_fingerprint"}


def test_another_answer_passes_through_unchanged() -> None:
    assert wsp_tools.without_prices_or_key({"unavailable": "backend"}) == {"unavailable": "backend"}
    assert wsp_tools.without_prices_or_key([1]) == [1]
