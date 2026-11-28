"""The WSP run tools (2026-11-26, the 019 P4 MCP gaps): ``forge_wsp_get_run``,
``forge_wsp_get_transcript``, ``forge_wsp_get_run_ledger`` and ``forge_wsp_replay_run``, through
the LIVE server (``build_server``, the registry production serves) and the real ``ForgeClient``
over an ``httpx`` mock transport, so removing a registration, a call, its query or the withheld-key
filter turns these red.

- each is registered in the ``wsp`` category with its hint (three reads, one write), ungated;
- each makes exactly one request, with exactly the documented method, path, query and body;
- no answer carries a price, the key's fingerprint or an ``api_key``, at any depth.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import parse_qsl

import httpx
import pytest

from mcp_server.config import MCPSettings
from mcp_server.health_gate import HealthGate
from mcp_server.server import build_server
from mcp_server.tools import wsp as wsp_tools
from tests.support.answer_examples import example_answer

RUN = "wsp_0123456789ab"
TOOLS = {
    "forge_wsp_get_run": "read",
    "forge_wsp_get_transcript": "read",
    "forge_wsp_get_run_ledger": "read",
    "forge_wsp_replay_run": "write",
}
#: Planted where a careless route could put them: at the top, in a nested object and in a list.
LEAKS = {
    "price_in_per_mtok": "PLANTED-PRICE-IN",
    "price_out_per_mtok": "PLANTED-PRICE-OUT",
    "price_cache_write_per_mtok": "PLANTED-PRICE-CW",
    "price_cache_read_per_mtok": "PLANTED-PRICE-CR",
    "key_fingerprint": "PLANTED-FINGERPRINT",
    "api_key": "PLANTED-KEY",
}


@pytest.fixture(autouse=True)
def _open_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unavailable(self: Any, *_: str) -> None:
        return None

    monkeypatch.setattr(HealthGate, "unavailable", unavailable)


def _server(answer: Any, status: int = 200) -> tuple[Any, list[httpx.Request]]:
    mcp, client = build_server(MCPSettings(auth_token="x" * 32))
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=answer)

    client._http = httpx.AsyncClient(  # the transport is the only stand-in
        base_url="http://backend/api/v1",
        transport=httpx.MockTransport(handler),
        headers={"X-Miforge-Agent": client.identity},
    )
    return mcp, seen


def _call(mcp: Any, tool: str, args: dict[str, Any]) -> Any:
    result = asyncio.run(mcp.call_tool(tool, args))
    return json.loads(result.content[0].text)


def _query(request: httpx.Request) -> dict[str, str]:
    return dict(parse_qsl(request.url.query.decode()))


def test_each_is_registered_in_the_wsp_category_with_its_hint() -> None:
    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    for name, kind in TOOLS.items():
        tool = tools[name]
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is (kind == "read"), name
        assert tool.annotations.destructive_hint is False, name
        assert tool.input_schema["properties"]["run_id"]["pattern"] == wsp_tools.WSP_RUN_ID_PATTERN
    off, _ = build_server(MCPSettings(auth_token="x" * 32, tool_categories="core"))
    assert not set(TOOLS) & {t.name for t in asyncio.run(off.list_tools())}


@pytest.mark.mcp_fake_is_not_a_route_answer(
    reason="plants every withheld key at every depth, in a partial shape, to prove the tool "
    "removes them; the route never sends them"
)
def test_get_run_reads_the_record_once_and_drops_every_withheld_key() -> None:
    answer = {
        "run_id": RUN,
        "row": {"run_id": RUN, "status": "completed", "metered": {"cost_usd": 0.0, **LEAKS}},
        "document": {
            "kind": "miforge.wsp-run/v1",
            "request": {"endpoint": {"id": "ep", "model": "m", **LEAKS}},
            "lineage": {"chain": [{"ref": "x", **LEAKS}]},
            **LEAKS,
        },
        **LEAKS,
    }
    mcp, seen = _server(answer)
    result = _call(mcp, "forge_wsp_get_run", {"run_id": RUN})
    assert [(r.method, r.url.path, r.url.query, r.content) for r in seen] == [
        ("GET", f"/api/v1/wsp/runs/{RUN}/record", b"", b"")
    ]
    text = json.dumps(result)
    for key, value in LEAKS.items():
        assert key not in text and value not in text, key
    assert result["row"]["metered"] == {"cost_usd": 0.0}
    assert result["document"]["request"]["endpoint"] == {"id": "ep", "model": "m"}
    assert result["document"]["lineage"]["chain"] == [{"ref": "x"}]


@pytest.mark.mcp_fake_is_not_a_route_answer(
    reason="plants every withheld key at every depth, in a partial shape, to prove the tool "
    "removes them; the route never sends them"
)
def test_get_transcript_always_sends_a_bounded_page() -> None:
    page = {"run_id": RUN, "entries": [{"turn": 1, **LEAKS}], "offset": 0, "total": 1}
    mcp, seen = _server(page)
    result = _call(mcp, "forge_wsp_get_transcript", {"run_id": RUN})
    (request,) = seen
    assert (request.method, request.url.path, request.content) == (
        "GET",
        f"/api/v1/wsp/runs/{RUN}/transcript",
        b"",
    )
    assert _query(request) == {
        "offset": "0",
        "limit": str(wsp_tools.TRANSCRIPT_PAGE_DEFAULT),
        "max_bytes": str(wsp_tools.TRANSCRIPT_BYTES_DEFAULT),
    }
    assert result["entries"] == [{"turn": 1}]

    mcp, seen = _server(page)
    _call(mcp, "forge_wsp_get_transcript", {"run_id": RUN, "offset": 40, "limit": 100})
    assert _query(seen[0])["offset"] == "40" and _query(seen[0])["limit"] == "100"


@pytest.mark.parametrize(
    "args",
    [
        {"limit": wsp_tools.TRANSCRIPT_PAGE_MAX + 1},
        {"limit": 0},
        {"max_bytes": wsp_tools.TRANSCRIPT_BYTES_MAX + 1},
        {"max_bytes": 100},
        {"offset": -1},
    ],
)
def test_get_transcript_refuses_an_unbounded_page_before_any_request(args: dict[str, Any]) -> None:
    mcp, seen = _server({})
    result = asyncio.run(mcp.call_tool("forge_wsp_get_transcript", {"run_id": RUN, **args}))
    assert result.is_error
    assert seen == []


@pytest.mark.mcp_fake_is_not_a_route_answer(
    reason="plants every withheld key at every depth, in a partial shape, to prove the tool "
    "removes them; the route never sends them"
)
def test_get_run_ledger_sends_only_the_cursors_given() -> None:
    answer = {
        "run_id": RUN,
        "reservations": [{"id": 1, "endpoint_id": "ep", **LEAKS}],
        "usage": [{"seq": 4, "endpoint_id": "ep"}],
        "totals": {"endpoint_ids": ["ep"]},
    }
    mcp, seen = _server(answer)
    result = _call(mcp, "forge_wsp_get_run_ledger", {"run_id": RUN, "after_seq": 3})
    (request,) = seen
    assert (request.method, request.url.path, request.content) == (
        "GET",
        f"/api/v1/wsp/runs/{RUN}/ledger",
        b"",
    )
    assert _query(request) == {"after_seq": "3"}
    assert result["reservations"] == [{"id": 1, "endpoint_id": "ep"}]

    mcp, seen = _server(answer)
    _call(
        mcp,
        "forge_wsp_get_run_ledger",
        {"run_id": RUN, "after_reservation_id": 9, "limit": 5},
    )
    assert _query(seen[0]) == {"after_reservation_id": "9", "limit": "5"}


def test_replay_posts_once_with_no_body_and_is_not_gated() -> None:
    replayed = {
        **example_answer("POST /api/v1/wsp/runs/{run_id}/replay"),  # 021: the route's whole answer
        "run_id": "wsp_ffffffffffff",
        "status": "queued",
    }
    mcp, seen = _server(replayed, status=201)
    result = _call(mcp, "forge_wsp_replay_run", {"run_id": RUN})
    assert [(r.method, r.url.path, r.url.query) for r in seen] == [
        ("POST", f"/api/v1/wsp/runs/{RUN}/replay", b"")
    ]
    assert seen[0].content in (b"", b"null")
    assert result["run_id"] == "wsp_ffffffffffff" and "approval" not in result


@pytest.mark.parametrize("tool", sorted(TOOLS))
@pytest.mark.parametrize("bad", ["run-1", "wsp_../x", "wsp_", "wsp_0123456789abc", "../wsp_1"])
def test_a_run_id_that_is_not_a_wsp_id_is_refused_before_any_request(tool: str, bad: str) -> None:
    mcp, seen = _server({})
    result = asyncio.run(mcp.call_tool(tool, {"run_id": bad}))
    assert result.is_error
    assert seen == []


def test_a_refusal_passes_through_unchanged() -> None:
    assert wsp_tools.withheld_removed({"unavailable": "backend", "reason": "x"}) == {
        "unavailable": "backend",
        "reason": "x",
    }
    assert wsp_tools.withheld_removed([{"api_key": 1, "a": [{"key_fingerprint": 2}]}]) == [
        {"a": [{}]}
    ]


def test_the_run_reads_withhold_the_endpoint_lists_keys_and_api_key() -> None:
    assert set(wsp_tools.RUN_READ_WITHHELD) == set(wsp_tools.WITHHELD) | {"api_key"}
