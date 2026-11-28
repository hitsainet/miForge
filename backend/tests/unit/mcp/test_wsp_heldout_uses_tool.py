"""``forge_wsp_list_heldout_uses`` (2026-11-27): each held-out set's use count over
``GET /api/v1/wsp/heldout-uses``. The phase-2 acceptance round read ``uses_before`` over REST on the
node because no tool read it.

- it is in the LIVE registry (``build_server``), in the ``wsp`` category, read-only and typed;
- it makes exactly one request, ``GET /api/v1/wsp/heldout-uses`` with no query and no body, and
  answers what the route answered;
- against the real app, it serves the rows the table holds, and the answer validates against the
  tool's published output schema.

Control: removing its registration (``@mcp.tool`` in ``tools/wsp.py``) turns this file red.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from typing import Any

import httpx
import jsonschema
import pytest

from mcp_server.config import MCPSettings
from mcp_server.health_gate import HealthGate
from mcp_server.server import OUTPUT_SCHEMAS_PATH, build_server
from tests.integration.wsp.test_endpoints_api import env as env  # the WSP world (a fixture)

TOOL = "forge_wsp_list_heldout_uses"
ROW = {
    "owner": "probe-ssn",
    "set_name": "heldout-a",
    "row_ids_sha256": "a" * 64,
    "heldout_uses": 2,
    "last_used_at": "2026-11-26T12:00:00+00:00",
    "last_candidate_id": "wspc_0123456789a",
}


@pytest.fixture(autouse=True)
def _open_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unavailable(self: Any, *_: str) -> None:
        return None

    monkeypatch.setattr(HealthGate, "unavailable", unavailable)


def _schema() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(OUTPUT_SCHEMAS_PATH.read_text(encoding="utf-8"))
    return dict(document["typed"][TOOL])


def test_it_is_registered_read_only_in_the_wsp_category_and_typed() -> None:
    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    tool = tools[TOOL]
    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is True
    assert tool.annotations.destructive_hint is False
    assert tool.input_schema.get("properties", {}) == {}
    assert tool.output_schema == _schema()
    off, _ = build_server(MCPSettings(auth_token="x" * 32, tool_categories="core"))
    assert TOOL not in {t.name for t in asyncio.run(off.list_tools())}


def test_it_reads_the_route_once_and_answers_what_it_answered() -> None:
    answer = {"heldout_uses": [ROW]}
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
    result = asyncio.run(mcp.call_tool(TOOL, {}))
    assert [(r.method, r.url.path, r.url.query, r.content) for r in seen] == [
        ("GET", "/api/v1/wsp/heldout-uses", b"", b"")
    ]
    assert result.structured_content == answer


def test_against_the_real_app_it_serves_the_tables_rows(env: Any) -> None:
    from sqlalchemy.orm import Session

    from wsp.models import WspHeldoutUse

    with Session(env.engine) as session:
        session.add(
            WspHeldoutUse(
                **{**ROW, "last_used_at": dt.datetime(2026, 11, 26, 12, tzinfo=dt.UTC)},
            )
        )
        session.commit()

    mcp, client = build_server(MCPSettings(auth_token="x" * 32))
    client._http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=env.client.app), base_url="http://forge.test/api/v1"
    )
    result = asyncio.run(mcp.call_tool(TOOL, {}))
    answer = result.structured_content
    assert answer is not None
    (row,) = answer["heldout_uses"]
    assert {k: row[k] for k in ROW if k != "last_used_at"} == {
        k: v for k, v in ROW.items() if k != "last_used_at"
    }
    assert dt.datetime.fromisoformat(row["last_used_at"].replace("Z", "+00:00")) == dt.datetime(
        2026, 11, 26, 12, tzinfo=dt.UTC
    )
    jsonschema.Draft202012Validator(_schema()).validate(answer)
