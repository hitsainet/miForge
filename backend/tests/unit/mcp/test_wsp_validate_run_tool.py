"""``forge_wsp_validate_run`` (2026-11-26): a dry run of a WSP start body, through the LIVE server
(``build_server``, the registry production serves) and the real ``ForgeClient`` over an ``httpx``
mock transport, so removing the tool's registration or its call turns these red.

``POST /wsp/runs/validate`` existed and no tool called it, so an agent could not check a start body
before filing a start that waits for the operator.

- registered in the ``wsp`` category, read-only, ungated, with the start tool's input schema;
- exactly one ``POST /api/v1/wsp/runs/validate`` carrying the body unchanged, and nothing else;
- the route's answer returned as it came; the published output schema is the route's typed
  answer (``schemas/wsp_validate.py``).
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
from tests.support.answer_examples import example_answer
from tests.support.mcp_gate_probe import decision
from tests.support.mcp_ledger import EXPECTED_CALLS

BODY = {
    "bundle": {"owner": "miworkbench", "bundle_id": "b-1", "bundle_sha256": "0" * 64},
    "kind": "repair",
    "mode": "session",
    "task": "fewer false alarms",
    "endpoint_id": "ep-local",
    "sensitivity": "not_sensitive",
    "primary_sets": ["eval"],
    "budgets": {"tokens": 1000, "max_steps": 4},
    "python": "off",
    "caller_ref": "check-first",
}
#: The route's answer (``wsp_ops.py:validate_run``), abridged where the tool does not look.
ANSWER = {
    # 021: every field the route sends (the contract's example), then this test's values.
    **example_answer("POST /api/v1/wsp/runs/validate"),
    "request": {
        **example_answer("POST /api/v1/wsp/runs/validate")["request"],
        "kind": "repair",
        "mode": "session",
    },
    "request_sha256": "a" * 64,
    "approval_action": "wsp_session",
    "locality": "local",
    "sensitivity": {"value": "not_sensitive", "effective": "not_sensitive"},
    "caller_ref": "check-first",
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


def _tools(**settings: Any) -> dict[str, Any]:
    mcp, _ = build_server(MCPSettings(auth_token="x" * 32, **settings))
    return {t.name: t for t in asyncio.run(mcp.list_tools())}


def test_it_is_registered_read_only_in_the_wsp_category_with_the_start_tools_body() -> None:
    tools = _tools()
    tool = tools["forge_wsp_validate_run"]
    assert tool.annotations is not None and tool.annotations.read_only_hint is True
    assert tool.annotations.destructive_hint is False
    # The same body as the start, so what validates is what a start takes.
    start = tools["forge_wsp_start_run"].input_schema
    assert {k: v for k, v in tool.input_schema.items() if k != "title"} == {
        k: v for k, v in start.items() if k != "title"
    }
    assert "forge_wsp_validate_run" not in _tools(tool_categories="core")


def test_one_post_with_the_body_unchanged_and_the_answer_as_it_came() -> None:
    mcp, seen = _server(ANSWER)
    result = asyncio.run(mcp.call_tool("forge_wsp_validate_run", {"body": BODY}))
    assert len(seen) == 1
    (request,) = seen
    assert (request.method, request.url.path, request.url.query) == (
        "POST",
        "/api/v1/wsp/runs/validate",
        b"",
    )
    assert json.loads(request.content) == BODY
    assert request.headers["X-Miforge-Agent"]
    assert json.loads(result.content[0].text) == ANSWER


def test_a_refusal_is_returned_not_retried() -> None:
    refusal = {"code": "wsp_bundle_not_found", "message": "No such bundle; import it first."}
    mcp, seen = _server({"error": refusal}, status=404)
    result = asyncio.run(mcp.call_tool("forge_wsp_validate_run", {"body": BODY}))
    assert len(seen) == 1
    assert "wsp_bundle_not_found" in result.content[0].text


def test_it_files_no_approval_for_the_ledgers_body() -> None:
    """Ungated: the live gate decides no action for its call (the start's own body waits as
    wsp_session)."""
    assert EXPECTED_CALLS["forge_wsp_validate_run"][4] == "plan"
    assert decision("forge_wsp_validate_run")[1] is None
    assert decision("forge_wsp_start_run")[1] == "wsp_session"


def test_its_output_schema_is_the_routes_typed_answer() -> None:
    from schemas.wsp_validate import WspRunValidation

    schema = _tools()["forge_wsp_validate_run"].output_schema
    assert schema is not None
    # 021 P5: the schema is the route's contract answer at the root's anyOf (FTDD §6.6).
    (answer,) = [b for b in schema["anyOf"] if b != {"$ref": "#/$defs/Unavailable"}]
    target = schema["$defs"][answer["$ref"].rsplit("/", 1)[-1]]
    assert set(WspRunValidation.model_fields) <= set(target["properties"])
