"""The output-schema overlays are exactly the tools whose answer is not their route's (021
FR-021.18; FT6, FTID §8.2), and the withheld keys stay withheld (FPRD A4).

- **Sentinel.** Every live tool is called with its ledger arguments against a recording client
  that answers one SENTINEL body. A tool that answers something other than the sentinel reshapes
  its route's answer; with the tools that share a route with another tool, and the tools whose
  schema would otherwise name a withheld key (FTDD §8), that set EQUALS
  ``backend/scripts/output_overlays.py``'s ``OVERLAYS``.
- **Hand schema.** ``forge_howto``'s schema validates its real answer for every topic and for none.
- **Withheld.** ``forge_wsp_list_endpoints`` against the real app, with an endpoint that has all
  four prices and a sealed key, answers none of the five ``WITHHELD`` keys, its schema declares
  none of them, and the operator's own route read carries all five.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import pytest

from mcp_server.server import OUTPUT_SCHEMAS_PATH
from mcp_server.tools.wsp import WITHHELD
from tests.integration.wsp.test_endpoints_api import ANTHROPIC
from tests.integration.wsp.test_endpoints_api import env as env  # the endpoint world (a fixture)
from tests.support.mcp_harness import build_harness, call_tool
from tests.support.mcp_ledger import EXPECTED_CALLS, full, normalise

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "backend" / "scripts"))
from output_overlays import OVERLAYS  # noqa: E402 - the generator's own table

SENTINEL: dict[str, Any] = {
    "sentinel": True,
    "endpoints": [{"id": "e1", **dict.fromkeys(WITHHELD, "kept")}],
}


def _sharing() -> set[str]:
    by_route: dict[tuple[str, str], list[str]] = {}
    for tool, (method, path, *_rest) in EXPECTED_CALLS.items():
        by_route.setdefault((method, normalise(full(path))), []).append(tool)
    return {t for tools in by_route.values() if len(tools) > 1 for t in tools}


def test_the_overlays_are_the_tools_that_reshape_or_share_a_route() -> None:
    mcp, _client = build_harness(response=SENTINEL, check=False)
    tools = {t.name for t in asyncio.run(mcp.list_tools())}
    reshaping = set()
    for tool in sorted(tools):
        args = EXPECTED_CALLS[tool][2] if tool in EXPECTED_CALLS else {}
        answer = call_tool(mcp, tool, args, check=False)
        if answer != SENTINEL:
            reshaping.add(tool)
    # The third reason (FTDD §8): an overlay that keeps a withheld key out of a tool's SCHEMA
    # where the route's answer, projected without it, would name one (review round 2, F5).
    needed = reshaping | (_sharing() & tools) | _schema_needs_a_withhold(tools)
    assert needed == set(OVERLAYS), {
        "reshape without an overlay": sorted(needed - set(OVERLAYS)),
        "overlay without a reason": sorted(set(OVERLAYS) - needed),
    }


def _schema_needs_a_withhold(tools: set[str]) -> set[str]:
    """The tools whose projected schema, with no overlay, names a withheld key."""
    import generate_output_schemas as generator

    from mcp_server.tools.wsp import RUN_READ_WITHHELD

    contract = json.loads((REPO / "docs" / "schemas" / "miforge-answers-v1.json").read_text())
    language = json.loads(generator.RESULT_LANGUAGE.read_text())
    calls = {t: (EXPECTED_CALLS[t][0], EXPECTED_CALLS[t][1]) for t in tools if t in EXPECTED_CALLS}
    hand = {t: o for t, o in OVERLAYS.items() if o.hand is not None}
    bare = generator.output_transform(contract, calls, hand, language, set())
    named = set()
    for tool, schema in bare.items():
        if set(RUN_READ_WITHHELD) & set(json.dumps(schema).split('"')):
            named.add(tool)
    return named


def test_no_tool_schema_names_a_withheld_key() -> None:
    """FTDD §8: the four prices, the key's fingerprint and the key itself appear in no tool's
    output schema (as a property name), whatever the routes behind them answer."""
    from mcp_server.tools.wsp import RUN_READ_WITHHELD

    document = json.loads(OUTPUT_SCHEMAS_PATH.read_text(encoding="utf-8"))
    named: dict[str, list[str]] = {}
    for tool, schema in document["typed"].items():
        stack: list[Any] = [schema]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                props = node.get("properties")
                if isinstance(props, dict):
                    hits = sorted(set(props) & set(RUN_READ_WITHHELD))
                    if hits:
                        named.setdefault(tool, []).extend(hits)
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    assert not named, named


def test_each_continue_tool_takes_only_its_own_variant() -> None:
    """FR-021.17's variant rule: the plan answer is forge_plan_continue's and not
    forge_continue_run's; the submitted continue the reverse (one route, two tools)."""
    from tests.support.answer_examples import example_answer

    route = "POST /api/v1/runs/{run_id}/continue"
    plan, submitted = example_answer(route, "200"), example_answer(route, "202")
    plan_tool = jsonschema.Draft202012Validator(_schema("forge_plan_continue"))
    continue_tool = jsonschema.Draft202012Validator(_schema("forge_continue_run"))
    assert plan_tool.is_valid(plan) and not continue_tool.is_valid(plan)
    assert continue_tool.is_valid(submitted) and not plan_tool.is_valid(submitted)


def _schema(tool: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(OUTPUT_SCHEMAS_PATH.read_text(encoding="utf-8"))
    return dict(document["typed"][tool])


def test_the_howto_schema_validates_every_topic_and_none() -> None:
    from mcp_server.tools.core import HOWTO

    mcp, _client = build_harness()
    validator = jsonschema.Draft202012Validator(_schema("forge_howto"))
    for topic in [None, *sorted(HOWTO)]:
        args = {} if topic is None else {"topic": topic}
        answer = call_tool(mcp, "forge_howto", args)
        assert validator.is_valid(answer), (topic, answer)
    assert not validator.is_valid({"topic": "results"}), "a topic without its text passed"


def test_the_withheld_keys_are_withheld_from_the_tool(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An endpoint with all four prices and a sealed key, registered by the operator."""
    from mcp_server.client import ForgeClient
    from mcp_server.config import MCPSettings
    from mcp_server.health_gate import HealthGate
    from mcp_server.server import build_server

    async def open_gate(self: Any, *_: str) -> None:  # the backend is the app itself here
        return None

    monkeypatch.setattr(HealthGate, "unavailable", open_gate)

    created = env.client.post("/api/v1/wsp/endpoints", json=ANTHROPIC)
    assert created.status_code == 201, created.text
    priced_endpoint = ANTHROPIC["id"]
    operator = env.client.get("/api/v1/wsp/endpoints").json()["endpoints"]
    row = next(r for r in operator if r["id"] == priced_endpoint)
    assert all(row.get(key) is not None for key in WITHHELD), row

    mcp, client = build_server(MCPSettings(auth_token="x" * 32))
    assert isinstance(client, ForgeClient)
    client._http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=env.client.app), base_url="http://forge.test/api/v1"
    )
    result = asyncio.run(mcp.call_tool("forge_wsp_list_endpoints", {}))
    answer = result.structured_content
    assert answer is not None
    tool_row = next(r for r in answer["endpoints"] if r["id"] == priced_endpoint)
    assert not set(WITHHELD) & set(tool_row), tool_row
    # FPRD A4 names the five, independently of the code's list: every price the operator's row
    # carries, and the key's fingerprint.
    named = {k for k in row if k.startswith("price_")} | {"key_fingerprint"}
    assert len(named) == 5 and set(WITHHELD) == named, sorted(named ^ set(WITHHELD))
    assert not named & set(tool_row), tool_row
    schema = _schema("forge_wsp_list_endpoints")
    assert not set(WITHHELD) & set(json.dumps(schema).split('"'))
    jsonschema.Draft202012Validator(schema).validate(answer)
