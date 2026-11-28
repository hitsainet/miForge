"""G1 (FR-012.5, MR-11): the tool inventory equals the agent-callable route set, LIVE.

Every ``(method, path)`` in the app's OpenAPI document has a tool that calls it (an
``EXPECTED_CALLS`` entry whose path matches the template) or a ``ROUTE_EXEMPT`` reason of at
least 20 characters; no route is both; no ledger entry names a route nothing serves; and the
live registry equals the ledger in both directions. ``app.openapi()``, never ``app.routes``
alone: the document is what a proxy or an agent sees.
"""

from __future__ import annotations

import asyncio
from functools import cache
from typing import Any

from mcp_server.config import MCPSettings
from mcp_server.context import ESCALATES_TO_ATTACH, GATED_TOOLS
from mcp_server.server import build_server
from tests.support.mcp_gate_probe import decision
from tests.support.mcp_ledger import (
    CALLER_ASSERTION_EXEMPT,
    EXPECTED_CALLS,
    LEDGER_TOOL_TOTAL,
    NOT_IN_SCHEMA,
    ROUTE_EXEMPT,
    full,
    normalise,
    template_matches,
)

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


@cache
def served() -> dict[tuple[str, str], dict[str, Any]]:
    """``{(METHOD, normalised template): operation}`` from the live OpenAPI document."""
    from api.app import create_app

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for path, operations in create_app().openapi()["paths"].items():
        for method, operation in operations.items():
            if method in HTTP_METHODS:
                out[(method.upper(), normalise(path))] = {**operation, "_template": path}
    return out


def tool_route(tool: str) -> tuple[str, str] | None:
    method, path, _, _, _ = EXPECTED_CALLS[tool]
    concrete = full(path)
    matches = [
        key
        for key, operation in served().items()
        if key[0] == method and template_matches(operation["_template"], concrete)
    ]
    return min(matches, key=lambda key: key[1].count("{}"), default=None)


def registered_tools() -> set[str]:
    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    return {t.name for t in asyncio.run(mcp.list_tools())}


def test_the_scan_sees_the_api() -> None:
    assert len(served()) >= 80, f"the live OpenAPI document lists only {len(served())} routes"


def test_every_served_route_has_a_tool_or_a_reasoned_exemption() -> None:
    called = {tool_route(t) for t in EXPECTED_CALLS}
    missing = {key for key in served() if key not in called and key not in ROUTE_EXEMPT}
    assert not missing, (
        f"SERVED ROUTES WITH NO TOOL: {sorted(missing)}. Add a tool (and its EXPECTED_CALLS "
        "entry) or a ROUTE_EXEMPT reason in tests/support/mcp_ledger.py (FR-012.5)."
    )


def test_no_route_is_both_called_and_exempted() -> None:
    called = {tool_route(t) for t in EXPECTED_CALLS}
    both = called & set(ROUTE_EXEMPT)
    assert not both, f"routes both exempted and called by a tool: {sorted(both)}"


def test_every_exemption_has_a_reason() -> None:
    thin = [
        key for key, reason in {**ROUTE_EXEMPT, **NOT_IN_SCHEMA}.items() if len(reason.strip()) < 20
    ]
    thin += [tool for tool, reason in CALLER_ASSERTION_EXEMPT.items() if len(reason.strip()) < 20]
    assert not thin, f"exemptions without a reason: {thin}"


def test_every_tool_calls_a_served_route() -> None:
    dangling = {t: EXPECTED_CALLS[t][:2] for t in EXPECTED_CALLS if tool_route(t) is None}
    assert not dangling, f"tools calling routes nothing serves: {dangling}"


def test_every_exemption_names_a_served_route() -> None:
    stale = sorted(key for key in ROUTE_EXEMPT if key not in served())
    assert not stale, f"exemptions for routes that are not served: {stale}"


def test_the_routes_outside_the_schema_are_still_outside_it() -> None:
    for path in NOT_IN_SCHEMA:
        assert not any(t == normalise(path) for _, t in served()), path


def test_the_registry_equals_the_ledger() -> None:
    registered = registered_tools()
    accounted = set(EXPECTED_CALLS) | set(CALLER_ASSERTION_EXEMPT)
    assert registered == accounted, (
        f"registered but not in the ledger: {sorted(registered - accounted)}; in the ledger but "
        f"not registered: {sorted(accounted - registered)}"
    )
    assert len(registered) == LEDGER_TOOL_TOTAL


def test_gated_tools_state_what_the_live_gate_decides() -> None:
    """``GATED_TOOLS`` is a statement in descriptions and the contract; the gate is REST's.
    The statement must equal what the gate decides for the call each tool actually makes."""
    for tool in EXPECTED_CALLS:
        _route, action = decision(tool)
        assert GATED_TOOLS.get(tool) == action, (tool, GATED_TOOLS.get(tool), action)
    stale = set(GATED_TOOLS) - set(EXPECTED_CALLS)
    assert not stale, f"GATED_TOOLS names tools that do not exist: {sorted(stale)}"
    assert ESCALATES_TO_ATTACH <= set(GATED_TOOLS)


def test_every_gated_route_is_reached_by_a_gated_tool_or_is_exempt() -> None:
    called_gated = {tool_route(t) for t in EXPECTED_CALLS if decision(t)[1] is not None}
    for key, operation in served().items():
        if operation.get("x-approval-action"):
            assert key in called_gated or key in ROUTE_EXEMPT, key
