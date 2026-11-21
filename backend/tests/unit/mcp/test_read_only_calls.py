"""A read-only tool's calls are reads (FR-017.14; 017-D12).

Each tool the LIVE registry marks ``readOnlyHint: true`` is replayed through the recording
client with its ledger arguments. Every call it makes must be a ``GET`` or one of the six
``READ_ONLY_ROUTES``, each of which ``tests/integration/test_read_only_routes_write_nothing.py``
proves writes nothing. So a read-only claim rests on the calls the tool really makes, not on its
name or a hand list.
"""

from __future__ import annotations

import asyncio

from mcp_server.config import MCPSettings
from mcp_server.server import build_server
from tests.support.mcp_harness import build_harness, call_tool
from tests.support.mcp_ledger import (
    CALLER_ASSERTION_EXEMPT,
    EXPECTED_CALLS,
    READ_ONLY_ROUTES,
    full,
    template_matches,
)


def _read_only_tools() -> list[str]:
    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    return sorted(
        t.name
        for t in asyncio.run(mcp.list_tools())
        if t.annotations is not None and t.annotations.read_only_hint
    )


def _listed(method: str, path: str) -> set[tuple[str, str]]:
    """The ``READ_ONLY_ROUTES`` entries this concrete call matches."""
    return {
        (m, template)
        for m, template in READ_ONLY_ROUTES
        if method == m and template_matches(template.replace("{}", "{x}"), full(path))
    }


def test_every_read_only_tools_calls_are_reads() -> None:
    tools = _read_only_tools()
    replayed = 0
    offenders: dict[str, list[tuple[str, str]]] = {}
    for tool in tools:
        if tool in CALLER_ASSERTION_EXEMPT:
            continue
        mcp, client = build_harness()
        call_tool(mcp, tool, dict(EXPECTED_CALLS[tool][2]))
        assert client.calls, tool  # it made its call
        replayed += 1
        bad = [
            (method, path)
            for method, path, _ in client.calls
            if method != "GET" and not _listed(method, path)
        ]
        if bad:
            offenders[tool] = bad
    assert replayed >= 58, replayed  # 60 read-only tools, forge_howto makes no call
    assert not offenders, offenders


def test_the_six_routes_are_each_called_by_a_read_only_tool() -> None:
    """A route listed here that no read-only tool calls is a stale exemption."""
    tools = set(_read_only_tools())
    used: set[tuple[str, str]] = set()
    for tool, (method, path, *_rest) in EXPECTED_CALLS.items():
        if tool in tools and method != "GET":
            used |= _listed(method, path)
    assert used == set(READ_ONLY_ROUTES)
    assert len(READ_ONLY_ROUTES) == 6
