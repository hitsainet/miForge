"""G2 (FR-012.6): each tool sends exactly its documented call - method, path, payload, ONCE.

Every module of every category is registered (the harness iterates ``CATEGORY_MODULES``), each
tool is called through the SDK's own argument validation with the ledger's arguments, and the
recorded calls must EQUAL ``[(method, path, payload)]``: a second call, a missing call, a
different path or a different body each fail.
"""

from __future__ import annotations

import pytest

from tests.support.mcp_harness import build_harness, call_tool
from tests.support.mcp_ledger import CALLER_ASSERTION_EXEMPT, EXPECTED_CALLS


@pytest.mark.parametrize("tool", sorted(EXPECTED_CALLS))
def test_the_tool_sends_its_documented_call_once(tool: str) -> None:
    method, path, arguments, recorded, _class = EXPECTED_CALLS[tool]
    mcp, client = build_harness()
    call_tool(mcp, tool, arguments)
    assert client.calls == [(method, path, recorded)], client.calls


@pytest.mark.parametrize("tool", sorted(CALLER_ASSERTION_EXEMPT))
def test_an_exempt_tool_makes_no_call(tool: str) -> None:
    mcp, client = build_harness()
    call_tool(mcp, tool, {})
    assert client.calls == []


def test_the_harness_registered_every_ledger_tool() -> None:
    mcp, _ = build_harness()
    names = {tool.name for tool in mcp._tool_manager.list_tools()}
    assert names, "the harness registered nothing - the scan would assert nothing"
    assert set(EXPECTED_CALLS) | set(CALLER_ASSERTION_EXEMPT) == names
