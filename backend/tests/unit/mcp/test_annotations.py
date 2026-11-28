"""Every live tool carries a title and both hints, from one table (FR-017.12, .17; 017-D3, D8).

Read through ``build_server`` and ``list_tools()`` (the call an MCP client makes), so deleting the
``apply_annotations`` line in ``build_server`` turns these red; a tool with no row in
``TOOL_CLASS`` makes ``build_server`` refuse to start.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mcp_server import annotations
from mcp_server.config import MCPSettings
from mcp_server.server import build_server
from tests.support.mcp_ledger import EXPECTED_CALLS, LEDGER_TOOL_TOTAL


def _live() -> list[Any]:
    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    return asyncio.run(mcp.list_tools())


def test_every_live_tool_has_a_unique_title_and_both_hints() -> None:
    tools = _live()
    assert len(tools) >= LEDGER_TOOL_TOTAL  # read the whole registry, not an empty one
    titles = [t.title for t in tools]
    assert all(isinstance(t, str) and t.strip() for t in titles), titles
    assert len(set(titles)) == len(titles), sorted(titles)
    for tool in tools:
        assert tool.annotations is not None, tool.name
        assert tool.annotations.read_only_hint is not None, tool.name
        assert tool.annotations.destructive_hint is not None, tool.name
        # A tool is never both read-only and destructive.
        assert not (tool.annotations.read_only_hint and tool.annotations.destructive_hint)
        # 017-D8: no rule for these two in the suite (A13), so they stay unset.
        assert tool.annotations.idempotent_hint is None, tool.name
        assert tool.annotations.open_world_hint is None, tool.name


def test_the_hints_on_the_wire_are_the_tables() -> None:
    """The JSON a client receives carries the table's hints, by alias."""
    tools = {t.name: t for t in _live()}
    for name, kind in annotations.TOOL_CLASS.items():
        wire = tools[name].model_dump(by_alias=True, exclude_none=True)
        assert wire["annotations"] == {
            "readOnlyHint": kind == "read",
            "destructiveHint": kind == "destructive",
        }, name
        assert wire["title"] == annotations.title_for(name)


def test_titles_spell_the_acronyms() -> None:
    tools = {t.name: t.title for t in _live()}
    assert tools["forge_list_gpus"] == "List GPUs"
    assert tools["forge_midataworks_status"] == "miDataworks status"
    assert tools["forge_check_hub_revision"] == "Check Hub revision"
    assert tools["forge_howto"] == "How-to guide"
    assert tools["forge_get_run"] == "Get run"


def test_a_tool_without_a_row_refuses_the_start(monkeypatch: pytest.MonkeyPatch) -> None:
    table = dict(annotations.TOOL_CLASS)
    del table["forge_list_runs"]
    monkeypatch.setattr(annotations, "TOOL_CLASS", table)
    with pytest.raises(SystemExit) as refused:
        build_server(MCPSettings(auth_token="x" * 32))
    assert "forge_list_runs" in str(refused.value)
    assert "TOOL_CLASS" in str(refused.value)


def test_the_table_has_no_row_for_a_tool_that_does_not_exist() -> None:
    """A stale row would hide a renamed tool's missing classification."""
    live = {t.name for t in _live()}
    assert set(annotations.TOOL_CLASS) == live
    assert set(EXPECTED_CALLS) | {"forge_howto"} == live
