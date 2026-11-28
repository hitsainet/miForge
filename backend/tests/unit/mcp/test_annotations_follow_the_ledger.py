"""The hints follow the route ledger's approval classes (FR-017.13, .15, .16; 017-D4, D5, D26).

The expected kind of each tool is DERIVED here from ``tests/support/mcp_ledger.py``
``EXPECTED_CALLS`` (whose class column ``test_mcp_parity.py`` pins to what the live gate decides),
never read from ``mcp_server/annotations.py``. The named exceptions are spelled in this file and
each has its own assertion, so moving one in the module's own sets cannot carry the test along.

Rules (suite CHAIN_PLAN A13; 017 FTDD section 4):
- ``read`` / ``plan`` -> read, unless the tool calls an inference provider (017-D4) or fills the
  Hub cache (017-D26) -> write;
- ``delete`` / ``config`` / ``gpu_or_attach`` / ``attach`` -> destructive, except
  ``forge_submit_run`` (its run cannot train) -> write;
- ``write`` / ``gpu`` -> write, except the cancelling tools -> destructive (A13: cancels);
- ``forge_howto`` (no call) -> read.
"""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest

from mcp_server import annotations
from mcp_server.config import MCPSettings
from mcp_server.server import build_server
from tests.support.mcp_ledger import EXPECTED_CALLS

PROVIDER_CALLING = {
    "forge_check_provider",
    "forge_evaluate_completions",
    "forge_score_sample",
    "forge_dry_run_draft",
}
HUB_FETCHING = {"forge_validate_run", "forge_estimate_budget"}
CANCELLING = {"forge_cancel_run", "forge_stop_chain"}
SUBMIT_CANNOT_TRAIN = "forge_submit_run"

#: 016's WSP ledger classes map by the same rule when they exist (FR-017.16): a gated class is
#: never read-only. Unknown classes fail loudly rather than defaulting.
CLASS_KIND = {
    "read": "read",
    "plan": "read",
    "write": "write",
    "gpu": "write",
    "delete": "destructive",
    "config": "destructive",
    "gpu_or_attach": "destructive",
    "attach": "destructive",
    "wsp": "write",
    "wsp_external": "write",  # 018: the protocol test, gated only for an external endpoint
    "wsp_endpoint": "write",  # 2026-11-20: a local endpoint proposal; registers, deletes nothing
}


def expected_kind(tool: str) -> str:
    if tool == "forge_howto":
        return "read"
    ledger_class = EXPECTED_CALLS[tool][4]
    kind = CLASS_KIND[ledger_class]
    if tool == SUBMIT_CANNOT_TRAIN:
        return "write"
    if tool in CANCELLING:
        return "destructive"
    if kind == "read" and tool in PROVIDER_CALLING | HUB_FETCHING:
        return "write"
    return kind


def _live_kinds() -> dict[str, str]:
    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    out: dict[str, str] = {}
    for tool in asyncio.run(mcp.list_tools()):
        assert tool.annotations is not None, tool.name
        if tool.annotations.destructive_hint:
            out[tool.name] = "destructive"
        elif tool.annotations.read_only_hint:
            out[tool.name] = "read"
        else:
            out[tool.name] = "write"
    return out


def test_every_live_tools_hint_is_the_ledgers_rule() -> None:
    live = _live_kinds()
    assert len(live) >= 90
    wrong = {t: (k, expected_kind(t)) for t, k in live.items() if k != expected_kind(t)}
    assert not wrong, f"tool: (live, ledger rule) {wrong}"


def test_the_table_is_the_ledgers_rule() -> None:
    wrong = {
        t: (k, expected_kind(t)) for t, k in annotations.TOOL_CLASS.items() if k != expected_kind(t)
    }
    assert not wrong, f"tool: (table, ledger rule) {wrong}"


def test_the_counts_of_2026_11_15() -> None:
    """60 read-only, 21 writes, 9 destructive over the base tools (FR-017.15)."""
    base = {t: k for t, k in _live_kinds().items() if not t.startswith("forge_wsp_")}
    assert Counter(base.values()) == {"read": 60, "write": 21, "destructive": 9}


def test_submit_run_is_a_write_not_destructive() -> None:
    """Its ledger class is gpu_or_attach, but its run cannot train (017-D5)."""
    assert EXPECTED_CALLS[SUBMIT_CANNOT_TRAIN][4] == "gpu_or_attach"
    assert _live_kinds()[SUBMIT_CANNOT_TRAIN] == "write"


@pytest.mark.parametrize("tool", sorted(CANCELLING))
def test_the_cancelling_tools_are_destructive(tool: str) -> None:
    """Ledger class write (stopping is never gated), destructive under A13 (cancels)."""
    assert EXPECTED_CALLS[tool][4] == "write"
    assert _live_kinds()[tool] == "destructive"


@pytest.mark.parametrize("tool", sorted(HUB_FETCHING))
def test_the_hub_fetching_pair_is_a_write(tool: str) -> None:
    """Ledger class plan, but a Hub source fills the Hub cache (017-D26)."""
    assert EXPECTED_CALLS[tool][4] == "plan"
    assert _live_kinds()[tool] == "write"


@pytest.mark.parametrize("tool", sorted(PROVIDER_CALLING))
def test_the_provider_calling_tools_are_writes(tool: str) -> None:
    assert EXPECTED_CALLS[tool][4] == "plan"
    assert _live_kinds()[tool] == "write"


def test_the_modules_exception_sets_are_this_files() -> None:
    """The module documents the same exceptions this test enforces."""
    assert set(annotations.PROVIDER_CALLING) == PROVIDER_CALLING
    assert set(annotations.HUB_FETCHING) == HUB_FETCHING


def test_check_hub_revision_stays_read() -> None:
    """It reads metadata only (repo_info, list_repo_files); nothing enters the Hub cache."""
    assert _live_kinds()["forge_check_hub_revision"] == "read"


def test_every_gated_tool_is_never_read_only() -> None:
    """A13: a tool that files an approval is never read-only."""
    from mcp_server.context import GATED_TOOLS

    live = _live_kinds()
    assert GATED_TOOLS
    assert all(live[t] != "read" for t in GATED_TOOLS)
