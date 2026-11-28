"""``next_steps`` entries: what to call next, in the suite's result language (017 FR-017.6).

One helper, so every workflow point and refusal builds the same shape
(``mistudio.result-language/v1`` ``$defs.next_step``: ``{tool, args, why}``), and so a name that
is not a miForge MCP tool cannot be emitted. The backend imports no MCP code (ADR-017): the
names are checked against the live MCP registry by ``tests/integration/test_next_points.py``,
and the prefix here refuses a sibling's tool (``dataworks_*``, ``miserver_*``) outright.

miForge has no legacy ``next`` key, so it emits only ``next_steps`` (CHAIN_PLAN A6, amended
2026-11-15).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PREFIX = "forge_"
WHY_LIMIT = 300


def next_for(tool: str, args: Mapping[str, Any] | None, why: str) -> dict[str, Any]:
    """One ``next_step``: ``{"tool", "args", "why"}``.

    Raises:
        ValueError: ``tool`` is not a miForge MCP tool name, or ``why`` is empty (the contract
            requires 1 - 300 characters).
    """
    if not tool.startswith(PREFIX):
        raise ValueError(f"next_steps names a miForge MCP tool (forge_*), not {tool!r}")
    if not why or len(why) > WHY_LIMIT:
        raise ValueError(f"a next step's why is 1 - {WHY_LIMIT} characters")
    return {"tool": tool, "args": dict(args or {}), "why": why}


__all__ = ["next_for"]
