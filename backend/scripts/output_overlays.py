"""The overlays of the MCP output schemas (021 FR-021.18; FT6, FTID §7.2).

A tool's output schema is its ledger route's answer in the published contract
(``generate_output_schemas.output_transform``). An overlay says where a tool's answer is NOT that
route's answer, and how:

- ``withheld``: keys the tool removes at a JSON pointer of the answer (the WSP endpoint listing
  drops the five keys of ``mcp_server.tools.wsp.WITHHELD``, which is imported, so the two lists are
  one: FPRD A4);
- ``withheld_anywhere``: keys the tool's SCHEMA names nowhere (FTDD §8): the WSP run reads remove
  them from their answers at any depth (``tools/wsp.py:withheld_removed``); the approval reads
  carry them only as ``[redacted]``, and the open schema need not name them;
- ``variant``: the statuses of the route's answer the tool can return (the two continue tools share
  one route; the plan tool never submits, the continue tool always does);
- ``hand``: the schema of a tool that makes no backend call (``forge_howto``).

``tests/unit/mcp/test_output_overlays.py`` holds this table equal to the tools whose answer differs
from the route's (a sentinel answer through every live tool), the tools that share a route, and the
tools whose projected schema, without an overlay, would name a withheld key (FTDD §8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp_server.tools.wsp import RUN_READ_WITHHELD, WITHHELD


@dataclass(frozen=True)
class Overlay:
    withheld: dict[str, tuple[str, ...]] | None = None
    withheld_anywhere: tuple[str, ...] | None = None
    variant: tuple[str, ...] | None = None
    hand: dict[str, Any] | None = None


#: ``forge_howto``'s two answers (``mcp_server/tools/core.py``): the topics with the overview, or
#: one topic's text.
HOWTO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "anyOf": [
        {
            "type": "object",
            "description": "No topic: the topics and the overview.",
            "properties": {
                "topics": {"type": "array", "items": {"type": "string"}},
                "overview": {"type": "string"},
            },
            "required": ["topics", "overview"],
        },
        {
            "type": "object",
            "description": "One topic's guidance.",
            "properties": {"topic": {"type": "string"}, "text": {"type": "string"}},
            "required": ["topic", "text"],
        },
    ],
}

OVERLAYS: dict[str, Overlay] = {
    "forge_wsp_list_endpoints": Overlay(withheld={"/properties/endpoints/items": WITHHELD}),
    "forge_plan_continue": Overlay(variant=("200",)),
    "forge_continue_run": Overlay(variant=("202",)),
    "forge_howto": Overlay(hand=HOWTO_SCHEMA),
    # The WSP run reads drop the prices and key names at any depth (tools/wsp.py:withheld_removed).
    "forge_wsp_get_run": Overlay(withheld_anywhere=RUN_READ_WITHHELD),
    "forge_wsp_get_run_ledger": Overlay(withheld_anywhere=RUN_READ_WITHHELD),
    "forge_wsp_get_transcript": Overlay(withheld_anywhere=RUN_READ_WITHHELD),
    # No tool schema names a withheld key (FTDD §8): the approval reads carry them only redacted,
    # and the register tool's own 201 (a row with prices) is never an agent's answer.
    "forge_get_approval_status": Overlay(withheld_anywhere=RUN_READ_WITHHELD),
    "forge_list_approvals": Overlay(withheld_anywhere=RUN_READ_WITHHELD),
    "forge_wsp_register_endpoint": Overlay(withheld_anywhere=RUN_READ_WITHHELD),
}
