"""``tools/list`` stays under FT7's ceiling, and the descriptions live in the contract (021
FR-021.22; operator, 2026-11-26: FT7 option (b), amending Q-T7).

The MCP output schemas carry no ``description`` key (nor a generated ``title``): the published
answer contract keeps every description, and the tool descriptions and ``forge_howto`` carry the
guidance an agent reads. Measured with every description kept, ``tools/list`` was 463,451 bytes for
95 tools, over the 350,000-byte ceiling; without them, 338,305. After train F3 (100 tools, its run
record typed by reference) it measured 376,787, and the operator raised the ceiling to 400,000 on
2026-11-26 (B15): about 23 KB of headroom, roughly seven more tools at about 3 KB each.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mcp_server.server import OUTPUT_SCHEMAS_PATH

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "backend" / "scripts"))
import measure_tools_list  # noqa: E402 - FTID §13's measuring script

NAMED = ("properties", "$defs", "patternProperties", "dependentSchemas")


def _schema_keys(node: Any, key: str) -> int:
    """How many schema positions at or below ``node`` carry ``key`` (a property NAMED ``key`` is
    not one)."""
    found = 0
    if isinstance(node, dict):
        found += key in node
        for name, value in node.items():
            if name in NAMED and isinstance(value, dict):
                found += sum(_schema_keys(sub, key) for sub in value.values())
            else:
                found += _schema_keys(value, key)
    elif isinstance(node, list):
        found += sum(_schema_keys(item, key) for item in node)
    return found


def test_tools_list_is_under_the_ceiling() -> None:
    # The operator's figure, not the script's own (review round 3): raising CEILING alone is red.
    assert measure_tools_list.CEILING == 400_000, "FT7's ceiling is the operator's (B15)"
    size = measure_tools_list.tools_list_bytes()
    assert size <= measure_tools_list.CEILING, (
        f"tools/list is {size:,} bytes, over FT7's {measure_tools_list.CEILING:,}: the operator "
        "decides (FT7), it is not worked around"
    )


def test_the_output_schemas_carry_no_description_or_title() -> None:
    typed = json.loads(OUTPUT_SCHEMAS_PATH.read_text(encoding="utf-8"))["typed"]
    assert _schema_keys(typed, "description") == 0
    assert _schema_keys(typed, "title") == 0


def test_the_contract_keeps_its_descriptions() -> None:
    contract = json.loads((REPO / "docs" / "schemas" / "miforge-answers-v1.json").read_text())
    assert _schema_keys(contract["$defs"], "description") >= 300
    assert "description" in contract["$defs"]["ApprovalViewDoc"]
