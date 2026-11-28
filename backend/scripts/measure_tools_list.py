"""The three sizes FT7 bounds (021 FR-021.22; FTID §13): the answer contract, the generated
``output_schemas.json``, and ``tools/list`` as an MCP client receives it (compact JSON of every
``Tool``). FT7's ceiling for ``tools/list`` is :data:`CEILING` bytes; over it is the operator's
decision, not a workaround.

    PYTHONPATH=backend/src:backend python backend/scripts/measure_tools_list.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
#: FT7's ceiling for ``tools/list``. 350,000 at Q-T7; raised to 400,000 by the operator on
#: 2026-11-26 (about 14:40 ET) when 100 tools measured 376,787 with option (b) applied (B15).
CEILING = 400_000


def tools_list_bytes() -> int:
    """``tools/list`` of the live server, as compact JSON."""
    from mcp_server.config import MCPSettings
    from mcp_server.server import build_server

    mcp, _ = build_server(MCPSettings(auth_token="measure-tools-list"))
    tools = asyncio.run(mcp.list_tools())
    listing = [t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in tools]
    return len(json.dumps({"tools": listing}, separators=(",", ":")).encode("utf-8"))


def sizes() -> dict[str, int]:
    return {
        "contract": (REPO / "docs" / "schemas" / "miforge-answers-v1.json").stat().st_size,
        "output_schemas": (REPO / "backend" / "src" / "mcp_server" / "output_schemas.json")
        .stat()
        .st_size,
        "tools_list": tools_list_bytes(),
    }


if __name__ == "__main__":
    for name, size in sizes().items():
        print(f"{name}: {size:,} bytes")
    print(f"ceiling (tools/list): {CEILING:,} bytes")
