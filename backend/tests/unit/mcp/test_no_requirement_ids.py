"""Agent-facing text says what a tool does in words, never by requirement id (FR-017.22, .23).

Reads the LIVE registry (``build_server`` + ``list_tools()``): every tool description, every
parameter description in each input schema (recursively), the server instructions and every
``forge_howto`` text. An agent cannot look up ``R-02.53`` or ``D52``; it needs the words.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator
from typing import Any

from mcp_server.config import MCPSettings
from mcp_server.server import build_server
from mcp_server.tools.core import HOWTO

ID = re.compile(r"\bR-\d{2}\.\d+|\bFR-\d{3}\.\d+|\bMR-\d+\b|\bD\d{1,3}\b|\bF\d{1,2}'s\b")


def _descriptions(schema: Any, where: str) -> Iterator[tuple[str, str]]:
    if isinstance(schema, dict):
        text = schema.get("description")
        if isinstance(text, str):
            yield where, text
        for key, value in schema.items():
            yield from _descriptions(value, f"{where}.{key}")
    elif isinstance(schema, list):
        for i, value in enumerate(schema):
            yield from _descriptions(value, f"{where}[{i}]")


def _texts() -> list[tuple[str, str]]:
    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    texts: list[tuple[str, str]] = [("instructions", mcp.instructions or "")]
    for tool in asyncio.run(mcp.list_tools()):
        texts.append((tool.name, tool.description or ""))
        texts.extend(_descriptions(tool.input_schema, f"{tool.name}.inputSchema"))
    texts.extend((f"howto:{topic}", text) for topic, text in HOWTO.items())
    return texts


def test_no_agent_facing_text_cites_an_id() -> None:
    texts = _texts()
    tool_descriptions = [w for w, _ in texts if w.startswith("forge_") and "." not in w]
    assert len(tool_descriptions) >= 80  # it read the registry, not an empty one
    cited = {where: ID.findall(text) for where, text in texts if ID.search(text)}
    assert not cited, cited


def test_submit_and_define_say_which_to_use() -> None:
    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    tools = {t.name: t.description or "" for t in asyncio.run(mcp.list_tools())}
    assert "forge_define_run" in tools["forge_submit_run"]
    assert "run_definition_missing" in tools["forge_submit_run"]
    assert "forge_validate_run" in tools["forge_define_run"]


def test_the_hub_tools_say_an_agents_fetches_are_anonymous() -> None:
    """FR-017.29: the agent learns a private repository will fail, and why."""
    from mcp_server.tools.core import HOWTO

    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    tools = {t.name: t.description or "" for t in asyncio.run(mcp.list_tools())}
    for name in (
        "forge_check_hub_revision",
        "forge_define_run",
        "forge_submit_run",
        "forge_create_chain",
    ):
        assert "anonymous" in tools[name], name
    assert "anonymous" in HOWTO["gpu"]


def test_the_guidance_names_the_result_language() -> None:
    """FR-017.24: the instructions and forge_howto('results') explain approval, next_steps and
    the error layout; the instructions stay within 4,096 characters."""
    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    text = mcp.instructions or ""
    assert len(text) <= 4096, len(text)
    for word in ("approval", "server_runs", "next_steps", "fix:", "next:"):
        assert word in text, word
    result = asyncio.run(mcp.call_tool("forge_howto", {"topic": "results"}))
    body = result.content[0].text
    for word in ("approval", "next_steps", "fix:", "next:"):
        assert word in body, word
