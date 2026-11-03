# Origin (shape): miDataworks (Onegaishimas/miDataworks) backend/tests/support/mcp_harness.py
# @ a055a38 - RecordingClient and the caller harness, adapted (012 FTID section 8).
"""The caller harness for MCP tools: one recording client, EVERY module in the registry.

The harness iterates ``CATEGORY_MODULES`` itself, never a hand-picked subset. The recording
client's method signatures must EQUAL ``ForgeClient``'s (``tests/unit/mcp/test_client.py``
compares them with ``inspect.signature``): a stand-in more forgiving than the real client lets
a tool calling it wrongly pass every test and raise ``TypeError`` in production.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.mcpserver import Context, MCPServer

from mcp_server.config import MCPSettings
from mcp_server.context import ToolContext
from mcp_server.tools import CATEGORY_MODULES

PENDING_BODY = {
    "approval_id": "apr_test",
    "status": "pending",
    "action": "gpu_start",
    "summary": {},
    "request_digest": "sha256:0",
    "expires_at": "2026-10-20T00:00:00+00:00",
}


class RecordingClient:
    """Records ``(method, path, kwargs)`` for every call a tool issues."""

    def __init__(self, response: Any = None) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.response = {"ok": True} if response is None else response

    async def _record(self, method: str, path: str, **kwargs: Any) -> Any:
        self.calls.append((method, path, kwargs))
        return self.response

    async def get(self, path: str, **params: Any) -> Any:
        return await self._record("GET", path, **params)

    async def post(self, path: str, json_body: Any = None, **params: Any) -> Any:
        return await self._record("POST", path, json_body=json_body, **params)

    async def put(self, path: str, json_body: Any) -> Any:
        return await self._record("PUT", path, json_body=json_body)

    async def delete(self, path: str, **params: Any) -> Any:
        return await self._record("DELETE", path, **params)


class FakeGate:
    """A health gate that is open, or closed on one dependency."""

    def __init__(self, closed: str | None = None) -> None:
        self.closed = closed
        self.invalidated = 0

    async def unavailable(self, *dependencies: str) -> dict[str, str] | None:
        if self.closed is not None and self.closed in dependencies:
            return {"unavailable": self.closed, "reason": f"{self.closed} down: test"}
        return None

    def invalidate(self) -> None:
        self.invalidated += 1


def build_harness(
    gate: FakeGate | None = None, response: Any = None
) -> tuple[MCPServer, RecordingClient]:
    """Register EVERY module of every category against one recording client."""
    mcp = MCPServer("harness")
    client = RecordingClient(response)
    ctx = ToolContext(settings=MCPSettings(auth_token="x" * 32), gate=gate or FakeGate())  # type: ignore[arg-type]
    for modules in CATEGORY_MODULES.values():
        for module in modules:
            module.register(mcp, client, ctx)  # type: ignore[arg-type]
    return mcp, client


def call_tool(mcp: MCPServer, name: str, arguments: dict[str, Any]) -> Any:
    """Call a tool through MCPServer's own argument validation and return the raw result."""
    return asyncio.run(mcp._tool_manager.call_tool(name, arguments, Context(mcp_server=mcp)))
