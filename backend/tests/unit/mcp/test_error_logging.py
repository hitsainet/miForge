"""Converting a tool failure keeps the SDK's log line (017 FR-017.32; A30).

The SDK's handler logs every failure before answering; ``ForgeMCPServer.call_tool`` returns a
result instead of raising, so it must log first, the same three ways: a crash with its traceback,
an argument error by field names only, any other failure by its text.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import pytest

from mcp_server.config import MCPSettings
from mcp_server.health_gate import HealthGate
from mcp_server.server import build_server

SDK_LOGGER = "mcp.server.mcpserver.server"


@pytest.fixture(autouse=True)
def _open_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unavailable(self: Any, *_: str) -> None:
        return None

    monkeypatch.setattr(HealthGate, "unavailable", unavailable)


def _mcp(response: httpx.Response) -> Any:
    mcp, client = build_server(MCPSettings(auth_token="x" * 32))
    client._http = httpx.AsyncClient(
        base_url="http://backend/api/v1", transport=httpx.MockTransport(lambda r: response)
    )
    return mcp


def _records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == SDK_LOGGER]


def test_a_crash_logs_one_error_with_its_traceback(caplog: pytest.LogCaptureFixture) -> None:
    mcp = _mcp(httpx.Response(200, json={}))

    @mcp.tool()
    async def forge_crash_stub() -> Any:
        raise RuntimeError("boom")

    with caplog.at_level(logging.INFO, logger=SDK_LOGGER):
        result = asyncio.run(mcp.call_tool("forge_crash_stub", {}))
    assert result.is_error
    errors = [r for r in _records(caplog) if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].exc_info is not None
    assert "RuntimeError" in "".join(
        logging.Formatter().formatException(errors[0].exc_info)  # type: ignore[arg-type]
    )


def test_an_argument_error_logs_field_names_never_the_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mcp = _mcp(httpx.Response(200, json={}))
    with caplog.at_level(logging.INFO, logger=SDK_LOGGER):
        asyncio.run(mcp.call_tool("forge_list_runs", {"limit": "s3cret-value"}))
    records = _records(caplog)
    assert len(records) == 1 and records[0].levelno == logging.INFO
    assert "s3cret-value" not in records[0].getMessage()
    # Exactly the field names: the value is hidden today by the strict model's
    # hide_input_in_errors too, so asserting only its absence would pass against a log line
    # that printed the whole error (a masked guard; control C9.4b).
    assert records[0].getMessage() == "Tool 'forge_list_runs' rejected arguments: ['limit']"


def test_a_refusal_logs_one_info_line(caplog: pytest.LogCaptureFixture) -> None:
    envelope = {"error": {"code": "run_not_found", "message": "No run 'r'.", "details": {}}}
    mcp = _mcp(httpx.Response(404, json=envelope))
    with caplog.at_level(logging.INFO, logger=SDK_LOGGER):
        asyncio.run(mcp.call_tool("forge_get_run", {"run_id": "r"}))
    records = _records(caplog)
    assert len(records) == 1 and records[0].levelno == logging.INFO
    assert "run_not_found" in records[0].getMessage()
