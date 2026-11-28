"""FR-012.23: one structured line per tool call - tool, identity, args digest, status, duration."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from mcp_server.server import AuditToolLogger, wrap_tool_with_audit
from tests.support.mcp_harness import PENDING_BODY, FakeGate, build_harness, call_tool


def _lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "mcp_tool_call" in r.getMessage()]


@pytest.mark.parametrize(
    ("response", "gate", "status"),
    [
        (None, None, "ok"),  # the route's example answer (tests/support/answer_examples.py)
        (PENDING_BODY, None, "pending"),
        (None, "backend", "unavailable"),
    ],
)
def test_one_line_per_call_per_status(
    caplog: pytest.LogCaptureFixture, response: Any, gate: str | None, status: str
) -> None:
    mcp, _ = build_harness(gate=FakeGate(closed=gate), response=response)
    mcp.forge_identity = "agent:forge-mcp"  # type: ignore[attr-defined]
    # 2026-11-20: plus forge_wsp_register_endpoint; 019 P3: plus forge_wsp_get_pool;
    # 2026-11-25: plus forge_wsp_list_endpoints; 2026-11-26: plus the four WSP run tools and
    # forge_wsp_validate_run; 2026-11-27: plus forge_wsp_list_heldout_uses.
    assert wrap_tool_with_audit(mcp) == 101
    with caplog.at_level(logging.INFO, logger="mcp_server.server"):
        call_tool(mcp, "forge_define_run", {"definition": {"run_id": "r"}})
    (line,) = _lines(caplog)
    assert "tool=forge_define_run" in line and "identity=agent:forge-mcp" in line
    assert f"status={status}" in line and "duration_ms=" in line
    assert (
        f"args_digest={AuditToolLogger.digest('forge_define_run', {'definition': {'run_id': 'r'}})}"
        in line
    )


def test_an_error_is_logged_with_its_type(caplog: pytest.LogCaptureFixture) -> None:
    mcp, client = build_harness()

    async def boom(path: str, **params: Any) -> Any:
        raise RuntimeError("x")

    client.get = boom  # type: ignore[method-assign]
    wrap_tool_with_audit(mcp)
    with (
        caplog.at_level(logging.INFO, logger="mcp_server.server"),
        pytest.raises(Exception, match="x"),
    ):
        call_tool(mcp, "forge_get_run", {"run_id": "run-1"})
    (line,) = _lines(caplog)
    assert "status=error:" in line
