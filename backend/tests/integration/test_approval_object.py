"""The suite's ``approval`` object on every pending answer and read (017 FR-017.3 - .5, .31).

Driven through the production gate and approval routes (``tests/support/approval_world.py``).
Every shape is validated against the VENDORED contract (``tests/support/result_language.py``),
and every tool named in ``next_steps`` is looked up in the LIVE MCP registry.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest

from tests.support.approval_world import AGENT, World, definition
from tests.support.result_language import validate

LEGACY_PENDING_KEYS = {
    "approval_id",
    "status",
    "action",
    "summary",
    "request_digest",
    "expires_at",
    "hint",
}


def _live_tools() -> set[str]:
    from mcp_server.config import MCPSettings
    from mcp_server.server import build_server

    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    return {t.name for t in asyncio.run(mcp.list_tools())}


def _file(world: World, action: str) -> Any:
    if action == "gpu_start":
        return world.client.post("/api/v1/runs/define", json=definition("run-g"), headers=AGENT)
    if action == "hotupdate_attach":
        return world.client.post(
            "/api/v1/runs/define", json=definition("run-a", mode="attach"), headers=AGENT
        )
    if action == "delete":
        created = world.client.post("/api/v1/pack-drafts", json={"name": "legal-draft"})
        assert created.status_code in (200, 201), created.text
        return world.client.delete(
            f"/api/v1/pack-drafts/{created.json()['draft_id']}", headers=AGENT
        )
    assert action == "config_write"
    return world.client.put(
        "/api/v1/settings/log_level", json={"value": "DEBUG", "reason": "test"}, headers=AGENT
    )


@pytest.mark.parametrize("action", ["gpu_start", "hotupdate_attach", "delete", "config_write"])
def test_an_agents_202_carries_the_approval_and_keeps_every_legacy_key(
    world: World, action: str
) -> None:
    response = _file(world, action)
    assert response.status_code == 202, response.text
    body = response.json()
    assert LEGACY_PENDING_KEYS <= set(body), set(body)
    assert body["action"] == action
    approval = body["approval"]
    validate("approval", approval)
    assert approval == {
        "id": body["approval_id"],
        "status": "pending",
        "decide_in": "miforge",
        "then": "server_runs",
        "summary": approval["summary"],
        "action": action,
        "expires_at": body["expires_at"],
        "reason": None,
    }
    # FR-017.4: poll, with the id, by a tool that exists.
    assert body["next_steps"] == [
        {
            "tool": "forge_get_approval_status",
            "args": {"approval_id": body["approval_id"]},
            "why": body["next_steps"][0]["why"],
        }
    ]
    for step in body["next_steps"]:
        validate("next_step", step)
    assert body["next_steps"][0]["tool"] in _live_tools()
    validate("result_fields", body)
    assert "next" not in body


def test_every_approval_read_carries_the_approval(world: World) -> None:
    filed = _file(world, "gpu_start").json()
    one = world.client.get(f"/api/v1/approvals/{filed['approval_id']}").json()
    validate("approval", one["approval"])
    listed = world.client.get("/api/v1/approvals").json()["approvals"]
    assert listed and all(validate("approval", row["approval"]) is None for row in listed)
    approved = world.client.post(f"/api/v1/approvals/{filed['approval_id']}/approve").json()
    assert approved["approval"]["status"] == approved["status"] == "executed"
    validate("approval", approved["approval"])
    rejected_id = _file(world, "config_write").json()["approval_id"]
    rejected = world.client.post(
        f"/api/v1/approvals/{rejected_id}/reject", json={"reason": "not now"}
    ).json()
    assert rejected["approval"]["status"] == "rejected"
    assert rejected["approval"]["reason"] == "not now"
    validate("approval", rejected["approval"])


def test_every_status_the_queue_writes_maps_to_itself() -> None:
    from models.approval import APPROVAL_STATUSES
    from services.result_language import STATUS_MAP
    from tests.support.result_language import status_enum

    assert set(STATUS_MAP) == set(APPROVAL_STATUSES)
    for status in APPROVAL_STATUSES:
        assert STATUS_MAP[status] == status
        assert status in status_enum()
    assert "approved" not in STATUS_MAP.values()


def test_the_view_of_an_unswept_expired_row_is_expired() -> None:
    """017-D10: a pending row past its expiry that no read or sweep has marked yet."""
    from types import SimpleNamespace

    from services import approval_service
    from services.result_language import approval_view

    now = approval_service.now()
    row = SimpleNamespace(
        id="apr_x",
        status="pending",
        action="gpu_start",
        summary={"run_id": "r"},
        expires_at=now - timedelta(seconds=1),
        reason=None,
        error=None,
    )
    view = approval_view(row, now)
    assert view["status"] == "expired"
    validate("approval", view)
    assert approval_view(row, now - timedelta(minutes=1))["status"] == "pending"


def test_an_unswept_expired_pending_row_reads_expired(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    from services import approval_service

    filed = _file(world, "gpu_start").json()
    later = approval_service.now() + timedelta(days=2)
    monkeypatch.setattr(approval_service, "now", lambda: later)
    body = world.client.get(f"/api/v1/approvals/{filed['approval_id']}").json()
    assert body["approval"]["status"] == "expired"
    validate("approval", body["approval"])


def test_server_executed_one_call_then_status_reads(world: World) -> None:
    """FR-017.31 (A18): ``then`` is ``server_runs``, never ``call_again_with``; an invented
    re-call header is not one: a second identical filing is a second pending request."""
    first = _file(world, "gpu_start").json()
    assert first["approval"]["then"] == "server_runs"
    assert "call_again_with" not in first["approval"]
    second = world.client.post(
        "/api/v1/runs/define",
        json=definition("run-g"),
        headers={**AGENT, "X-Miforge-Approval-Request": first["approval_id"]},
    )
    assert second.status_code == 202, second.text
    assert second.json()["approval_id"] != first["approval_id"]
    still = world.client.get(f"/api/v1/approvals/{first['approval_id']}").json()
    assert still["status"] == "pending"
    for row in world.client.get("/api/v1/approvals").json()["approvals"]:
        assert row["approval"]["then"] == "server_runs"
        assert "call_again_with" not in row["approval"]
