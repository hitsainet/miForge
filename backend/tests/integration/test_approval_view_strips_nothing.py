"""The typed approval view strips nothing (017 FR-017.30; 017-D14).

``ApprovalView`` is the ``response_model`` of the approval routes, and FastAPI filters a response
through its model: an undeclared key would vanish and a declared one could be coerced. For one
stored row in each status (and an unswept expired ``pending``), the JSON each route answers must
equal ``as_payload(row)`` key for key and value for value.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest

from tests.support.approval_world import World

STATUSES = ("pending", "executing", "executed", "failed", "rejected", "expired", "unswept")


def _row(world: World, status: str) -> str:
    from models.approval import Approval
    from services import approval_service

    now = approval_service.now()
    stored = "pending" if status == "unswept" else status
    row = Approval(
        id=f"apr_{status}",
        action="gpu_start",
        executor_key="api.v1.endpoints.runs_loop.define_run",
        summary={"route": "POST /api/v1/runs/define", "run_id": "run-1", "nested": {"a": [1]}},
        payload={"definition": {"run_id": "run-1"}},
        request_digest="0" * 64,
        requested_by="agent:forge-mcp",
        status=stored,
        # A define's answer, nested (021 P6: an executed result is checked by its action).
        result=(
            {
                "run_id": "run-1",
                "status": "pending",
                "resolved": {"seed": 7, "loop": {"stop_conditions": {"max_steps": {"steps": 2}}}},
            }
            if stored == "executed"
            else None
        ),
        error={"code": "gpu_lease_held", "message": "taken"} if stored == "failed" else None,
        decided_by=None if stored in ("pending", "executing") else "operator",
        reason="not now" if stored == "rejected" else None,
        created_at=now - timedelta(hours=1),
        decided_at=None if stored == "pending" else now,
        expires_at=now - timedelta(minutes=1) if status == "unswept" else now + timedelta(hours=23),
    )
    with world.session() as session:
        session.add(row)
        session.commit()
    return row.id


def _expected(world: World, approval_id: str) -> dict[str, Any]:
    from services import approval_service

    with world.session() as session:
        payload = approval_service.as_payload(approval_service.get(session, approval_id))
    loaded: dict[str, Any] = json.loads(json.dumps(payload))
    return loaded


@pytest.mark.parametrize("status", STATUSES)
def test_the_reads_answer_as_payload_exactly(world: World, status: str) -> None:
    approval_id = _row(world, status)
    one = world.client.get(f"/api/v1/approvals/{approval_id}")
    assert one.status_code == 200, one.text
    expected = _expected(world, approval_id)
    body = one.json()
    # The one-approval read adds next_steps for an executed request that started a run (017
    # FR-017.7, point 6); every as_payload key is still there, unchanged.
    steps = body.pop("next_steps", None)
    assert (steps is not None) == (status == "executed")
    assert body == expected
    listed = {
        row["approval_id"]: row for row in world.client.get("/api/v1/approvals").json()["approvals"]
    }
    assert listed[approval_id] == expected


def test_the_decisions_answer_as_payload_exactly(world: World) -> None:
    rejected_id = _row(world, "pending")
    rejected = world.client.post(f"/api/v1/approvals/{rejected_id}/reject", json={"reason": "no"})
    assert rejected.status_code == 200, rejected.text
    assert rejected.json() == _expected(world, rejected_id)
    assert set(rejected.json()) >= {
        "approval_id",
        "action",
        "status",
        "summary",
        "payload",
        "request_digest",
        "requested_by",
        "decided_by",
        "reason",
        "batch_id",
        "result",
        "error",
        "created_at",
        "decided_at",
        "expires_at",
        "approval",
    }
