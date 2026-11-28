"""``api.v1.endpoints.approvals``'s routes answer as the contract says (021 FR-021.13).

Every approval here is filed by the real gate (an agent's ``PUT /settings/{key}`` or
``POST /runs/define`` on the approval world), never planted, so each answer carries the
summary, payload and result the gate and its executor actually store.
"""

from __future__ import annotations

from typing import Any

from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import AGENT, World, definition

ROUTER = "api.v1.endpoints.approvals"


def _ask_set(world: World, key: str, value: Any) -> str:
    response = world.client.put(f"/api/v1/settings/{key}", json={"value": value}, headers=AGENT)
    return str(answered("PUT /api/v1/settings/{key}", response, pending=True)["approval_id"])


def _ask_define(world: World, run_id: str) -> str:
    response = world.client.post("/api/v1/runs/define", json=definition(run_id), headers=AGENT)
    assert response.status_code == 202, response.text
    return str(response.json()["approval_id"])


@case(ROUTER, "GET /api/v1/approvals")
def test_list_approvals(world: World) -> None:
    """One waiting settings change and one decided run request: both statuses listed."""
    _ask_set(world, "checkpoint_lifecycle_keep_last_n", 3)
    decided = _ask_define(world, "run-listed")
    assert world.client.post(f"/api/v1/approvals/{decided}/approve").status_code == 200
    body = answered("GET /api/v1/approvals", world.client.get("/api/v1/approvals"))
    assert body["count"] == 2


@case(ROUTER, "GET /api/v1/approvals/actions")
def test_list_actions(world: World) -> None:
    answered("GET /api/v1/approvals/actions", world.client.get("/api/v1/approvals/actions"))


@case(ROUTER, "GET /api/v1/approvals/{approval_id}")
def test_get_approval(world: World) -> None:
    """Pending, then executed with the run it started (the ``next_steps`` branch)."""
    approval_id = _ask_define(world, "run-polled")
    key = "GET /api/v1/approvals/{approval_id}"
    pending = answered(key, world.client.get(f"/api/v1/approvals/{approval_id}"))
    assert pending["status"] == "pending"
    assert world.client.post(f"/api/v1/approvals/{approval_id}/approve").status_code == 200
    executed = answered(key, world.client.get(f"/api/v1/approvals/{approval_id}"))
    assert executed["status"] == "executed" and executed["next_steps"]


@case(ROUTER, "POST /api/v1/approvals/{approval_id}/approve")
def test_approve(world: World) -> None:
    """The operator approves a stored run request: it executes, and the result is the run."""
    approval_id = _ask_define(world, "run-approved")
    body = answered(
        "POST /api/v1/approvals/{approval_id}/approve",
        world.client.post(f"/api/v1/approvals/{approval_id}/approve"),
    )
    assert body["status"] == "executed" and body["result"]["run_id"] == "run-approved"


@case(ROUTER, "POST /api/v1/approvals/approve-batch")
def test_approve_batch(world: World) -> None:
    ids = [
        _ask_set(world, "checkpoint_lifecycle_enabled", True),
        _ask_set(world, "checkpoint_lifecycle_keep_last_n", 3),
    ]
    body = answered(
        "POST /api/v1/approvals/approve-batch",
        world.client.post("/api/v1/approvals/approve-batch", json={"approval_ids": ids}),
    )
    assert body["count"] == 2


@case(ROUTER, "POST /api/v1/approvals/{approval_id}/reject")
def test_reject(world: World) -> None:
    approval_id = _ask_set(world, "checkpoint_lifecycle_keep_last_n", 3)
    body = answered(
        "POST /api/v1/approvals/{approval_id}/reject",
        world.client.post(f"/api/v1/approvals/{approval_id}/reject", json={"reason": "not now"}),
    )
    assert body["status"] == "rejected"


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
