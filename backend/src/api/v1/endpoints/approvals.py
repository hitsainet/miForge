"""The approval queue's routes  (F12; FR-012.12 - .16; MR-7).

Five routes. The three reads are tools an agent calls to follow its request
(``forge_list_approvals``, ``forge_list_approval_actions``, ``forge_get_approval_status``).
**Approve and reject are the operator's alone**: an agent call is refused
``agent_cannot_decide`` (403) before the row is touched, so they are exempt from the MCP
inventory with that reason.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy.orm import Session

from api.agent_origin import APPROVAL_ACTIONS, Actor, get_actor, refuse_agent
from api.deps import session_dependency
from services import approval_service

router = APIRouter(prefix="/approvals", tags=["approvals"])


@router.get("", summary="Approval requests, newest first")
def list_approvals(
    session: Annotated[Session, Depends(session_dependency)],
    status: Annotated[str | None, Query(max_length=16)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> dict[str, Any]:
    """Every approval (or one status), with the summary the operator decides on."""
    rows = approval_service.listing(session, status=status, limit=limit)
    payload = [approval_service.as_payload(row) for row in rows]
    return {
        "approvals": payload,
        "count": len(payload),
        "pending": sum(1 for row in rows if row.status == "pending"),
        "limit": limit,
    }


@router.get("/actions", summary="The four approval actions and what each means")
def list_actions() -> dict[str, Any]:
    """The live action set (FR-012.16): exactly the actions the gate can store."""
    return {"actions": dict(APPROVAL_ACTIONS), "count": len(APPROVAL_ACTIONS)}


@router.get("/{approval_id}", summary="One approval and its outcome")
def get_approval(
    approval_id: str, session: Annotated[Session, Depends(session_dependency)]
) -> dict[str, Any]:
    """Poll this after a tool returned a pending approval."""
    return approval_service.as_payload(approval_service.get(session, approval_id))


@router.post("/{approval_id}/approve", summary="Approve: run the stored request exactly once")
def approve(
    approval_id: str,
    session: Annotated[Session, Depends(session_dependency)],
    actor: Annotated[Actor, Depends(get_actor)],
) -> dict[str, Any]:
    """The operator's decision. ``409 approval_not_pending`` unless the row is pending."""
    refuse_agent(actor)
    return approval_service.as_payload(approval_service.approve(session, approval_id))


@router.post("/{approval_id}/reject", summary="Reject: the stored request never runs")
def reject(
    approval_id: str,
    session: Annotated[Session, Depends(session_dependency)],
    actor: Annotated[Actor, Depends(get_actor)],
    body: Annotated[dict[str, Any] | None, Body()] = None,
) -> dict[str, Any]:
    """The operator's decision, with an optional ``{"reason"}`` the agent will read."""
    refuse_agent(actor)
    reason = None if body is None else body.get("reason")
    return approval_service.as_payload(
        approval_service.reject(
            session, approval_id, reason=None if reason is None else str(reason)[:2000]
        )
    )
