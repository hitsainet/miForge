"""The approval queue's routes  (F12; FR-012.12 - .16; MR-7).

Six routes. The three reads are tools an agent calls to follow its request
(``forge_list_approvals``, ``forge_list_approval_actions``, ``forge_get_approval_status``).
**Approve and reject are the operator's alone**: an agent call is refused
``agent_cannot_decide`` (403) before the row is touched, so they are exempt from the MCP
inventory with that reason. **Approve-batch** (ENH-006 D79) is the operator's too: it approves
several waiting settings changes with one action, each through the single-approve path.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query
from pydantic import BaseModel, ConfigDict, Field

from api.agent_origin import APPROVAL_ACTIONS, Actor, get_actor, refuse_agent
from api.deps import SessionDep
from schemas.answers_approvals import (
    ApprovalActionListActionsAnswer,
    ApprovalListDoc,
    ApprovalViewDoc,
    ApproveBatchApproveBatchAnswer,
)
from schemas.approvals import ApprovalList, ApprovalView
from services import approval_service
from services.next_step import next_for

router = APIRouter(prefix="/approvals", tags=["approvals"])


class ApproveBatch(BaseModel):
    """``{"approval_ids": [...]}``: the waiting settings changes to approve together."""

    model_config = ConfigDict(extra="forbid")

    approval_ids: list[Annotated[str, Field(min_length=1, max_length=40)]] = Field(
        min_length=1, max_length=approval_service.BATCH_MAX
    )


@router.get(
    "",
    summary="Approval requests, newest first",
    response_model=ApprovalList,
    responses={200: {"model": ApprovalListDoc}},
)
def list_approvals(
    session: SessionDep,
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


@router.get(
    "/actions",
    summary="The four approval actions and what each means",
    responses={200: {"model": ApprovalActionListActionsAnswer}},
)
def list_actions() -> dict[str, Any]:
    """The live action set (FR-012.16): exactly the actions the gate can store."""
    return {"actions": dict(APPROVAL_ACTIONS), "count": len(APPROVAL_ACTIONS)}


@router.get(
    "/{approval_id}",
    summary="One approval and its outcome",
    response_model=ApprovalView,
    responses={200: {"model": ApprovalViewDoc}},
)
def get_approval(approval_id: str, session: SessionDep) -> dict[str, Any]:
    """Poll this after a tool returned a pending approval."""
    body = approval_service.as_payload(approval_service.get(session, approval_id))
    result = body.get("result")
    if body["status"] == "executed" and isinstance(result, dict) and result.get("run_id"):
        # 017 FR-017.7, point 6: the approved request started a run; follow it.
        body["next_steps"] = [
            next_for(
                "forge_get_run",
                {"run_id": str(result["run_id"])},
                "the approved request started this run; read its status",
            )
        ]
    return body


@router.post(
    "/{approval_id}/approve",
    summary="Approve: run the stored request exactly once",
    response_model=ApprovalView,
    responses={200: {"model": ApprovalViewDoc}},
)
def approve(
    approval_id: str,
    session: SessionDep,
    actor: Annotated[Actor, Depends(get_actor)],
) -> dict[str, Any]:
    """The operator's decision. ``409 approval_not_pending`` unless the row is pending."""
    refuse_agent(actor)
    return approval_service.as_payload(approval_service.approve(session, approval_id))


@router.post(
    "/approve-batch",
    summary="Approve every named waiting settings change with one operator action",
    responses={200: {"model": ApproveBatchApproveBatchAnswer}},
)
def approve_batch(
    body: ApproveBatch,
    session: SessionDep,
    actor: Annotated[Actor, Depends(get_actor)],
) -> dict[str, Any]:
    """The operator's decision on several settings changes (ENH-006 D79).

    ``409 approval_batch_refused`` naming every id that is not a waiting settings change, and
    nothing is approved. Otherwise each is approved through the single-approve path and
    recorded on its own row with the shared ``batch_id``; one failing does not stop the rest.
    Answers ``{"batch_id", "count", "results"}``, one result per id in the order given.
    """
    refuse_agent(actor)
    return approval_service.approve_batch(session, list(body.approval_ids))


@router.post(
    "/{approval_id}/reject",
    summary="Reject: the stored request never runs",
    response_model=ApprovalView,
    responses={200: {"model": ApprovalViewDoc}},
)
def reject(
    approval_id: str,
    session: SessionDep,
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
