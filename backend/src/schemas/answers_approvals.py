"""The answers of the two approval routes ``schemas/approvals.py`` does not type (feature 021, P3):
``GET /api/v1/approvals/actions`` and ``POST /api/v1/approvals/approve-batch``.

Documentation only (MF9): the routes keep ``response_model=None`` and serve what they build. The
enforced ``schemas/approvals.py`` models are not reused, so declaring these changes none of them.
"""

from pydantic import BaseModel

from .answers_common import (
    FREE_FORM,
    JsonValue,
    NextStep,
    free_form,
    history_form,
    may_be_absent,
)


class ApprovalActionListActionsAnswer(BaseModel):
    """``api/v1/endpoints/approvals.py:list_actions`` (``api/agent_origin.py:APPROVAL_ACTIONS``)."""

    actions: dict[str, str]
    count: int


class ApproveBatchApprovalObject(BaseModel):
    """``services/result_language.py:approval_view``: the suite's ``approval`` object."""

    id: str
    status: str
    decide_in: str
    then: str
    summary: str
    action: str
    expires_at: str
    reason: str | None


class ApproveBatchError(BaseModel):
    """``core/errors.py:Refusal.envelope`` ``error`` (or ``services/approval_service.py``'s
    ``execution_failed``): why a request failed."""

    code: str
    message: str
    details: dict[str, JsonValue] | None = may_be_absent(
        json_schema_extra={FREE_FORM: "the refusal's facts, which each refusal names"},
        description="Absent from a stored error written without it (an older or hand-made row).",
    )


class ApprovalViewDoc(BaseModel):
    """``services/approval_service.py:as_payload`` plus ``approval``: one approval on the wire. The
    documentation twin of ``schemas/approvals.py:ApprovalView`` (021 FT2), which stays the enforced
    model of the approval reads and decisions."""

    approval_id: str
    action: str
    status: str
    summary: dict[str, JsonValue] = free_form(
        "the summary the gate's summariser built for the operator; its keys depend on the action"
    )
    payload: dict[str, JsonValue] = free_form(
        "the stored request (its path, query and body values), as the caller sent it"
    )
    request_digest: str
    requested_by: str
    decided_by: str | None
    reason: str | None
    batch_id: str | None
    result: JsonValue = history_form(
        "What the executed request answered (jsonable-encoded; {status_code} for a route that "
        "answers no body), or null when it did not execute. Checked by `action` in its history "
        "form (the contract's description says why)."
    )
    error: ApproveBatchError | None
    created_at: str
    decided_at: str | None
    expires_at: str
    approval: ApproveBatchApprovalObject
    next_steps: list[NextStep] | None = may_be_absent(
        description="Sent by GET /approvals/{approval_id} while the request still waits."
    )


class ApprovalListDoc(BaseModel):
    """``api/v1/endpoints/approvals.py:list_approvals``: the documentation twin of
    ``schemas/approvals.py:ApprovalList`` (021 FT2)."""

    approvals: list[ApprovalViewDoc]
    count: int
    pending: int
    limit: int


class ApproveBatchResult(ApprovalViewDoc):
    """``services/approval_service.py:approve_batch``: ``as_payload`` of one row, plus ``ok``."""

    ok: bool


class ApproveBatchApproveBatchAnswer(BaseModel):
    """``api/v1/endpoints/approvals.py:approve_batch`` (``services/approval_service.py``)."""

    batch_id: str
    count: int
    results: list[ApproveBatchResult]
