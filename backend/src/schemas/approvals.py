"""The typed approval answer (017-D14, FR-017.30).

``ApprovalView`` is the ``response_model`` of the approval reads and decisions, so OpenAPI is
typed and the MCP output schema of ``forge_get_approval_status`` / ``forge_list_approvals`` is
generated from it. **It strips nothing:** FastAPI filters a response through its model, so the
model allows extra keys and declares every key ``services.approval_service.as_payload`` emits
with the JSON type it already emits (the times are ISO strings there; ``payload``, ``summary``
and ``result`` are free-form objects; ``error`` is whatever the executor stored).
``tests/integration/test_approval_view_strips_nothing.py`` compares the JSON key for key.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ApprovalObject(BaseModel):
    """The suite's ``approval`` object (``mistudio.result-language/v1`` ``$defs.approval``)."""

    model_config = ConfigDict(extra="allow")

    id: str
    status: str
    decide_in: str
    then: str
    summary: str
    action: str | None = None
    expires_at: str | None = None
    reason: str | None = None


class ApprovalView(BaseModel):
    """One approval on the wire: ``as_payload``'s keys, typed, plus ``approval``."""

    model_config = ConfigDict(extra="allow")

    approval_id: str
    status: str
    approval: ApprovalObject
    action: str | None = None
    summary: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    request_digest: str | None = None
    requested_by: str | None = None
    decided_by: str | None = None
    reason: str | None = None
    batch_id: str | None = None
    result: Any = None
    error: Any = None
    created_at: str | None = None
    decided_at: str | None = None
    expires_at: str | None = None


class ApprovalList(BaseModel):
    """``GET /approvals``."""

    model_config = ConfigDict(extra="allow")

    approvals: list[ApprovalView]
    count: int
    pending: int
    limit: int


__all__ = ["ApprovalList", "ApprovalObject", "ApprovalView"]
