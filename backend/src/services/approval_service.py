# Origin: miDataworks (Onegaishimas/miDataworks) backend/src/services/approval_service.py
# @ a055a38. Mode: adapt (docs/REUSE.md). Kept: the lifecycle, expiry on approve, the digest
# check, `executing` committed before the executor runs, the in-process call of the original
# route function. Changed: sync; the claim is a conditional UPDATE ... WHERE status='pending'
# with its row count checked (atomic on SQLite and PostgreSQL alike, no FOR UPDATE); no
# encrypted secret payload (no gated miForge route accepts a secret, D5); the result stored
# verbatim; refusals through core.errors (012 FTID section 3.2).
"""The approval queue  (F12; FR-012.9, .12 - .14; MR-6 - MR-8).

Lifecycle: ``pending`` -> ``executing`` -> ``executed`` | ``failed``; or ``pending`` ->
``rejected`` | ``expired``.

**Exactly once.** Approving claims the row with ``UPDATE mf_approvals SET status='executing'
WHERE id = :id AND status = 'pending'`` and commits **before** the executor runs. Of two
concurrent approves exactly one sees a row count of 1; the other is refused
``approval_not_pending`` - so the route function is called once, whatever the timing.

**In-process, never an HTTP self-call.** A self-call would have to forge or drop the agent
header and would re-enter the gate. The executor is the original route function, called with
the stored values, a session, and a fresh ``Response``.

**Expiry** is evaluated on read and on approve, so no beat task is needed and an expired row
can never execute.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import sqlalchemy as sa
from fastapi import Response
from fastapi.encoders import jsonable_encoder
from pydantic import TypeAdapter

from core.canonical import sha256_of
from core.config import get_settings
from core.errors import Refusal, refuse
from models.approval import APPROVAL_STATUSES, Approval

log = logging.getLogger(__name__)

HINT = "Waiting for the operator to approve this in miForge's Approvals view."


def now() -> datetime:
    """The current time, aware. Module-level so a test can move the clock."""
    return datetime.now(UTC)


def _aware(moment: datetime) -> datetime:
    """SQLite returns naive datetimes for timezone columns; they were written as UTC."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def new_id() -> str:
    return f"apr_{uuid.uuid4().hex}"


def create_pending(
    session: Any,
    *,
    action: str,
    executor_key: str,
    payload: dict[str, Any],
    requested_by: str,
    summary: dict[str, Any],
) -> Approval:
    """Store an agent's request. Commits, so the 202 the agent receives names a real row."""
    stored = jsonable_encoder(payload)
    approval = Approval(
        id=new_id(),
        action=action,
        executor_key=executor_key,
        summary=jsonable_encoder(summary),
        payload=stored,
        request_digest=sha256_of(stored),
        requested_by=requested_by,
        status="pending",
        created_at=now(),
        expires_at=now() + timedelta(hours=get_settings().approval_ttl_hours),
    )
    session.add(approval)
    session.commit()
    log.info(
        "approval requested",
        extra={
            "code": "approval_requested",
            "approval_id": approval.id,
            "action": action,
            "requested_by": requested_by,
        },
    )
    return approval


def pending_body(approval: Approval) -> dict[str, Any]:
    """The ``202`` body an agent receives (FR-012.9)."""
    return {
        "approval_id": approval.id,
        "status": approval.status,
        "action": approval.action,
        "summary": approval.summary,
        "request_digest": "sha256:" + approval.request_digest,
        "expires_at": _aware(approval.expires_at).isoformat(),
        "hint": HINT,
    }


def as_payload(approval: Approval) -> dict[str, Any]:
    """One approval on the wire. The stored request is included: nothing in it is secret."""
    return {
        "approval_id": approval.id,
        "action": approval.action,
        "status": approval.status,
        "summary": approval.summary,
        "payload": approval.payload,
        "request_digest": "sha256:" + approval.request_digest,
        "requested_by": approval.requested_by,
        "decided_by": approval.decided_by,
        "reason": approval.reason,
        "result": approval.result,
        "error": approval.error,
        "created_at": _aware(approval.created_at).isoformat(),
        "decided_at": (
            None if approval.decided_at is None else _aware(approval.decided_at).isoformat()
        ),
        "expires_at": _aware(approval.expires_at).isoformat(),
    }


def expire_due(session: Any) -> int:
    """Mark every pending approval past its expiry ``expired``. Returns how many."""
    result = session.execute(
        sa.update(Approval)
        .where(Approval.status == "pending", Approval.expires_at <= now())
        .values(status="expired", decided_at=now())
    )
    session.commit()
    return int(getattr(result, "rowcount", 0) or 0)


def listing(session: Any, status: str | None = None, limit: int = 200) -> list[Approval]:
    """Approvals, newest first, after expiring what is due."""
    if status is not None and status not in APPROVAL_STATUSES:
        raise refuse(
            "invalid_filter",
            f"status must be one of {', '.join(APPROVAL_STATUSES)}.",
            status=status,
        )
    expire_due(session)
    query = sa.select(Approval).order_by(Approval.created_at.desc()).limit(limit)
    if status is not None:
        query = query.where(Approval.status == status)
    return list(session.execute(query).scalars())


def get(session: Any, approval_id: str) -> Approval:
    expire_due(session)
    approval: Approval | None = session.get(Approval, approval_id)
    if approval is None:
        raise refuse("approval_not_found", f"No approval {approval_id!r}.", approval_id=approval_id)
    return approval


def _not_pending(approval: Approval) -> Refusal:
    return refuse(
        "approval_not_pending",
        f"Approval {approval.id} is {approval.status}; only a pending approval can be decided.",
        approval_id=approval.id,
        status=approval.status,
    )


def _claim(session: Any, approval_id: str, to_status: str, **values: Any) -> bool:
    """Move a row from ``pending`` to ``to_status`` iff it is still pending. Commits."""
    result = session.execute(
        sa.update(Approval)
        .where(Approval.id == approval_id, Approval.status == "pending")
        .values(status=to_status, **values)
    )
    session.commit()
    return int(getattr(result, "rowcount", 0) or 0) == 1


def approve(session: Any, approval_id: str, decided_by: str = "operator") -> Approval:
    """Run the stored request exactly once and record the outcome (FR-012.12, .14)."""
    from api.agent_origin import EXECUTORS

    approval = session.get(Approval, approval_id)
    if approval is None:
        raise refuse("approval_not_found", f"No approval {approval_id!r}.", approval_id=approval_id)
    if approval.status == "pending" and _aware(approval.expires_at) <= now():
        _claim(session, approval_id, "expired", decided_at=now())
        session.refresh(approval)
        raise refuse(
            "approval_expired",
            f"Approval {approval_id} expired at {_aware(approval.expires_at).isoformat()} and "
            f"will never run. The agent must ask again.",
            approval_id=approval_id,
        )
    if approval.status != "pending":
        raise _not_pending(approval)
    executor = EXECUTORS.get(approval.executor_key)
    if executor is None:
        raise refuse(
            "approval_executor_missing",
            f"Approval {approval_id} names {approval.executor_key!r}, which this server no "
            f"longer serves. Reject it.",
            approval_id=approval_id,
        )
    if sha256_of(approval.payload) != approval.request_digest:
        envelope = refuse(
            "approval_tampered",
            "The stored request no longer matches the digest taken when it was made; it will "
            "not run.",
            approval_id=approval_id,
        ).envelope()
        _claim(session, approval_id, "failed", error=envelope["error"], decided_at=now())
        raise refuse(
            "approval_tampered",
            "The stored request no longer matches the digest taken when it was made; it will "
            "not run.",
            approval_id=approval_id,
        )

    payload = dict(approval.payload)
    # ---- THE CLAIM. Exactly one caller gets past this line for a given row. --------------
    if not _claim(session, approval_id, "executing", decided_by=decided_by, decided_at=now()):
        session.refresh(approval)
        raise _not_pending(approval)

    try:
        kwargs: dict[str, Any] = {
            name: TypeAdapter(annotation).validate_python(payload.get(name))
            for name, annotation in executor.value_params.items()
        }
        if executor.session_param is not None:
            kwargs[executor.session_param] = session
        if executor.response_param is not None:
            kwargs[executor.response_param] = Response()
        # AS THE REQUESTER (013 G2): the stored request runs with the agent's identity and this
        # approval's id in force, so what it records names who asked and who approved it.
        from api.agent_origin import Actor, executing_as

        with executing_as(
            Actor(origin="agent", agent=approval.requested_by, approval_id=approval.id)
        ):
            outcome = executor.fn(**kwargs)
        session.commit()
    except Refusal as refusal:
        session.rollback()
        return _finish(session, approval_id, status="failed", error=refusal.envelope()["error"])
    except Exception:
        session.rollback()
        log.exception(
            "an approved request failed",
            extra={"code": "approval_execution_failed", "approval_id": approval_id},
        )
        return _finish(
            session,
            approval_id,
            status="failed",
            error={
                "code": "execution_failed",
                "message": "The approved request failed unexpectedly. Check the server logs "
                "for the correlated entry.",
                "details": {},
            },
        )
    return _finish(session, approval_id, status="executed", result=_result_of(outcome))


def _result_of(outcome: Any) -> Any:
    if isinstance(outcome, Response):
        return {"status_code": outcome.status_code}
    encoded = jsonable_encoder(outcome)
    return {"value": None} if encoded is None else encoded


def _finish(session: Any, approval_id: str, **values: Any) -> Approval:
    session.execute(
        sa.update(Approval)
        .where(Approval.id == approval_id, Approval.status == "executing")
        .values(**values)
    )
    session.commit()
    approval = session.get(Approval, approval_id)
    session.refresh(approval)
    log.info(
        "approval decided",
        extra={"code": "approval_" + str(values.get("status")), "approval_id": approval_id},
    )
    return approval  # type: ignore[no-any-return]


def reject(
    session: Any, approval_id: str, decided_by: str = "operator", reason: str | None = None
) -> Approval:
    """Reject a pending approval. It will never run."""
    approval = get(session, approval_id)
    if approval.status != "pending":
        raise _not_pending(approval)
    if not _claim(
        session, approval_id, "rejected", decided_by=decided_by, decided_at=now(), reason=reason
    ):
        session.refresh(approval)
        raise _not_pending(approval)
    session.refresh(approval)
    return approval
