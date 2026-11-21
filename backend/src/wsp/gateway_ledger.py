"""The gateway's money and token ledger (feature 016; FTDD §5.4, §5.5; D16, D19; P50, P61, R3A-18).

The gateway keeps no spend in memory that matters: every upstream request **reserves** its worst
case here first and **settles** its metered usage after. Both take the run's row lock, then (for an
external endpoint) the UTC day's lock row, always in that order, so a reserve racing a terminal
settle cannot deadlock (R3A-18). A reservation is closed exactly once: a second settle, or a settle
after the run's terminal transition closed it at worst case, is refused ``wsp_reservation_closed``
and appends nothing.
"""

from __future__ import annotations

import datetime as dt
import hmac
import re
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from core.errors import refuse
from wsp.models import CALL_OUTCOMES, WspReservation, WspRun, WspSpendDay, WspUsage
from wsp.settings import WSPSettings

ACTIVE = ("pushing", "running")
NO_GATEWAY_STYLES = ("program_rerun", "replay")
ZERO = Decimal("0")
#: An identifier the call record may keep (a finish reason, a refusal code): lower-case words,
#: never text. The gateway reads a finish reason from the UPSTREAM's answer, which an editor
#: endpoint controls, so anything else is stored ``other`` (a finish reason) or nothing (a code).
_FINISH = re.compile(r"[a-z][a-z0-9_]{0,31}")
_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
#: The largest body a served call's ``sent_bytes`` records (the gateway's body limit is far lower).
MAX_SENT_BYTES = 1 << 31
#: The longest latency stored: one day, in milliseconds.
MAX_LATENCY_MS = 86_400_000


def call_columns(call: Any) -> dict[str, Any]:
    """What became of one editor call, as ``mf_wsp_usage`` stores it (live progress, 2026-11-20).

    **Counts and codes only.** Every field is checked here, on the backend's side, and a field
    that fails its check is dropped (a finish reason becomes ``other``): the record of spend is
    never refused for an annotation, and no free text ever reaches the table. Keys other than the
    fields below are ignored. ``sent_bytes``, ``prompt_messages`` and ``conversation_sha256``
    (migration 0033) are a served call's prompt evidence: two counts and a digest."""
    c = call if isinstance(call, dict) else {}

    def bounded(value: Any, low: int, high: int) -> int | None:
        if isinstance(value, int) and not isinstance(value, bool) and low <= value <= high:
            return value
        return None

    outcome = c.get("outcome")
    finish = c.get("finish_reason")
    code = c.get("refusal_code")
    digest = c.get("conversation_sha256")
    return {
        "outcome": outcome if isinstance(outcome, str) and outcome in CALL_OUTCOMES else None,
        "http_status": bounded(c.get("http_status"), 100, 599),
        "latency_ms": bounded(c.get("latency_ms"), 0, MAX_LATENCY_MS),
        "finish_reason": (
            None
            if finish is None
            else finish if isinstance(finish, str) and _FINISH.fullmatch(finish) else "other"
        ),
        "refusal_code": code if isinstance(code, str) and _CODE.fullmatch(code) else None,
        "sent_bytes": bounded(c.get("sent_bytes"), 1, MAX_SENT_BYTES),
        "prompt_messages": bounded(c.get("prompt_messages"), 0, MAX_SENT_BYTES),
        "conversation_sha256": (
            digest if isinstance(digest, str) and _SHA256.fullmatch(digest) else None
        ),
    }


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _aware(value: dt.datetime | None) -> dt.datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=dt.UTC)
    return value


def _write_lock(session: Session, table: Any, column: Any, key: Any) -> None:
    """SQLite has no row locks and ignores ``FOR UPDATE``: a no-op write as the transaction's
    FIRST statement takes its database write lock, so concurrent reserves and settles serialise
    (a read first would let two transactions both decide on the same stale total; R1B-A2)."""
    if session.get_bind().dialect.name == "sqlite":
        session.execute(sa.update(table).where(column == key).values({column.key: column}))


def _locked_run(session: Session, run_id: str) -> WspRun | None:
    _write_lock(session, WspRun.__table__, WspRun.__table__.c.run_id, run_id)
    return session.scalar(sa.select(WspRun).where(WspRun.run_id == run_id).with_for_update())


def _locked_day(session: Session, day: dt.date) -> WspSpendDay:
    """The UTC day's row, locked. Created by ``INSERT ... ON CONFLICT DO NOTHING`` first (review
    OR1-016-10): a SELECT-then-INSERT let two first reserves of a day both find no row and both
    insert, and the second's unique violation became a 500. On PostgreSQL the insert waits for an
    uncommitted rival row and then does nothing; the ``FOR UPDATE`` read then locks the one row."""
    from sqlalchemy.dialects import postgresql, sqlite

    values = {"utc_day": day, "external_settled_usd": ZERO, "external_open_usd": ZERO}
    target = [WspSpendDay.__table__.c.utc_day]
    if session.get_bind().dialect.name == "postgresql":
        session.execute(
            postgresql.insert(WspSpendDay)
            .values(values)
            .on_conflict_do_nothing(index_elements=target)
        )
    else:
        session.execute(
            sqlite.insert(WspSpendDay).values(values).on_conflict_do_nothing(index_elements=target)
        )
    row = session.scalar(sa.select(WspSpendDay).where(WspSpendDay.utc_day == day).with_for_update())
    assert row is not None
    return row


def session_active(run: WspRun | None, token_sha256: str, now: dt.datetime) -> bool:
    """The run takes gateway traffic: ``pushing`` or ``running``, a session style, this token,
    not expired (R3A-13, R3A-18)."""
    if run is None or run.status not in ACTIVE or run.style in NO_GATEWAY_STYLES:
        return False
    stored = run.gateway_token_sha256
    if stored is None or not hmac.compare_digest(stored, token_sha256):
        return False
    expires = _aware(run.token_expires_at)
    return expires is not None and expires > now


def spent(session: Session, run_id: str) -> tuple[Decimal, int]:
    """Settled usage plus open reservations at worst case: ``(usd, tokens)``."""
    usage = session.execute(
        sa.select(
            sa.func.coalesce(sa.func.sum(WspUsage.cost_usd), 0),
            sa.func.coalesce(sa.func.sum(WspUsage.input_tokens + WspUsage.output_tokens), 0),
        ).where(WspUsage.run_id == run_id)
    ).one()
    open_ = session.execute(
        sa.select(
            sa.func.coalesce(sa.func.sum(WspReservation.worst_case_usd), 0),
            sa.func.coalesce(sa.func.sum(WspReservation.worst_case_tokens), 0),
        ).where(WspReservation.run_id == run_id, WspReservation.settled_at.is_(None))
    ).one()
    return Decimal(str(usage[0])) + Decimal(str(open_[0])), int(usage[1]) + int(open_[1])


#: How many served calls' evidence a session entry carries (the gateway's ``RATIO_MEMORY``).
EVIDENCE_CALLS = 3


def prompt_evidence(session: Session, run_id: str) -> list[dict[str, Any]]:
    """The run's last :data:`EVIDENCE_CALLS` served calls that carry prompt evidence, oldest
    first: ``{sent_bytes, prompt_tokens, messages, conversation_sha256}``. Counts and a digest
    only; empty for a run whose calls predate migration 0033."""
    rows = session.execute(
        sa.select(
            WspUsage.sent_bytes,
            WspUsage.input_tokens,
            WspUsage.prompt_messages,
            WspUsage.conversation_sha256,
        )
        .where(
            WspUsage.run_id == run_id,
            WspUsage.outcome == "served",
            WspUsage.sent_bytes.is_not(None),
            WspUsage.input_tokens > 0,
        )
        .order_by(WspUsage.seq.desc())
        .limit(EVIDENCE_CALLS)
    ).all()
    return [
        {
            "sent_bytes": int(r[0]),
            "prompt_tokens": int(r[1]),
            "messages": None if r[2] is None else int(r[2]),
            "conversation_sha256": r[3],
        }
        for r in reversed(rows)
        if r[0] is not None
    ]


def reserve(
    session: Session,
    *,
    run_id: str,
    token_sha256: str,
    worst_case_usd: Decimal,
    worst_case_tokens: int,
    external: bool,
    settings: WSPSettings,
    now: dt.datetime | None = None,
) -> int:
    """Reserve one request's worst case; the reservation id, or a refusal (nothing written)."""
    now = now or utcnow()
    if worst_case_usd < 0 or worst_case_tokens < 0:
        raise refuse("validation_failed", "A worst case is never negative.")
    run = _locked_run(session, run_id)
    if not session_active(run, token_sha256, now):
        raise refuse("wsp_session_not_active", "This session takes no gateway traffic.")
    assert run is not None
    open_count = session.scalar(
        sa.select(sa.func.count())
        .select_from(WspReservation)
        .where(WspReservation.run_id == run_id, WspReservation.settled_at.is_(None))
    )
    if open_count:
        raise refuse("wsp_request_in_flight", "One request per session at a time.")
    usd, tokens_used = spent(session, run_id)
    budgets = run.request.get("budgets") or {}
    cap_usd = budgets.get("cost_usd")
    if cap_usd is not None and usd + worst_case_usd > Decimal(str(cap_usd)):
        raise refuse("wsp_budget_exhausted", "The session's cost cap would be exceeded.")
    cap_tokens = budgets.get("tokens")
    if cap_tokens is not None and tokens_used + worst_case_tokens > int(cap_tokens):
        raise refuse("wsp_budget_exhausted", "The session's token budget would be exceeded.")
    if external:
        day = _locked_day(session, now.date())
        total = day.external_settled_usd + day.external_open_usd + worst_case_usd
        if total > Decimal(str(settings.wsp_external_spend_max_usd_per_day)):
            raise refuse(
                "wsp_daily_spend_cap_reached",
                "Today's external spend cap (UTC) would be exceeded; it resets at UTC midnight.",
            )
        day.external_open_usd += worst_case_usd
    row = WspReservation(
        run_id=run_id,
        endpoint_id=str(run.endpoint_id),
        external=external,
        worst_case_usd=worst_case_usd,
        worst_case_tokens=worst_case_tokens,
        created_at=now,
    )
    session.add(row)
    session.flush()
    return int(row.id)


def settle(
    session: Session,
    *,
    reservation_id: int,
    usage: dict[str, Any],
    call: dict[str, Any] | None = None,
    now: dt.datetime | None = None,
) -> None:
    """Close a reservation with its metered usage and append it to ``mf_wsp_usage``, with what
    became of the call (``call``: outcome, upstream status, latency, finish reason, refusal code;
    checked by :func:`call_columns`)."""
    now = now or utcnow()
    table = WspReservation.__table__
    _write_lock(session, table, table.c.id, reservation_id)
    found = session.get(WspReservation, reservation_id)
    if found is None:
        raise refuse("wsp_reservation_closed", "No such open reservation.")
    _locked_run(session, found.run_id)  # the run's lock first, then the day's (R3A-18)
    reservation = session.scalar(
        sa.select(WspReservation).where(WspReservation.id == reservation_id).with_for_update()
    )
    assert reservation is not None
    if reservation.settled_at is not None:
        raise refuse("wsp_reservation_closed", "This reservation is already closed.")
    cost = Decimal(str(usage.get("cost_usd", 0)))
    _close(session, reservation, cost, now)
    _append_usage(
        session,
        reservation,
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        cost=cost,
        now=now,
        **call_columns(call),
    )


def record_refusal(
    session: Session,
    *,
    run_id: str,
    token_sha256: str,
    code: str,
    now: dt.datetime | None = None,
) -> int:
    """Append a call the gateway REFUSED before anything was spent: zero tokens, zero cost,
    ``outcome: refused`` and its code (live progress, 2026-11-20). Only for a live session holding
    this token (the reserve's own check), so a refusal never lands on an ended run. The row's
    ``seq``, or a refusal (nothing written). It is not a request: :func:`wsp.runs.metered_usage`
    does not count it."""
    now = now or utcnow()
    run = _locked_run(session, run_id)
    if not session_active(run, token_sha256, now):
        raise refuse("wsp_session_not_active", "This session takes no gateway traffic.")
    assert run is not None
    columns = call_columns({"outcome": "refused", "refusal_code": code})
    if columns["refusal_code"] is None:
        raise refuse("validation_failed", "A refusal code is a lower-case identifier.")
    return _append(
        session,
        run_id=run_id,
        endpoint_id=str(run.endpoint_id),
        input_tokens=0,
        output_tokens=0,
        cost=ZERO,
        now=now,
        **columns,
    )


def close_open_at_worst(session: Session, run_id: str, now: dt.datetime | None = None) -> int:
    """On a run's terminal transition: every open reservation is settled at its worst case."""
    now = now or utcnow()
    count = 0
    for reservation in session.scalars(
        sa.select(WspReservation)
        .where(WspReservation.run_id == run_id, WspReservation.settled_at.is_(None))
        .with_for_update()
    ).all():
        _close(session, reservation, reservation.worst_case_usd, now)
        # Recorded as usage at worst case, so the run's metered total never drops it.
        _append_usage(
            session,
            reservation,
            input_tokens=0,
            output_tokens=int(reservation.worst_case_tokens),
            cost=reservation.worst_case_usd,
            now=now,
            outcome="closed_at_worst",
        )
        count += 1
    return count


def _append_usage(
    session: Session,
    reservation: WspReservation,
    *,
    input_tokens: int,
    output_tokens: int,
    cost: Decimal,
    now: dt.datetime,
    **call: Any,
) -> None:
    _append(
        session,
        run_id=reservation.run_id,
        endpoint_id=reservation.endpoint_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        now=now,
        **call,
    )


def _append(
    session: Session,
    *,
    run_id: str,
    endpoint_id: str,
    input_tokens: int,
    output_tokens: int,
    cost: Decimal,
    now: dt.datetime,
    **call: Any,
) -> int:
    seq = int(
        session.scalar(
            sa.select(sa.func.coalesce(sa.func.max(WspUsage.seq), -1) + 1).where(
                WspUsage.run_id == run_id
            )
        )
        or 0
    )
    session.add(
        WspUsage(
            run_id=run_id,
            seq=seq,
            endpoint_id=endpoint_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            at=now,
            **call,
        )
    )
    session.flush()
    return seq


def _close(session: Session, reservation: WspReservation, cost: Decimal, now: dt.datetime) -> None:
    if reservation.external:
        day = _locked_day(session, _aware(reservation.created_at).date())  # type: ignore[union-attr]
        day.external_open_usd -= reservation.worst_case_usd
        day.external_settled_usd += cost
    reservation.settled_at = now
    reservation.settled_cost_usd = cost
