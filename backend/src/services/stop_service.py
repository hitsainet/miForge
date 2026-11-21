"""Requesting a stop  (FR-006.26, E-7, E-8).

A thin layer over the foundation's cooperative-cancellation primitive
(:func:`workers.cancellation.request_stop`), which F6 **reuses and does not replace**: F1
owns ``POST /api/v1/runs/{run_id}/cancel`` and the column, and F6 owns the condition, the
precedence and the terminal transition (AMB-3). What this module adds is the one thing the
primitive does not express - **a terminal run refuses**.

## Why a terminal run must refuse rather than silently succeed  (E-7)

``request_stop`` is idempotent and collapses two requests into one, which is right: one
stop is what was wanted, and ``cancel_requested_at`` keeps *when* it was first asked so a
slow shutdown is distinguishable from an ignored request. But a ``200`` for a stop that
stopped nothing tells the operator something untrue - they believe they stopped a run that
had already finished, or already failed, and the next thing they do is based on that belief.
So a request against a terminal row is ``run_not_stoppable``, naming the state it is in.

## A queued run is stopped without ever allocating  (E-8)

Honoured, not refused: the row transitions terminal without acquiring the parameters and
the lease is released. Refusing it would leave an operator unable to cancel a queue, which
on a one-card node is the queue for the only card there is.

**No Celery control message is sent on this path, ever.** ``revoke(terminate=True)``
signals a pool child and the solo pool has none, so the call returns successfully and
nothing happens - the operator sees a cancellation that was accepted. A guard test asserts
the string appears nowhere in ``backend/src/``.
"""

from __future__ import annotations

import logging
from typing import Any

import sqlalchemy as sa

from core.errors import refuse
from training import run_state
from training.stop.base import StopReason
from workers.cancellation import request_stop as _request_stop
from workers.cancellation import stop_requested_at
from workers.run_record import RUNS, TERMINAL_STATES, require_run

log = logging.getLogger(__name__)


def request_stop(session: Any, run_id: str, *, label: str | None = None) -> dict[str, Any]:
    """Record an operator stop. Idempotent; refuses a terminal run.

    Args:
        session: an open session. The caller commits.
        run_id: the run to stop.
        label: the requesting identity's audit label (FR-006.24). Logged, **not** stored:
            ``mf_runs`` has no column for it and inventing one here would be a migration
            this feature does not own. The DDL is in the feature report.

    Returns:
        ``{"run_id", "stop_pending", "cancel_requested_at", "collapsed", "stopped_now"}``.
        ``collapsed`` is ``True`` when a request was already outstanding - the effect
        collapses, and the *fact* that two were asked for does not. ``stopped_now`` is
        ``True`` when the run was still ``pending`` and this call took it terminal itself,
        which is the only status where there is no step boundary to honour the request at.

    Raises:
        Refusal: ``run_not_found``; ``run_not_stoppable`` on a terminal run (E-7).
    """
    status = require_run(session, run_id)
    if status in TERMINAL_STATES:
        raise refuse(
            "run_not_stoppable",
            f"Run {run_id!r} is already {status}, so there is nothing to stop. Refused "
            f"rather than accepted silently: a success for a stop that stopped nothing "
            f"would tell you that you had stopped a run that had already finished. "
            f"Continue from its last checkpoint if you meant to carry on, or branch from "
            f"one if you meant to try something else.",
            run_id=run_id,
            status=status,
        )

    already = stop_requested_at(session, run_id)
    _bound_lock_wait(session)
    try:
        pending = _request_stop(session, run_id)
    except sa.exc.OperationalError as exc:
        if not _is_lock_timeout(exc):
            raise
        session.rollback()
        raise refuse(
            "run_row_busy",
            f"Run {run_id!r}'s row stayed locked by another transaction for longer than "
            f"{LOCK_WAIT_SECONDS} s, so the stop was NOT recorded. Nothing was changed; ask "
            f"again. A worker that holds its run row this long is itself a defect - check "
            f"the worker's database session.",
            run_id=run_id,
            lock_wait_seconds=LOCK_WAIT_SECONDS,
        ) from exc
    requested_at = stop_requested_at(session, run_id)

    # E-7 AFTER THE WRITE, NOT ONLY BEFORE IT. The status read above can be stale by the
    # time the UPDATE runs - in F11 window 5 the UPDATE waited for the whole run and found
    # the row `completed` - and the route then answered 202 with `cancel_requested_at:
    # null`, a stop that stopped nothing. So a request that recorded nothing is refused
    # with the state the row is in NOW.
    if not pending or requested_at is None:
        now_status = require_run(session, run_id)
        raise refuse(
            "run_not_stoppable",
            f"Run {run_id!r} finished ({now_status}) before the stop request landed, so "
            f"there was nothing to stop and nothing was recorded. Continue from its last "
            f"checkpoint if you meant to carry on.",
            run_id=run_id,
            status=now_status,
        )

    # E-8, PERFORMED RATHER THAN RECORDED. A `pending` run has no worker and therefore no
    # step boundary, and cooperative cancellation happens *at* a step boundary (ADR-007) -
    # so for this one status the flag alone is an intention nothing will ever act on. The
    # row stayed `pending` until the janitor reaped it, and an operator's own cancelled run
    # surfaced as *worker lost*: the wrong cause of death on the record, with the GPU lease
    # held to its expiry instead of released.
    #
    # `stop_queued_run`'s own docstring named this caller - *"the janitor sweep, or the
    # request itself when the row is still `pending`"* - and **neither existed**. Found by
    # `0xcc/scripts/check_reachability.py`: two test references, no production caller.
    #
    # It re-reads the status itself and returns False for anything but `pending`, so a
    # `running` run is left to its own loop. That direction is asserted too: reaching past
    # a live worker to take its row terminal would strand it mid-step and lose the
    # checkpoint its boundary was about to take.
    stopped_now = stop_queued_run(session, run_id)

    log.info(
        "stop requested",
        extra={
            "code": "run_stop_requested",
            "run_id": run_id,
            "status": status,
            # FR-006.24's audit label. Logged rather than dropped, so the audit trail
            # exists even though the column does not.
            "requested_by": label,
            "collapsed": already is not None,
            "stopped_now": stopped_now,
        },
    )
    return {
        "run_id": run_id,
        "stop_pending": pending,
        "cancel_requested_at": None if requested_at is None else requested_at.isoformat(),
        "collapsed": already is not None,
        # Additive. `POST /runs/{id}/cancel` selects its keys explicitly, so the wire shape
        # is unchanged; this is here so a caller CAN tell "it has stopped" from "it will
        # stop at the next boundary", which are different facts and were one.
        "stopped_now": stopped_now,
    }


#: How long the stop's UPDATE may wait on the run row's lock (PostgreSQL only). A healthy
#: worker holds the row for milliseconds - one short transaction per step (F11 18.5) - so
#: a wait this long means a defect, and the route says so instead of hanging for a run.
LOCK_WAIT_SECONDS = 5


def _bound_lock_wait(session: Any) -> None:
    """Bound this transaction's lock waits, on PostgreSQL. A no-op on SQLite (no such knob,
    and its own ``timeout`` already bounds a wait)."""
    if session.get_bind().dialect.name == "postgresql":
        session.execute(sa.text(f"SET LOCAL lock_timeout = '{LOCK_WAIT_SECONDS}s'"))


def _is_lock_timeout(exc: sa.exc.OperationalError) -> bool:
    """PostgreSQL's ``lock_not_available`` (SQLSTATE 55P03), which ``lock_timeout`` raises."""
    return getattr(getattr(exc, "orig", None), "sqlstate", None) == "55P03"


def stop_queued_run(session: Any, run_id: str) -> bool:
    """Take a ``pending`` run terminal without ever allocating  (E-8).

    Called by whoever notices a stop request on a run no worker has picked up - the janitor
    sweep, or the request itself when the row is still ``pending``. Returns whether the row
    moved.

    The lease is released **after** the terminal transition (FR-006.29), because a lease
    has an expiry and a run row does not: a crash between the two leaves the lease to free
    the card by itself.
    """
    row = session.execute(
        sa.select(RUNS.c.status, RUNS.c.gpu_uuid).where(RUNS.c.id == run_id)
    ).one_or_none()
    if row is None or row.status != "pending":
        return False

    moved = run_state.mark_completed(
        session,
        run_id,
        stop_reason=StopReason.OPERATOR_STOP.value,
        detail="stopped before the run started; no parameters were acquired",
    )
    if moved:
        run_state.release_lease(session, run_id, row.gpu_uuid)
    return moved


__all__ = ["request_stop", "stop_queued_run"]
