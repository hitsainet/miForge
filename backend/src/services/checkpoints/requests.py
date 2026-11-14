"""The "Save now" request, as database state  (FR-005.1, E-12; ADR-006, ADR-007).

**Never a Celery control message.** Every worker runs ``--pool=solo -c 1``, so
``revoke(terminate=True)`` has no pool child to signal and a *busy* worker never reads the
control queue at all - which is precisely the worker a save request is for. ADR-007's rule
is the one that applies: **reach a running job through the database.** The loop reads this
at a step boundary, so a save never interrupts a step.

**Two requests for one boundary collapse into one save** (E-12). That is a property of the
table rather than of a handler: ``run_id`` is the primary key, so a second request updates
the one row. ``request_count`` is kept so the operator can still see that two were made -
collapsing the *effect* is correct, silently discarding the *fact* is not.

## Why this is its own table, and what is owed

FR-005.1 says the flag is *"a flag on the ``mf_runs`` row"*. ``mf_runs`` is the foundation's
and is not this feature's to alter, and the Alembic revision graph is serialised through the
coordinator while several features are in flight. So the flag lives in a table of **this
feature's own metadata**, following ``services/parts/tables.py``'s worked example: the same
``mf_`` prefix, the same naming convention, and deliberately **not** on ``Base.metadata``,
whose ratchet asserts the base schema is exactly its seven tables.

The substance of the requirement is unchanged - durable state the loop consumes at a
boundary, collapsing - and the deviation is reported rather than absorbed. A
``mf_runs.save_requested_at`` column would be the straight reading and is a one-line
migration; this table is what can be built without touching another owner's file.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, cast

import sqlalchemy as sa

from core.errors import refuse
from models.run import Run

log = logging.getLogger("miforge.checkpoints.requests")

__all__ = [
    "consume_save_request",
    "pending_save_request",
    "request_save",
]

#: The run row. FR-005.1's straight reading - *"a flag on the ``mf_runs`` row"* - and the
#: better engineering besides: the loop already reads this row at every step boundary for
#: the heartbeat and the cancel flag, so a separate table would add a join to the hottest
#: path in the product in order to carry one timestamp.
#:
#: *This module built its own table on its own ``MetaData`` until 2026-10-14, because
#: ``backend/src/models/`` is not a feature agent's to edit. That was the right call for an
#: agent to make - deviate, say so, and offer the straight reading - and the coordinator
#: took the offer. ``mf_runs`` gained ``save_requested_at`` and ``save_request_count`` in
#: migration 0012.*
RUNS = cast(sa.Table, Run.__table__)


def request_save(session: Any, run_id: str) -> dict[str, Any]:
    """Record a save request for ``run_id``. Returns the pending request as it now stands.

    **Idempotent by construction.** A second request before the loop reaches a boundary
    updates the same row: there is no way to represent two pending saves for one run,
    because one save is what was wanted. ``save_request_count`` keeps the *fact* that two
    were asked for, so an operator who pressed the button twice can see the system noticed
    both - the effect collapses, the record does not.
    """
    now = datetime.now(UTC)
    row = (
        session.execute(sa.select(RUNS.c.save_request_count).where(RUNS.c.id == run_id))
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise refuse(
            "run_not_found",
            f"No run {run_id!r}, so a save cannot be requested against it.",
            run_id=run_id,
        )

    count = int(row["save_request_count"] or 0) + 1
    session.execute(
        sa.update(RUNS)
        .where(RUNS.c.id == run_id)
        .values(save_requested_at=now, save_request_count=count)
    )
    session.flush()

    log.info(
        "save requested",
        extra={
            "code": "checkpoint_save_requested",
            "run_id": run_id,
            "request_count": count,
        },
    )
    return {"run_id": run_id, "requested_at": now.isoformat(), "request_count": count}


def pending_save_request(session: Any, run_id: str) -> dict[str, Any] | None:
    """The pending request for ``run_id``, or ``None``. Read at a step boundary."""
    row = (
        session.execute(
            sa.select(RUNS.c.id, RUNS.c.save_requested_at, RUNS.c.save_request_count).where(
                RUNS.c.id == run_id
            )
        )
        .mappings()
        .one_or_none()
    )
    if row is None or row["save_requested_at"] is None:
        return None
    requested_at = row["save_requested_at"]
    return {
        "run_id": row["id"],
        "requested_at": (
            requested_at.isoformat() if hasattr(requested_at, "isoformat") else str(requested_at)
        ),
        "request_count": int(row["save_request_count"] or 0),
    }


def consume_save_request(session: Any, run_id: str) -> dict[str, Any] | None:
    """Take the pending request, if any, and clear it. Returns what was taken.

    **Cleared rather than flagged consumed.** A consumed-flag would be a second state the
    loop and the API would both have to agree about, and the request's whole life is one
    boundary. The historical record of the save is the checkpoint and its manifest - which
    is the durable artifact, so a second state here would carry no information the disk
    does not already carry.

    ``save_request_count`` is deliberately **not** reset: it counts requests over the run's
    life, not requests since the last save. Resetting it would lose the only evidence that
    an operator asked twice.
    """
    pending = pending_save_request(session, run_id)
    if pending is None:
        return None
    session.execute(sa.update(RUNS).where(RUNS.c.id == run_id).values(save_requested_at=None))
    session.flush()
    return pending
