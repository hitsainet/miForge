"""Archive and unarchive runs and parts  (operator decision 2026-10-20; migration 0022).

**Archive is visibility, never deletion.** An archived run or part is left out of the
default lists - ``GET /runs``, ``GET /parts``, the Runs screen, the run dropdown and the part
pickers - until the caller asks for ``include_archived``. Nothing else changes: the run row,
its steps, its run log, its checkpoints and its trainer state stay exactly where they were,
the run view and ``GET /parts/{id}`` still answer, and unarchiving restores the listing.
Deleting a checkpoint stays the separate, explicit, approval-gated operation it already is.

**Never a pending or running run.** :func:`archive_run` is one conditional ``UPDATE`` whose
``WHERE`` carries ``status IN terminal``, so a run that starts between a read and the write
is refused rather than hidden; and ``run_state.reopen_for_continue`` clears ``archived_at``
in the statement that reopens a run. The run list also never hides a non-terminal row,
whatever the column says (``runs_index.visible``).

Writes, but **non-destructive** ones, so an agent's call runs at once (F12 D9: "non-deleting
writes ... are ungated"); no route here carries ``requires_approval_when_agent``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.errors import refuse
from services.parts.tables import CONTRACT_PARTS, PART_ARCHIVE
from workers.run_record import RUNS, TERMINAL_STATES, as_aware, rowcount_of, utcnow

log = logging.getLogger("miforge.archive")


def _iso(value: Any) -> str | None:
    aware = as_aware(value)
    return None if aware is None else aware.isoformat()


def _run_row(session: Session, run_id: str) -> Any:
    row = session.execute(
        sa.select(RUNS.c.status, RUNS.c.archived_at).where(RUNS.c.id == run_id)
    ).one_or_none()
    if row is None:
        raise refuse(
            "run_not_found",
            f"No run with id {run_id!r}. Check the identifier.",
            run_id=run_id,
        )
    return row


def _run_answer(run_id: str, row: Any, *, changed: bool) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": row.status,
        "archived": row.archived_at is not None,
        "archived_at": _iso(row.archived_at),
        "changed": changed,
    }


def archive_run(session: Session, run_id: str, *, now: datetime | None = None) -> dict[str, Any]:
    """Hide a finished run from the default lists. Idempotent; deletes nothing.

    Raises:
        Refusal: ``run_not_found``; ``run_not_archivable`` (409) when the run is pending or
            running - an active run is never hidden.
    """
    result = session.execute(
        sa.update(RUNS)
        .where(RUNS.c.id == run_id)
        .where(RUNS.c.status.in_(sorted(TERMINAL_STATES)))
        .where(RUNS.c.archived_at.is_(None))
        .values(archived_at=now or utcnow())
    )
    changed = rowcount_of(result) > 0
    session.flush()
    row = _run_row(session, run_id)
    if row.archived_at is None:
        # Not terminal: the conditional UPDATE matched nothing and nothing set it.
        raise refuse(
            "run_not_archivable",
            f"Run {run_id!r} is {row.status}. Only a finished run (completed, failed or "
            f"cancelled) can be archived; stop it first, or wait for it to end.",
            run_id=run_id,
            status=row.status,
        )
    if changed:
        log.info("run archived", extra={"code": "run_archived", "run_id": run_id})
    return _run_answer(run_id, row, changed=changed)


def unarchive_run(session: Session, run_id: str) -> dict[str, Any]:
    """Return a run to the default lists. Idempotent.

    Raises:
        Refusal: ``run_not_found``.
    """
    result = session.execute(
        sa.update(RUNS)
        .where(RUNS.c.id == run_id)
        .where(RUNS.c.archived_at.is_not(None))
        .values(archived_at=None)
    )
    changed = rowcount_of(result) > 0
    session.flush()
    row = _run_row(session, run_id)
    if changed:
        log.info("run unarchived", extra={"code": "run_unarchived", "run_id": run_id})
    return _run_answer(run_id, row, changed=changed)


# ====================================================================== parts


def part_archived_at(session: Session, part_id: str) -> str | None:
    """When ``part_id`` was archived, or ``None`` when it is listed."""
    value = session.execute(
        sa.select(PART_ARCHIVE.c.archived_at).where(PART_ARCHIVE.c.part_id == part_id)
    ).scalar_one_or_none()
    return _iso(value)


def archived_parts(session: Session) -> dict[str, str | None]:
    """``{part_id: archived_at}`` for every archived part."""
    rows = session.execute(sa.select(PART_ARCHIVE.c.part_id, PART_ARCHIVE.c.archived_at)).all()
    return {str(row.part_id): _iso(row.archived_at) for row in rows}


def _part_answer(session: Session, part_id: str, *, changed: bool) -> dict[str, Any]:
    archived_at = part_archived_at(session, part_id)
    return {
        "part_id": part_id,
        "archived": archived_at is not None,
        "archived_at": archived_at,
        "changed": changed,
    }


def _require_indexed_part(session: Session, part_id: str) -> None:
    found = session.execute(
        sa.select(CONTRACT_PARTS.c.id).where(CONTRACT_PARTS.c.id == part_id)
    ).one_or_none()
    if found is None:
        raise refuse(
            "part_not_found",
            f"No part with id {part_id!r} in the index.",
            part_id=part_id,
        )


def archive_part(session: Session, part_id: str, *, now: datetime | None = None) -> dict[str, Any]:
    """Hide a part from the default lists and pickers. Idempotent; deletes nothing.

    A pack that already pins the part, and a run that used it, are untouched: archive is
    a listing filter, and every read by id still answers.

    Raises:
        Refusal: ``part_not_found`` when the id is not in the part index.
    """
    _require_indexed_part(session, part_id)
    changed = False
    if part_archived_at(session, part_id) is None:
        try:
            with session.begin_nested():
                session.execute(
                    sa.insert(PART_ARCHIVE).values(part_id=part_id, archived_at=now or utcnow())
                )
            changed = True
        except IntegrityError:
            # A concurrent archive of the same part won the insert; the answer is the same.
            changed = False
    if changed:
        log.info("part archived", extra={"code": "part_archived", "part_id": part_id})
    return _part_answer(session, part_id, changed=changed)


def unarchive_part(session: Session, part_id: str) -> dict[str, Any]:
    """Return a part to the default lists. Idempotent.

    An archive row whose part has since left the index (deleted from disk, then reindexed)
    can still be removed, so the call never strands one.

    Raises:
        Refusal: ``part_not_found`` when the id is neither indexed nor archived.
    """
    result = session.execute(sa.delete(PART_ARCHIVE).where(PART_ARCHIVE.c.part_id == part_id))
    changed = rowcount_of(result) > 0
    session.flush()
    if not changed:
        _require_indexed_part(session, part_id)
    else:
        log.info("part unarchived", extra={"code": "part_unarchived", "part_id": part_id})
    return _part_answer(session, part_id, changed=changed)
