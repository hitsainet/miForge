"""``GET /api/v1/runs`` - every run, newest first, a page at a time  (F9 task 14.0, AC-009.40).

**Until 2026-10-18 this route did not exist.** The path carried only F1's ``POST /runs``, so a
GET there answered **405**, and the Runs screen could only *open* a run whose identifier the
operator already knew. Six real runs on the cluster were invisible to the person who had
started them. F9's **F-2** had recorded that no requirement describes a run list; the
operator's request of 2026-10-18 ("selection boxes should be dropdowns that list what is
available") is the requirement.

## What a row carries, and what it deliberately does not

Exactly what a list row and a picker need: ``run_id``, ``status``, ``run_type``,
``created_at``, ``finished_at``, ``last_step``, ``stop_reason``, ``failure_code``. **Not** the
liveness block, the definition, the scores or the attachment: those are
``GET /runs/{run_id}``'s, which reads the run log for each, and a list that tailed one log per
row would make the cheapest screen in the product the most expensive.

``stop_reason`` is therefore the **column** ``run_state.mark_completed`` writes (migration
0016). The run view still reads the run log, which is authoritative (ADR-004); a row whose
column is null and whose log has a boundary block shows the reason on the run view and a null
here, and that is a statement about which store was read, not a guess.

## Read-only, and its own include line

No write path. A separate module from ``runs.py`` (the foundation's ``GET /runs/{run_id}``)
so that deleting this router's include line removes **only** the listing - which is what the
removal control asserts - rather than taking the run view with it.
"""

from __future__ import annotations

from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.deps import session_dependency
from core.errors import refuse
from models.run import RUN_STATUSES, TERMINAL_STATUSES
from training import runtypes
from workers.run_record import RUNS, as_aware

router = APIRouter(prefix="/runs", tags=["runs"])

#: The page size when none is asked for, and the most one request may ask for. A list is a
#: navigational surface; a client that wants everything pages.
DEFAULT_LIMIT = 50
MAX_LIMIT = 200


def _iso(value: Any) -> str | None:
    aware = as_aware(value)
    return None if aware is None else aware.isoformat()


def run_row(record: dict[str, Any]) -> dict[str, Any]:
    """One run as the list serves it. The **only** producer of a list row."""
    return {
        # `run_id` on the wire, `id` in the column - the name every manifest, log line and
        # socket room uses, and the one `GET /runs/{run_id}` returns.
        "run_id": record["id"],
        "status": record["status"],
        "run_type": record["run_type"],
        "created_at": _iso(record.get("created_at")),
        "finished_at": _iso(record.get("finished_at")),
        "last_step": record.get("last_step"),
        "stop_reason": record.get("stop_reason"),
        "failure_code": record.get("failure_code"),
        # Archive (operator 2026-10-20): null = listed. An archived row reaches a list only
        # when the caller asked for `include_archived`, and says so here.
        "archived_at": _iso(record.get("archived_at")),
    }


def visible(include_archived: bool) -> list[Any]:
    """The archive filter. **A pending or running run is never hidden**, whatever its
    ``archived_at`` says: archive refuses an active run and a continue clears the column, and
    this is the third holder of the rule, at the read, so a row that got past both is still
    shown rather than vanishing while it trains."""
    if include_archived:
        return []
    return [
        sa.or_(
            RUNS.c.archived_at.is_(None),
            RUNS.c.status.not_in(sorted(TERMINAL_STATUSES)),
        )
    ]


@router.get("", summary="Every run, newest first, a page at a time")
def list_runs(
    session: Annotated[Session, Depends(session_dependency)],
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
    status: Annotated[str | None, Query(description=f"one of {sorted(RUN_STATUSES)}")] = None,
    run_type: Annotated[str | None, Query(description="a name from GET /api/v1/run-types")] = None,
    include_archived: Annotated[
        bool, Query(description="also list archived runs (hidden by default)")
    ] = False,
) -> dict[str, Any]:
    """Runs, newest first, with the total so a client can say "12 of 31".

    An unknown filter is **refused by name** rather than answered with an empty page: an
    empty list for a misspelled status reads as "there are no failed runs", which is a
    different and false statement.
    """
    if status is not None and status not in RUN_STATUSES:
        raise refuse(
            "run_status_unknown",
            f"{status!r} is not a run status. Runs are {sorted(RUN_STATUSES)}.",
            given=status,
            known=sorted(RUN_STATUSES),
        )
    if run_type:
        # F8's registry is the authority on what a run type is, and refuses an unknown one
        # by name (`run_type_unknown`, listing the known). Not the model's constant: a
        # second list of run types is the drift `DB_RUN_TYPE` once was. Registry names are
        # the stored values since migration 0014.
        runtypes.resolve(run_type)

    # A generic column filter, not a branch on a run type: F8's guard forbids the latter
    # outside its package (`test_runtype_registry.py`), and a list filter is neither.
    filters = {RUNS.c.status: status, RUNS.c.run_type: run_type or None}
    conditions = [column == value for column, value in filters.items() if value is not None]
    conditions += visible(include_archived)

    total = session.execute(
        sa.select(sa.func.count()).select_from(RUNS).where(*conditions)
    ).scalar_one()
    rows = session.execute(
        sa.select(
            RUNS.c.id,
            RUNS.c.status,
            RUNS.c.run_type,
            RUNS.c.created_at,
            RUNS.c.finished_at,
            RUNS.c.last_step,
            RUNS.c.stop_reason,
            RUNS.c.failure_code,
            RUNS.c.archived_at,
        )
        .where(*conditions)
        # Newest first; the id breaks a tie so a page boundary never repeats or drops a row
        # when two runs share a timestamp (SQLite's `now()` has one-second resolution).
        .order_by(RUNS.c.created_at.desc(), RUNS.c.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()

    return {
        "runs": [run_row(dict(row._mapping)) for row in rows],
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "include_archived": include_archived,
        # The filter vocabularies, so a client offers them as a list rather than typing them.
        "statuses": sorted(RUN_STATUSES),
        "run_types": sorted(runtypes.known()),
    }
