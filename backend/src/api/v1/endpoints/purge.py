"""Purge: permanently delete ARCHIVED runs and parts  (operator decision 2026-10-20; 0023).

``POST /runs/purge/plan`` and ``/parts/purge/plan`` say, per item, what a purge would
remove (rows, files, bytes) and what blocks it, changing nothing. ``POST /runs/purge`` and
``/parts/purge`` purge a selection, **each item independently** - one blocked item never
stops the rest - and answer a result per item. ``POST /runs/{id}/purge`` and
``/parts/{id}/purge`` purge one and refuse with the status the refusal names (404 not found,
409 ``not_archived`` / ``purge_blocked``). ``GET /purges`` reads the append-only tombstones.
The rules are :mod:`services.purge.service`'s.

**Operator only, on every route here.** The router's dependency refuses any request that
carries ``X-Miforge-Agent`` with ``403 purge_operator_only`` - including the preview and the
tombstone read, so an agent cannot even plan a permanent delete. There is **no MCP tool**:
``tests/support/mcp_ledger.py`` lists each route in ``ROUTE_EXEMPT`` with the reason
"operator-only permanent delete", and the parity guard keeps it that way.

Its own module and its own include line, so deleting that line removes exactly these routes -
which ``test_purge.py``'s reachability test asserts.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from api.agent_origin import get_actor
from api.deps import session_dependency
from core.errors import refuse
from services.purge import service


def operator_only(request: Request) -> None:
    """``403 purge_operator_only`` for any request that identifies as an agent."""
    actor = get_actor(request)
    if actor.origin == "agent":
        raise refuse(
            "purge_operator_only",
            "Permanent delete is the operator's alone, from miForge's own screens. Agents can "
            "archive and unarchive; they cannot purge, preview a purge or read the purge log.",
            agent=actor.agent,
        )


router = APIRouter(tags=["purge"], dependencies=[Depends(operator_only)])

SessionDep = Annotated[Session, Depends(session_dependency)]


class PurgeSelection(BaseModel):
    """A selection to preview or purge."""

    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(min_length=1, max_length=service.MAX_BATCH)
    #: Why, in the operator's words. Kept on every tombstone of the batch.
    reason: str | None = Field(default=None, max_length=2000)


class PurgeOne(BaseModel):
    """The optional body of a single-item purge."""

    model_config = ConfigDict(extra="forbid")

    reason: str | None = Field(default=None, max_length=2000)


# ================================================================================ runs


@router.post("/runs/purge/plan", summary="What purging these runs would remove or why not")
def preview_runs(body: PurgeSelection, session: SessionDep) -> dict[str, Any]:
    """Per run: rows, files and bytes a purge would remove, and every blocker. Changes nothing."""
    return {"kind": "run", "items": service.plan_runs(session, body.ids)}


@router.post("/runs/purge", summary="Permanently delete archived runs, each independently")
def purge_runs(body: PurgeSelection, session: SessionDep) -> dict[str, Any]:
    """Purge each run on its own; a blocked run is reported and the rest proceed."""
    return service.purge_many(session, "run", body.ids, reason=body.reason)


@router.post("/runs/{run_id}/purge", summary="Permanently delete one archived run")
def purge_run(run_id: str, session: SessionDep, body: PurgeOne | None = None) -> dict[str, Any]:
    """Refused ``not_archived`` or ``purge_blocked`` (409) with every blocker named."""
    return service.purge_run(session, run_id, reason=None if body is None else body.reason)


# =============================================================================== parts


@router.post("/parts/purge/plan", summary="What purging these parts would remove or why not")
def preview_parts(body: PurgeSelection, session: SessionDep) -> dict[str, Any]:
    """Per part: rows, files and bytes a purge would remove, and every blocker."""
    return {"kind": "part", "items": service.plan_parts(session, body.ids)}


@router.post("/parts/purge", summary="Permanently delete archived parts, each independently")
def purge_parts(body: PurgeSelection, session: SessionDep) -> dict[str, Any]:
    """Purge each part on its own; a blocked part is reported and the rest proceed."""
    return service.purge_many(session, "part", body.ids, reason=body.reason)


@router.post("/parts/{part_id}/purge", summary="Permanently delete one archived part")
def purge_part(part_id: str, session: SessionDep, body: PurgeOne | None = None) -> dict[str, Any]:
    """Refused ``not_archived`` or ``purge_blocked`` (409) with every blocker named."""
    return service.purge_part(session, part_id, reason=None if body is None else body.reason)


# ============================================================================ the log


@router.get("/purges", summary="The purge log: one tombstone per permanent delete")
def list_purges(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    """Newest first. Append-only: nothing in the API or the database can change a row."""
    items = service.list_tombstones(session, limit=limit)
    return {"items": items, "total": len(items)}
