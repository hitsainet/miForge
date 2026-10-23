"""Archive and unarchive runs and parts: ``POST /runs|parts/{id}/archive|unarchive``.

Operator decision 2026-10-20. **Visibility only** - see :mod:`services.archive`: nothing is
deleted, every read by id still answers, and the lists show archived items again with
``include_archived=true``. A pending or running run is refused ``run_not_archivable`` (409).

**Ungated.** Non-deleting writes run at once for an agent too (F12 D9), so no route here
carries ``requires_approval_when_agent``; ``test_mcp_parity.py`` checks the MCP tools state
the same.

Its own module and its own include line, so deleting that line removes exactly these four
routes - which is what ``test_archive.py``'s reachability tests assert.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.deps import session_dependency
from services import archive

router = APIRouter(tags=["archive"])


@router.post("/runs/{run_id}/archive", summary="Hide a finished run from the default lists")
def archive_run(
    run_id: str, session: Annotated[Session, Depends(session_dependency)]
) -> dict[str, Any]:
    """Archive a completed, failed or cancelled run. Deletes nothing; idempotent."""
    return archive.archive_run(session, run_id)


@router.post("/runs/{run_id}/unarchive", summary="Return a run to the default lists")
def unarchive_run(
    run_id: str, session: Annotated[Session, Depends(session_dependency)]
) -> dict[str, Any]:
    """Unarchive a run. Idempotent."""
    return archive.unarchive_run(session, run_id)


@router.post("/parts/{part_id}/archive", summary="Hide a part from the default lists and pickers")
def archive_part(
    part_id: str, session: Annotated[Session, Depends(session_dependency)]
) -> dict[str, Any]:
    """Archive an indexed part. Deletes nothing; a pack pinning it is untouched; idempotent."""
    return archive.archive_part(session, part_id)


@router.post("/parts/{part_id}/unarchive", summary="Return a part to the default lists")
def unarchive_part(
    part_id: str, session: Annotated[Session, Depends(session_dependency)]
) -> dict[str, Any]:
    """Unarchive a part. Idempotent."""
    return archive.unarchive_part(session, part_id)
