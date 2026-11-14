"""``GET /api/v1/runs/{run_id}/review`` - the run's review counts  (FR-010.28).

A separate module from ``runs.py`` because that route is F1's and F6 extends it; this one is
F10's. Two routers may share the ``/runs`` prefix, and keeping them apart means neither
feature edits the other's file (PPRD §6: *no two agents edit the same file*).

**Every number names its scale and sample** (ADR-012): the payload carries ``posted``,
``decided`` and ``pending`` *and* a rendered summary - *"128 pending of 500 posted"* - so a
UI showing a bare count has to go out of its way to do so.

**The counts come from the rows**, which is also where the janitor reads. Progress reported
only over the socket is invisible to it, and that is how a live 5.8-hour job was reaped as
"worker lost" (ADR-007).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.deps import session_dependency
from services.review import binding as binding_service

router = APIRouter(prefix="/runs", tags=["runs", "review"])


@router.get("/{run_id}/review", summary="One run's review binding and its counts")
def get_run_review(
    run_id: str,
    session: Annotated[Session, Depends(session_dependency)],
) -> dict[str, Any]:
    """The queue this run is bound to, and how many candidates are waiting.

    **miForge holds no queue** - this is a pointer to miDataworks' and three counts
    (R-02.36, FR-010.25). There is no item list here, no decision list and no way to record
    a decision: an adjudication route in this application would be the second answer to the
    one question miDataworks already answers.

    Raises:
        Refusal: ``review_binding_missing`` when the run has no queue - named, because
            "this run does not use human review" and "its queue was never created" have
            different fixes and an empty body would read as "no candidates".
    """
    return binding_service.review_payload(session, run_id)
