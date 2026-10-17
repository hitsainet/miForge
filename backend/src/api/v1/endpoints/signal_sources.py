"""The signal-source surface: what this installation can train with, and whether it can.

Two routes, both read-only and both needing no GPU (FPRD §7):

* ``GET /api/v1/signal-sources`` - the **live registry's** contents, each source with what
  it requires and whether that is currently satisfied, naming any unsatisfied service. It
  backs F9's New-run select, so a source is never *shown and then refused on submit*.
* ``POST /api/v1/signal-sources/preflight`` - a dry pre-flight with no run in hand, which is
  the call F7's pack validation makes to check a pack's parts against the source that will
  consume them **before a run exists**.

**There is no write route for a batch**, here or anywhere: a batch is produced by a run, not
posted to one, and an endpoint that accepted one would be a second provenance path -
provenance with two paths has two answers.

The request models live in this module rather than in ``schemas/``: that package is shared
state with one owner and this feature does not edit it, exactly as ``runs_acquisition.py``
declares ``RunSubmit`` beside the route that parses it.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from api.deps import session_dependency
from services import signal_source_service
from training.signals.base import RunContext, SourceSpec

router = APIRouter(tags=["signal-sources"])


class PreflightRequest(BaseModel):
    """A source specification to resolve. **Closed to unknown keys** (ADR-018).

    An unknown key is a refusal rather than an ignored field: a misspelled ``group_size``
    must not silently leave the real value at its default, which would pre-flight a run
    different from the one the caller described.
    """

    model_config = ConfigDict(extra="forbid")

    spec: SourceSpec


@router.get("/signal-sources", summary="The four signal sources and what each needs")
def list_signal_sources() -> dict[str, Any]:
    """Every registered source, each unavailable one **named with its reason**.

    Answers with no run, no spec and no GPU. Availability is computed from the current
    settings **per call**, never cached at import, so configuring a provider does not need a
    restart to be visible.
    """
    sources = signal_source_service.listing()
    return {
        "sources": sources,
        # ADR-012: every number names its scale and sample. The count is stated so a reader
        # never mistakes a filtered view for the whole surface.
        "count": len(sources),
        "available": sum(1 for row in sources if row["available"]),
    }


@router.post("/signal-sources/preflight", summary="Resolve a source specification, or refuse")
def preflight_signal_source(
    request: PreflightRequest,
    session: Annotated[Session, Depends(session_dependency)],
) -> dict[str, Any]:
    """Resolve parts, splits, columns, the selector and the grader - or refuse by name.

    Synchronous and off every queue: it needs no GPU and no model, which is what lets a form
    refuse **before** submit rather than failing a run at step 1 (FR-002.4, R-02.48).
    """
    return signal_source_service.preflight(
        request.spec, context=RunContext(run_id="preflight", session=session)
    )
