"""F8's API surface: run types, the budget preview, adapters and the merge  (FPRD section 7).

Five routes in three routers, because they are three surfaces with three readers and
FastAPI prefixes are per-router. One module, so this feature adds **one** file to a
directory several features write into concurrently.

## ``{name:path}``, and why the plain converter could not work

An adapter's name is ``<base_model>:<run_id>:adapter-<step>``, and ``base_model`` for a Hub
base is **a repo id containing a slash** - ``LiquidAI/LFM2.5-1.2B-Instruct``. With the
default converter ``GET /api/v1/adapters/{name}`` would 404 every adapter trained on a Hub
model, which is the ordinary case rather than an edge one. F8's FPRD renders the path with
the plain converter; that is a defect in the document and is reported. The path converter is
used here instead, and ``services.runtypes_service.read_adapter`` parses the name and
resolves the directory through F5's confined ``run_dir`` - a traversal is **refused, not
normalised**.

## The preview answers "will it fit" before anything is submitted

``POST /budget/estimate`` is what lets a New-run form show fit-or-not while a definition is
still incomplete (FR-008.18). **F6's own** ``POST /runs/validate`` carries the budget figures
once a definition is complete, so F9 should use F6's there and not ask both: two numbers on
one screen that could disagree is worse than one.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from api.agent_origin import requires_approval_when_agent
from api.deps import session_dependency
from api.v1.endpoints.devices import device_row
from core import gpu
from core.errors import Refusal
from core.gpu import Device
from services import approval_gates, merge_record, runtypes_service
from services.acquisition import resolve_submit_device
from training.acquisition.base import AcquisitionSpec
from training.run_config import RunConfig
from training.runtypes.adapters.config import AdapterConfigIn

log = logging.getLogger("miforge.runtypes")

run_types_router = APIRouter(prefix="/run-types", tags=["run-types"])
budget_router = APIRouter(prefix="/budget", tags=["budget"])
adapters_router = APIRouter(prefix="/adapters", tags=["adapters"])
merges_router = APIRouter(prefix="/merges", tags=["run-types"])


class BudgetRequest(BaseModel):
    """A proposed run, for an estimate only. Nothing is submitted and nothing is allocated.

    Closed to unknown keys: a misspelled ``architecture`` would otherwise leave the estimate
    quietly derived from nothing, and a preview of nothing is worse than a refusal.
    """

    model_config = ConfigDict(extra="forbid")

    run_type: str
    config: RunConfig
    #: The run definition's model source, as the New run form holds it. The server reads
    #: the model's size from it (``services.budget_source``), so the form never computes or
    #: guesses a size. Sent instead of ``architecture_config`` / ``param_count``.
    model_source: AcquisitionSpec | None = None
    architecture_config: dict[str, Any] | None = None
    param_count: int | None = Field(default=None, ge=1)
    adapter: AdapterConfigIn | None = None
    #: ``auto`` resolves the card the run would be pinned to, by UUID, through
    #: ``core.gpu``. Never an index and never a bus address (ADR-008); this module names no
    #: device of its own and adds no entry to the hard-coded-GPU ratchet's allowlist.
    gpu_request: str = "auto"


@run_types_router.get("")
def list_run_types() -> dict[str, Any]:
    """The three registered run types, **from the live registry**.

    The reachability test asserts this answers over the transport and that all three appear.
    Removing a registration must shorten this body and turn the suite red (M12): the module
    would still parse, still be unit-tested and still be documented, and would exist for no
    caller.
    """
    entries = runtypes_service.run_types()
    return {"run_types": entries, "count": len(entries)}


@budget_router.post("/estimate")
def estimate_budget(payload: Annotated[BudgetRequest, Body()]) -> dict[str, Any]:
    """The estimate, the per-term breakdown and the free figure, without submitting a run.

    The same estimator the guard uses. ``free_bytes`` comes from the resolved card's free
    memory, or is ``None`` on a machine with no card - in which case the verdict is
    ``unknown`` rather than ``fits``.
    """
    device = resolve_submit_device(payload.gpu_request)
    free = device.free_memory_bytes if device is not None else None
    result = runtypes_service.estimate_budget(
        run_type=payload.run_type,
        cfg=payload.config,
        model_source=payload.model_source,
        architecture_config=payload.architecture_config,
        param_count=payload.param_count,
        adapter=payload.adapter,
        free_bytes=free,
        cards=_candidate_cards(payload.gpu_request, device),
    )
    return {
        **result,
        "gpu_uuid": None if device is None else device.uuid,
        "gpu_name": None if device is None else device.name,
    }


def _candidate_cards(gpu_request: str, device: Device | None) -> list[dict[str, Any]]:
    """Every card the run could be placed on, each with its free memory.

    ``auto`` may land on any visible card, so every one is listed and gets its own verdict;
    a named UUID lands on that card only. Read live from ``core.gpu``, by UUID - however
    many cards the node has. An unreadable driver lists none.
    """
    if device is None:
        return []
    if gpu_request != "auto":
        return [
            device_row(
                device,
                (
                    {device.uuid: device.free_memory_bytes}
                    if device.free_memory_bytes is not None
                    else {}
                ),
            )
        ]
    try:
        devices = gpu.list_devices()
    except Refusal:
        return [device_row(device, {})]
    try:
        free = gpu.free_memory_by_uuid()
    except Refusal:
        free = {}
    return [device_row(card, free) for card in devices]


@adapters_router.get("")
def list_adapters(
    session: Annotated[Session, Depends(session_dependency)],
    run_id: str | None = None,
) -> dict[str, Any]:
    """Indexed adapters, newest-first within a run. The index, which ``reindex`` rebuilds."""
    rows = runtypes_service.list_adapters(session, run_id=run_id)
    return {"adapters": rows, "count": len(rows)}


@adapters_router.get("/{name:path}")
def read_adapter(
    name: str,
    session: Annotated[Session, Depends(session_dependency)],
) -> dict[str, Any]:
    """One adapter's manifest **as written on disk**, its path, and a disagreement flag.

    The manifest, not the index row: the manifest is authoritative and where they disagree
    the manifest wins **and the disagreement is logged** (ADR-004). An endpoint serving the
    row would hide exactly the divergence an operator needs to see.
    """
    return runtypes_service.read_adapter(session, name)


@merges_router.get("")
def list_merges(
    session: Annotated[Session, Depends(session_dependency)],
    adapter_name: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> dict[str, Any]:
    """What happened to each merge: ``written``, ``refused`` with its code, or ``interrupted``.

    **This route exists because `mf_merges` was written, constrained and tested while nothing
    read it** (found 2026-10-14). A table recording an irreversible operation that an operator
    can only reach by hand-querying the database is the same unreachable capability as a
    function nothing calls - the **sixth** in this project's pattern, after four rebuilders in
    ``services/reindex.py`` and ``checkpoint_tree``.

    **It is the only record for one of the three outcomes.** For ``written`` and
    ``interrupted`` a merged checkpoint may exist and its own manifest is authoritative
    (ADR-004); a **refused** merge writes no checkpoint at all, so this table is the whole of
    what is known about it. An operator asking *"why did my merge not happen"* has nowhere
    else to look.

    ``limit`` is bounded at 500 rather than unbounded: a merge is a rare, deliberate operation,
    and an endpoint that will stream an entire table on request is a denial of service against
    the one database this installation has.
    """
    rows = merge_record.listing(session, adapter_name=adapter_name, limit=limit)
    return {"merges": rows, "count": len(rows)}


def enqueue_merge(name: str, run_id: str, step: int) -> Any:
    """Send the merge to the ``train`` queue.

    A module-level function rather than a route parameter, so a test replaces it by
    monkeypatching **this name** and the route keeps its own signature - a route parameter
    called ``enqueue`` would have been read by FastAPI as a query string, which is a
    public parameter nobody asked for.
    """
    from workers.merge_tasks import merge_adapter as task

    return task.delay(name, run_id, step)


@adapters_router.post("/{name:path}/merge")
@requires_approval_when_agent(
    "gpu_start",
    validate=approval_gates.validate_merge,
    summary=approval_gates.summarise_merge,
)
def merge_adapter(
    name: str,
    session: Annotated[Session, Depends(session_dependency)],
    run_id: Annotated[str, Query(min_length=1, max_length=64)],
    step: Annotated[int, Query(ge=0)],
) -> dict[str, Any]:
    """Request a merge. **Refused** here, with the reason, rather than queued blindly.

    A merge touches the card and the card is leased, so the arithmetic belongs on the
    ``train`` queue under the lease with the record as the truth (ADR-006, ADR-007). What
    this route does today is the half that is reachable without a card and without the run
    loop: it reads the adapter's manifest - which refuses an unrecognised ``scaling`` - and
    runs **AC17's unmerged-base check**, so a merge that cannot be performed is refused
    before anything is enqueued and the operator is told what to do next.

    **The enqueue is wired as of 2026-10-14** (F6's tasks 11.5 and 12.5, which were marked
    ``[?]`` naming that feature). ``mf_merges`` exists via migration ``0014`` and
    :mod:`services.merge_record` writes it; the merge itself runs as
    ``miforge.train.merge_adapter`` on the ``train`` queue, where the base model is in hand.
    Until then this route refused by name and enqueued nothing, which was the honest shape
    for a seam with nothing in it - a route that enqueued a task nothing consumed would be
    the unreachable-capability failure this project's review discipline exists to catch.

    **A preflight refusal writes no ``mf_merges`` row.** Recording one would invent a
    semantic no requirement states: R-02.45's three outcomes describe *the merge*, not a
    request for one. The row is written for a merge that was admitted and then either
    written or refused on the card.

    Args:
        run_id: where the merged checkpoint is written. A merged checkpoint is a checkpoint
            of the run that asked for it, so the run must be named - **not defaulted**, and
            not invented from the adapter's own run, which would write a merged checkpoint
            into the lineage of the run that produced the adapter.
        step: the step the merged checkpoint is written at.
    """
    admitted = runtypes_service.preflight_merge(session, name)

    enqueue_merge(name, run_id, step)

    return {
        "status": "accepted",
        "adapter": admitted,
        "run_id": run_id,
        "step": step,
        "note": (
            "The adapter's manifest is valid, its scaling rule is one this release "
            "implements, and a checkpoint of the unmerged base exists. The merge is "
            "enqueued on the `train` queue, where the base model is in hand; its outcome "
            "is recorded in mf_merges, including a refusal."
        ),
    }
