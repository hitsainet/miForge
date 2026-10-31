"""F6's chain routes: create, list, detail, stop  (R-02.27; FR-006.39 - .46).

Four routes, on their own ``/chains`` prefix, in a module of F6's own. **The whole feature
is four files** - these routes, :mod:`services.chains.service`,
:mod:`services.chains.tables` and :mod:`workers.chain_tasks` - because BRD §9 has not
decided whether chaining is a first-release feature or a script on top of single runs
(AMB-5), and FR-006.46 makes the implementation's *shape* the requirement: deleting those
four removes a feature without unpicking the loop.

## ``POST /chains`` validates **every** stage, and that is the point of the route

FR-006.39. A curriculum's fourth stage is validated now, by the **same** validator a single
run goes through, so it cannot be discovered invalid three hours in. The refusal carries
every bad stage with every one of its failures, for the reason FR-006.59 collects a single
run's: an operator fixing a configuration wants the list, not the first item.

## There is no advance route, and the absence is a requirement

FR-006.45: **one** advance path, the ``miforge.maintenance.chain_sweep`` sweep. An
"advance now" route for responsiveness would be a second path and a double-start race, and
the thing it would race with is the sweep that exists because a dying task's enqueue stalls
a chain forever. ``test_chain_routes_reachable.py`` asserts no such route is mounted rather
than reading this docstring.

## ``POST /chains/{id}/stop`` uses the single-run mechanism

AMB-3, FR-006.24. The running stage is stopped through
:func:`services.stop_service.request_stop` - F6 adds no second route, no second flag and no
second mechanism - and the chain is halted with ``chain_halted_on_operator_stop``, the code
:func:`schemas.chain.halt_reason` already gives that event rather than a second name for it.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query, status
from sqlalchemy.orm import Session

from api.agent_origin import requires_approval_when_agent
from api.deps import session_dependency
from schemas.chain import ChainSpec
from services import approval_gates
from services.chains import service as chain_service

log = logging.getLogger(__name__)

router = APIRouter(prefix="/chains", tags=["chains"])


# ---- F12: the approval gate's view of a chain (FR-012.11, .15, .18) -------------------
# Here rather than in `services/approval_gates.py` because FR-006.46 requires that deleting
# the chain modules removes the feature: no module outside them may reach the chain code.


def _chain_is_attach(values: dict[str, Any]) -> bool:
    """Only stage 0 may attach: every later stage must be ``load`` (the chains service)."""
    stages = values["spec"].stages
    return bool(stages) and (
        approval_gates.mode_of(stages[0].definition.get("model_source")) == approval_gates.ATTACH
    )


def _validate_chain(values: dict[str, Any], session: Any) -> list[dict[str, Any]]:
    """Every stage through the one validator, exactly as ``create`` does (FR-006.39)."""
    _definitions, resolutions = chain_service._validate_every_stage(session, values["spec"])
    return resolutions


def _summarise_chain(values: dict[str, Any], validated: Any) -> dict[str, Any]:
    from schemas.run_definition import RunDefinition

    spec = values["spec"]
    first = RunDefinition.model_validate(dict(spec.stages[0].definition))
    resolved = (validated or [None])[0]
    return {
        "route": "POST /api/v1/chains",
        "chain": spec.name,
        "stage_count": len(spec.stages),
        "stage_0": approval_gates.summarise_definition(first, resolved),
    }


DEFAULT_CHAIN_PAGE = 100
MAX_CHAIN_PAGE = 500


@router.post("", status_code=status.HTTP_202_ACCEPTED, summary="Create a chain")
@requires_approval_when_agent(
    "gpu_start",
    escalate=("hotupdate_attach", _chain_is_attach),
    validate=_validate_chain,
    summary=_summarise_chain,
)
def create_chain(
    spec: ChainSpec,
    session: Annotated[Session, Depends(session_dependency)],
) -> dict[str, Any]:
    """Validate every stage, record the plan, lease the card, start stage 0.

    ``202`` because the chain has been *accepted*, not *performed*: stage 0's parameters are
    acquired in the worker, under the chain's lease, on the ``train`` queue. Every later
    stage is started by the sweep.

    Raises:
        Refusal: ``chain_stage_invalid`` listing every stage that cannot be run as
            configured, each with its own failures and each failure carrying the **owning
            feature's code unchanged** (FR-006.59) - F7's ``pack_incomplete`` reaches the
            client as itself. Also ``gpu_lease_held`` when another run holds the card.
    """
    return chain_service.create(session, spec)


@router.get("", summary="Every chain, newest first")
def list_chains(
    session: Annotated[Session, Depends(session_dependency)],
    limit: Annotated[int, Query(ge=1, le=MAX_CHAIN_PAGE)] = DEFAULT_CHAIN_PAGE,
) -> dict[str, Any]:
    """The chain list with each chain's position, status and halt code."""
    return chain_service.listing(session, limit=limit)


@router.get("/{chain_id}", summary="One chain, with per-stage outcome and advance decision")
def read_chain(
    chain_id: str,
    session: Annotated[Session, Depends(session_dependency)],
) -> dict[str, Any]:
    """Per-stage outcome **and** the advance decision for every finished stage.

    The decision is computed by the same :func:`schemas.chain.advance_filter` the sweep
    uses. A screen rendering the rule a second way would be worse than no screen: it would
    show "this advances" beside a chain the sweep had halted.

    Raises:
        Refusal: ``chain_not_found``.
    """
    return chain_service.detail(session, chain_id)


@router.post("/{chain_id}/stop", summary="Halt a chain and stop its running stage")
def stop_chain(
    chain_id: str,
    session: Annotated[Session, Depends(session_dependency)],
    body: Annotated[dict[str, Any] | None, Body()] = None,
) -> dict[str, Any]:
    """Halt the chain; stop the running stage through the single-run mechanism.

    The optional body carries ``requested_by``, FR-006.24's audit label. It is **logged
    rather than stored** for the stage's run, exactly as a single run's stop is, because
    ``mf_runs`` carries ``stop_requested_by`` for the run and ``mf_chains`` has no column
    for the chain's - inventing one here would be a migration this task does not own, and
    the DDL is reported instead.

    Raises:
        Refusal: ``chain_not_found``; ``chain_not_stoppable`` on a chain that already
            finished - *a success for a stop that stopped nothing would tell the operator
            they had stopped a curriculum that had already ended.*
    """
    label = None if body is None else body.get("requested_by")
    return chain_service.stop_chain(session, chain_id, label=None if label is None else str(label))
