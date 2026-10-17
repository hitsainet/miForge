"""F6's routes: defining a run, validating one, and reading the loop's own record.

Five routes, on the ``/runs`` prefix, in a module of F6's own - the third router on that
prefix, deliberately. ``GET /runs/{run_id}`` is the foundation's and carries the liveness
block, F1's four acquisition routes are its own, F10's review block is its own, and two
handlers for one path is how a response loses a block that only one of them knew to add.

## ``POST /runs/validate`` returns **200 either way**

Asking whether a definition is valid is a **successful question**. The body says ``ok:
false`` and lists every failure; the refusals fire where an *action* is attempted. That is
the same shape F7 uses for pack validity, and it is what lets F9's New-run form show
problems beside the fields they belong to without having to parse an error envelope.

**And it is the same validator the create path calls.** There is exactly one
implementation. A second, laxer copy behind the submit button is the drift this project
pays for, and a validator the create path does not actually call is the
unreachable-capability failure ADR-016 #1 exists to catch - which is why **M-15** makes the
create path skip it and ``test_run_definition_reachability.py`` asserts the **call count**,
not "was called".

## ``POST /runs/define`` is R-02.53's create path, and F1's ``POST /runs`` is not

F1's route predates R-02.53 and submits the *acquisition half* - no pack, no stop
condition, no scoring - which is precisely the shape R-02.53 forbids. It is left alone
rather than broken, and it is **not** a second gate: a run with no recorded definition is
refused by name in the worker (``run_definition_missing``), so that route can create a row
and cannot produce a run that trains. One path reaches a running run, and the gate is on it.

## There is no stop route here  (FR-006.24, AMB-3)

F6 adds **no second route, no second flag and no second mechanism**. F1's
``POST /runs/{run_id}/cancel`` is the stop, and F6's contribution is the *condition* that
reads the column, the precedence, and the refusal on a terminal run - which that handler now
routes through :func:`services.stop_service.request_stop`.
"""

from __future__ import annotations

from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.orm import Session

from api.deps import session_dependency
from core.errors import refuse
from schemas.run_definition import RunDefinition
from services import run_launch
from training import run_log, stop, tasks
from training.stop.base import StopContext
from workers.run_record import RUNS

router = APIRouter(prefix="/runs", tags=["runs"])

#: Newest-first, matching the step log's order and the shape a long run needs.
DEFAULT_BOUNDARY_PAGE = 50
MAX_BOUNDARY_PAGE = 500


@router.post("/validate", summary="Validate a run definition without creating anything")
def validate_definition(
    definition: RunDefinition,
    session: Annotated[Session, Depends(session_dependency)],
) -> dict[str, Any]:
    """The same validator, read-only  (FR-006.62, E-25).

    Allocates nothing, creates no row, writes no run-log entry. Returns the resolved
    summary - the pinned versions, the budget estimate beside free VRAM, the projected
    footprint beside free disk, any named-absent service - or the **complete** failure
    list.

    **There is no ``dry_run`` flag on the create path.** A flag on one handler is one
    ``if`` away from a create that validated and did not create, or a validate that
    created; two routes over one validator cannot drift that way.
    """
    return run_launch.validate_only(definition, session=session)


@router.post(
    "/define",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Define and start a run (R-02.53)",
)
def define_run(
    definition: RunDefinition,
    response: Response,
    session: Annotated[Session, Depends(session_dependency)],
) -> dict[str, Any]:
    """Validate the whole definition, record it, lease the card, enqueue  (R-02.53).

    ``202`` because the run has been *accepted*, not *performed*: the parameters are
    acquired in the worker, under the lease, on the ``train`` queue. The request path never
    loads a model - that would allocate outside the lease and block the API for a
    multi-gigabyte read.

    Raises:
        Refusal: ``run_definition_invalid`` carrying **every** failure, each with its own
            named code (FR-006.59). Nothing was allocated and no row was left behind.
    """
    accepted = run_launch.launch(definition, session=session)
    response.status_code = status.HTTP_202_ACCEPTED
    return accepted


@router.get("/{run_id}/definition", summary="What this run was asked to do")
def read_run_definition(
    run_id: str,
    session: Annotated[Session, Depends(session_dependency)],
) -> dict[str, Any]:
    """The recorded definition, from the authoritative run log  (FR-006.61, FR-006.63).

    R-02.53's closing sentence: *"A started run has an identity and a recorded definition,
    so the run log says what was asked for and not only what happened."* Both halves are
    served - ``submitted`` and ``resolved`` - because what the operator asked for and what
    it resolved to are different facts, and collapsing them loses the ability to say *"you
    asked for auto and got this card"*.
    """
    return run_launch.read_definition(session, run_id)


@router.get("/{run_id}/stop-conditions", summary="What will stop this run, and how close")
def read_stop_conditions(
    run_id: str,
    session: Annotated[Session, Depends(session_dependency)],
) -> dict[str, Any]:
    """Every enabled condition with its target, its latest observed value and its headroom.

    **And ``observed_at_step``, which is the field that carries the honesty** (US-8): a
    test threshold observed forty steps ago is a different statement from one observed last
    step, and a view that showed only the number would render a stale figure as a live one.

    Evaluated live rather than read from the last boundary record, so the answer is current
    for a run between boundaries - and ``observed_at_step`` is what says which of the two
    numbers in the answer is current and which is from an earlier evaluation.
    """
    row = session.execute(
        sa.select(RUNS.c.id, RUNS.c.status, RUNS.c.last_step, RUNS.c.definition_resolved).where(
            RUNS.c.id == run_id
        )
    ).one_or_none()
    if row is None:
        raise refuse(
            "run_not_found",
            f"No run with id {run_id!r}. Check the identifier.",
            run_id=run_id,
        )
    record = dict(row._mapping)
    loop_block = (record["definition_resolved"] or {}).get("loop") or {}
    enabled: dict[str, Any] = loop_block.get("stop_conditions") or {}
    step = int(record["last_step"] or 0)

    ctx = StopContext(run_id=run_id, step=step, round=0, session=session)
    verdicts = stop.evaluate_all(enabled, ctx)
    return {
        "run_id": run_id,
        "status": record["status"],
        "step": step,
        # ADR-012: every number names its scale and sample. The exposure window is stated
        # beside the conditions because it is how many steps may follow an unobserved
        # breach - `retention_every - 1`, and 0 at the default of 1 (FR-006.20).
        "retention_exposure_steps": loop_block.get("retention_exposure_steps"),
        "conditions": [
            {
                **verdict.as_evaluation(),
                # The structured figures, beside the contract's four keys. The contract's
                # `StopConditionEvaluation` is closed to unknown keys, so these live on the
                # API response and NOT in the run log - stated so nobody tries.
                "values": dict(verdict.values),
            }
            for verdict in verdicts
        ],
        "precedence": list(stop.precedence.PRECEDENCE),
        "all_fired": list(stop.precedence.fired(verdicts)),
    }


@router.get("/{run_id}/boundaries", summary="This run's boundary records, newest first")
def list_boundaries(
    run_id: str,
    session: Annotated[Session, Depends(session_dependency)],
    limit: Annotated[int, Query(ge=1, le=MAX_BOUNDARY_PAGE)] = DEFAULT_BOUNDARY_PAGE,
) -> dict[str, Any]:
    """The boundary blocks from the **authoritative run log**  (ADR-004).

    Read from the log rather than from an index, because there is no boundary index: the
    boundary record has no table, so the log is not merely authoritative here, it is the
    only copy. A read of one run's log is a single-file scan, which is what the pagination
    bounds.
    """
    del session  # the log is the record; no index is consulted
    path = run_log.run_log_path(tasks.run_log_root(), run_id)
    if not path.is_file():
        # Refused by name rather than returning an empty page: an empty list is
        # indistinguishable from "this run has taken no boundaries yet", and those have
        # different fixes.
        raise refuse(
            "run_log_not_found",
            f"No run log for {run_id!r}. The log is the record (ADR-004); if the run "
            f"exists, its log directory is missing rather than empty.",
            run_id=run_id,
            path=str(path),
        )
    blocks = list(run_log.read_blocks(path, block="boundary"))
    page = list(reversed(blocks))[:limit]
    return {
        "run_id": run_id,
        "boundaries": page,
        # The page size and the total, so a reader never mistakes a page for the whole run.
        "count": len(page),
        "limit": limit,
        "total": len(blocks),
    }


# ======================================================================================
# The two registries the New run form must READ rather than transcribe
#
# **F9 built its stop-condition field from F6's documented precedence order**, marked in
# its own code as *"documentation, not a registry"*, and swapped in ``details.available``
# the moment a ``run_has_no_stop_condition`` refusal arrived. That is the right defensive
# shape and it is still a **transcription of prose** - which is the drift that produced
# ``DB_RUN_TYPE`` and the stale ``RUN_TYPES`` constant, both in this codebase, both within
# a fortnight.
#
# ``training.stop.names()`` and ``training.refresh.names()`` were one line each from being
# an endpoint. These are those lines. The form now reads the **live registries**, and
# deleting either registration is red in the form's own data rather than in a comment.
#
# A collection route each rather than one combined, matching F2's ``/signal-sources``,
# F3's ``/selectors`` and F4's ``/providers``: they are two registries and a combined
# payload would make "which registry is this name from" a question.
#
# **Not under ``/runs/``.** ``GET /runs/{run_id}`` would match ``stop-conditions`` as a run
# id, so a collection there depends on route-registration order - which is a dependency on
# the order of lines in a different file.
# ======================================================================================

registries_router = APIRouter(tags=["runs"])


@registries_router.get("/stop-conditions", summary="The five stop conditions, from the registry")
def list_stop_conditions() -> dict[str, Any]:
    """R-02.25's five, read from the **live registry**  (FR-006.15, FR-006.52).

    Answers with no run and no definition in hand, so the New run form can offer the set
    before anything exists. Each entry carries what the form actually needs:

    * ``configured`` - whether the condition takes parameters at all. ``retention_floor``
      and ``operator_stop`` take **none**: the floor lives in exactly one place and a
      parameter here would be a third layer nobody asked for (F4 FR-004.23 already needed
      a field to tell the two that exist apart).
    * ``required_parameters`` - for ``loss_plateau``, **all four**, because R-02.25 defines
      none of it and a default would decide when a run ends invisibly (DEF-4).
    * ``precedence_rank`` - lower is more significant, so a form can render the order
      without knowing the rule.

    The precedence tuple is served **beside** the list rather than implied by the list's
    order, because the list is in R-02.25's own order and the precedence is not the same
    sequence - and a client that inferred one from the other would be right today and
    wrong the first time either changed.
    """
    from training.stop import UNCONFIGURED, names, precedence, resolve
    from training.stop.loss_plateau import METRICS, REQUIRED_PARAMETERS
    from training.stop.loss_plateau import NAME as PLATEAU

    conditions = []
    for name in names():
        configured = name not in UNCONFIGURED
        entry: dict[str, Any] = {
            "condition": name,
            "configured": configured,
            "required_parameters": (
                list(REQUIRED_PARAMETERS)
                if name == PLATEAU
                else (
                    ["total"]
                    if name == "step_count"
                    else ["threshold"] if name == "test_threshold" else []
                )
            ),
            "precedence_rank": precedence.rank(name),
            "summary": (resolve(name).__doc__ or "").strip().splitlines()[0],
        }
        if name == PLATEAU:
            entry["metrics"] = list(METRICS)
        conditions.append(entry)

    return {
        "conditions": conditions,
        # ADR-012: the count names the scale, so a reader never mistakes a page for the
        # set. R-02.25's set is CLOSED at five and this says so rather than implying it.
        "count": len(conditions),
        "closed": True,
        "precedence": list(precedence.PRECEDENCE),
        # FR-006.17 / R-02.53: a run must enable at least one. Published so the form can
        # say so before the operator presses the button rather than only after.
        "minimum_enabled": 1,
    }


@registries_router.get("/refresh-modes", summary="The two refresh modes, from the registry")
def list_refresh_modes() -> dict[str, Any]:
    """R-02.26's two, read from the **live registry**  (FR-006.32).

    Served for the same reason as the conditions: the form was going to transcribe these
    too. ``applies_to`` is the one thing a form gets wrong otherwise - a refresh applies to
    **self-generation runs only** (R-02.26's opening clause), and it is **refused** rather
    than ignored elsewhere, because an ignored refresh leaves the operator believing the
    prompt set rotated when it did not.
    """
    from training.refresh import SELF_GENERATION, names, resolve

    modes = [
        {
            "mode": name,
            "applies_to": SELF_GENERATION,
            "summary": (resolve(name).__doc__ or "").strip().splitlines()[0],
        }
        for name in names()
    ]
    return {
        "modes": modes,
        "count": len(modes),
        "closed": True,
        "applies_to": SELF_GENERATION,
        # FR-006.37, AMB-4: a "round" is this cadence's worth of completed steps, and
        # `refresh_every` is required whenever a rule is configured - without it no round
        # would ever end and the refresh would never happen.
        "requires_cadence": "refresh_every",
    }


__all__ = ["registries_router", "router"]
