"""Submitting a run, cancelling it, reading its steps, and listing the modes.

F1's four routes (FPRD section 7). ``GET /api/v1/runs/{run_id}`` is already served by
:mod:`api.v1.endpoints.runs` - the foundation's floor, carrying the liveness block - and is
deliberately **not** re-declared here: two handlers for one path is how a response loses a
block that only one of them knew to add.

## ``POST /runs`` returns 202 and does not load a model

It resolves the device, pre-flights, takes the lease and enqueues. Loading in the request
path would allocate **outside the lease** and block the API for a multi-gigabyte read, and
it would put the budget guard's decision after the allocation it precedes.

## The payload names exactly one source, and a payload naming two is refused

Not resolved by precedence. Picking one trains a parameter set the operator did not ask for
and reports success - the fabricated value ADR-018 forbids, at the scale of a whole run. The
rule lives in :class:`training.acquisition.base.AcquisitionSpec`'s validator, so the API and
the worker cannot disagree about it.

## ``POST /runs/{id}/cancel`` sets a flag

It never calls ``revoke(terminate=True)``: the solo pool has no child to signal, so the call
succeeds and nothing happens, and the operator sees a cancellation that was accepted. The
worker reads the column at a step boundary.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Annotated, Any, TypeVar
from uuid import uuid4

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from api.deps import session_dependency
from core.errors import refuse
from services import acquisition as acquisition_service
from training import acquisition
from training.acquisition.base import AcquisitionMode, AcquisitionSpec
from training.run_config import RunConfig

log = logging.getLogger(__name__)

router = APIRouter(tags=["runs"])

#: What :func:`_refusing_validation` builds. Generic so the two call sites keep their types
#: instead of widening to ``Any`` - which would make a typo in a field name invisible to MyPy
#: at exactly the boundary where every inbound value is still untrusted.
T = TypeVar("T")

#: Newest-first, which is both the mockup's step-log order and the shape a long run needs.
DEFAULT_STEP_PAGE = 100
MAX_STEP_PAGE = 1000


class RunSubmit(BaseModel):
    """The submit payload. Validated at the edge, refused rather than coerced.

    Closed to unknown keys: a misspelled ``gradient_accumulation`` must not silently leave
    the real field at its default, which would train a different run than was described.
    """

    model_config = ConfigDict(extra="forbid")

    mode: AcquisitionMode = AcquisitionMode.LOAD

    # Exactly one of these three groups. The rule is enforced by `AcquisitionSpec`, not
    # restated here, so there is one definition of it.
    path: str | None = None
    repo: str | None = None
    revision: str | None = None
    config: dict[str, Any] | None = None
    config_path: str | None = None

    #: ``auto`` or a ``GPU-...`` UUID. An index or a bus address is refused by name inside
    #: ``core.gpu``; this endpoint adds no device logic.
    gpu_request: str = "auto"

    #: R-02.13's four values. **No defaults for the three that change what is trained** -
    #: a default learning rate would train a model the operator did not choose a rate for.
    learning_rate: float
    warmup_steps: int
    batch_size: int
    gradient_accumulation: int = 1
    optimizer_state_precision: str = "8bit"

    #: The ``create`` run's seed (operator decision, 2026-10-14). Null is an assertion that
    #: no seed was set, and therefore that the starting hash is not reproducible.
    seed: int | None = None

    #: Supplied by the caller, or minted here. Accepted because R-02.22 continues a run by
    #: *starting miForge with the same ``run_id``*, so the identifier cannot always be the
    #: server's to choose.
    run_id: str | None = Field(default=None, min_length=1, max_length=64)

    def to_spec(self) -> AcquisitionSpec:
        """The in-process value object, with the refusals its validator owns."""
        return _refusing_validation(
            lambda: AcquisitionSpec(
                mode=self.mode,
                path=None if self.path is None else _as_path(self.path),
                repo=self.repo,
                revision=self.revision,
                config=self.config,
                config_path=None if self.config_path is None else _as_path(self.config_path),
                seed=self.seed,
                gpu_request=self.gpu_request,
            ),
            code="acquisition_source_invalid",
        )

    def to_config(self) -> RunConfig:
        """R-02.13's configuration, refused field by field and **never clamped**."""
        return _refusing_validation(
            lambda: RunConfig(
                learning_rate=self.learning_rate,
                warmup_steps=self.warmup_steps,
                batch_size=self.batch_size,
                gradient_accumulation=self.gradient_accumulation,
                optimizer_state_precision=self.optimizer_state_precision,  # type: ignore[arg-type]
                seed=self.seed,
            ),
            code="run_config_invalid",
        )


def _as_path(value: str) -> Any:
    from pathlib import Path

    return Path(value)


def _refusing_validation(build: Callable[[], T], *, code: str) -> T:
    """Turn a Pydantic ``ValidationError`` into ADR-013's envelope, naming the field.

    A raw validation error would reach the client as FastAPI's own 422 shape, which is a
    *different* envelope from every other refusal in this API - so a client would need two
    parsers, and ADR-013's one shape would hold for everything except the most common
    failure.
    """
    from pydantic import ValidationError

    try:
        return build()
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first.get("loc", ())) or "<payload>"
        raise refuse(
            code,
            f"{field}: {first.get('msg', 'is invalid')}. Refused rather than clamped to a "
            f"default - a substituted value trains a run the operator did not describe.",
            field=field,
            errors=[
                {"field": ".".join(str(p) for p in e.get("loc", ())), "message": e.get("msg")}
                for e in exc.errors()[:5]
            ],
        ) from exc


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED, summary="Submit a training run")
def submit_run(
    payload: RunSubmit,
    response: Response,
    session: Annotated[Session, Depends(session_dependency)],
) -> dict[str, Any]:
    """Validate, resolve the device, pre-flight, lease, enqueue. **Refuses before allocating.**

    Returns ``202`` with the acquisition summary F9's resource strip renders. The model is
    loaded in the worker, under the lease, on the ``train`` queue.
    """
    spec = payload.to_spec()
    cfg = payload.to_config()
    run_id = payload.run_id or uuid4().hex

    summary = acquisition_service.submit(session=session, run_id=run_id, spec=spec, cfg=cfg)
    response.status_code = status.HTTP_202_ACCEPTED
    return summary


@router.post(
    "/runs/{run_id}/cancel",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request cooperative cancellation",
)
def cancel_run(
    run_id: str, session: Annotated[Session, Depends(session_dependency)]
) -> dict[str, Any]:
    """Set the stop column and return ``202``. The worker lands it on a step boundary.

    ``202`` rather than ``200``: the stop has been *recorded*, not *performed*. A ``200``
    would tell the operator the run had stopped when what happened is that the request was
    accepted, and the difference is up to one step of training.

    **Routed through F6's stop service since 2026-10-14** (its FR-006.26, E-7). F6 adds no
    second route, no second flag and no second mechanism - F1 owns this route and the
    column, F6 owns the condition, the precedence and the terminal transition (its AMB-3).
    What the service adds is the one thing the primitive cannot express: a request against
    a **terminal** run is refused ``run_not_stoppable`` rather than accepted, because a
    202 for a stop that stopped nothing tells the operator something untrue.
    """
    from services import stop_service

    outcome = stop_service.request_stop(session, run_id)
    return {
        "run_id": run_id,
        # `cancel_pending` is kept as the wire name: it is what this route has always
        # returned and what F9 reads. Renaming it would be a second spelling of one fact.
        "cancel_pending": outcome["stop_pending"],
        # The timestamp, not just a boolean: it is what says whether the worker is shutting
        # down or ignoring the request, and a second press must not overwrite the first.
        "cancel_requested_at": outcome["cancel_requested_at"],
        "collapsed": outcome["collapsed"],
    }


@router.get("/runs/{run_id}/steps", summary="A run's step records, newest first")
def list_steps(
    run_id: str,
    session: Annotated[Session, Depends(session_dependency)],
    limit: Annotated[int, Query(ge=1, le=MAX_STEP_PAGE)] = DEFAULT_STEP_PAGE,
    before_step: Annotated[int | None, Query(ge=0)] = None,
) -> dict[str, Any]:
    """Paginated, newest first.

    Keyset pagination on the canonical step integer rather than an offset: a run writes rows
    while the operator is paging, and an offset would silently skip or repeat steps as the
    tail grows.
    """
    from models.run import Step

    table = Step.__table__
    query = sa.select(table).where(table.c.run_id == run_id)
    if before_step is not None:
        query = query.where(table.c.step < before_step)
    rows = session.execute(query.order_by(table.c.step.desc()).limit(limit)).all()

    steps = [dict(row._mapping) for row in rows]
    return {
        "run_id": run_id,
        "steps": [_step_payload(row) for row in steps],
        # ADR-012: every number names its scale and sample. The page size is stated so a
        # reader never mistakes a page for the whole run.
        "count": len(steps),
        "limit": limit,
        "next_before_step": steps[-1]["step"] if len(steps) == limit else None,
    }


def _step_payload(row: dict[str, Any]) -> dict[str, Any]:
    """One ``mf_steps`` row on the wire.

    The index's columns, not the log's full block. The two §5.10 timings have had columns
    since migration ``0011`` and are served since 2026-10-16 (F11 16.4: the run view's two
    extra series); both are ``null`` on every standalone step - the assertion that no bracket
    was held. *This docstring said they had no column until then.*
    """
    return {
        "step": row["step"],
        "loss": row.get("loss"),
        "grad_norm": row.get("grad_norm"),
        "learning_rate": row.get("learning_rate"),
        "wall_time_ms": row.get("wall_time_ms"),
        "batch_source_hashes": row.get("batch_source_hashes"),
        "quiesce_hold_ms": row.get("quiesce_hold_ms"),
        "commit_duration_ms": row.get("commit_duration_ms"),
    }


@router.get("/models/sources", summary="Which acquisition modes this build has")
def model_sources() -> dict[str, Any]:
    """Every acquisition mode, each unavailable one **named with its reason** (FR-001.15).

    An option missing from a response is indistinguishable from one that is broken
    (ADR-013), so ``attach`` appears here even in a build without the hot-update mode -
    reported as unavailable, with the capabilities it would have provided. Answers with no
    run in hand and touches neither the network nor the GPU.
    """
    return {
        "modes": acquisition.modes(),
        "default": acquisition.DEFAULT_MODE.value,
        # Published so a refused path is actionable before the operator types one.
        "model_roots": [str(path) for path in acquisition_service.model_roots()],
    }
