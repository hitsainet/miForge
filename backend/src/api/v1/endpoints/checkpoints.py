"""F5's eight routes  (FR-005.41, FR-005.43, FR-005.27).

**Addressed by ``(run_id, step)``, never by a surrogate id.** Database ids are reassigned
by a rebuild and foundation task 4.4 demands that every API response be *identical* across
one, so an exposed id would make ADR-004's central claim untestable. ``(run_id, step)`` is
manifest-derived and survives the index being dropped.

## Which reads go to disk and which go to the index

**Correctness reads go to a manifest** - one checkpoint, the lineage walk, verification.
**Speed reads over many rows go to the index** - the two listings. Where the two disagree
the manifest wins and the disagreement is logged, which
:func:`services.reindex.resolve_checkpoint` already implements; this module does not write a
second answer to that question.

## Path containment

``step`` is an ``int`` path parameter, so a traversal attempt is a **validation failure**
rather than a sanitised path, and ``run_id`` is resolved against its row with
:func:`workers.run_record.require_run` before any path is composed. Sanitising would compose
*some* path and operate on it, which is the silently-wrong answer ADR-013 forbids.

Two routes are deliberately **absent**: there is no export and no delete. PPRD section 7
names an "Export checkpoint" button and **no requirement defines the capability** (AMB-3);
a checkpoint is already a plain directory, so ``scp`` and ``tar`` are the current answer.
Nothing defines pruning, deletion or archival either. Inventing the noun before someone
asks for it is how this project got "packs" wrong twice.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from api.agent_origin import requires_approval_when_agent
from api.deps import session_dependency
from services import approval_gates
from services.checkpoints import manifest as manifest_module
from services.checkpoints import provider_source, requests
from services.checkpoints import resume as resume_module
from services.checkpoints.lineage import DEFAULT_LIMIT, walk
from services.checkpoints.paths import checkpoint_dir, iter_step_dirs, resolve_latest
from services.checkpoints.writer import verify_weights
from services.reindex import list_checkpoints
from training.tasks import run_log_root
from workers.run_record import require_run

log = logging.getLogger("miforge.api.checkpoints")

router = APIRouter(tags=["checkpoints"])


class ContinueRequest(BaseModel):
    """``POST /runs/{run_id}/continue``. Every field optional; the default is ``latest``."""

    model_config = ConfigDict(extra="forbid")

    step: int | None = Field(default=None, ge=0)
    #: The optimizer ``config_hash`` the resumed configuration produces, when the caller
    #: knows it. A difference restores anyway and is logged with both values (FR-005.34).
    optimizer_config_hash: str | None = None
    #: **Submit the continue, not only plan it** (D52, F11 FTASKS 18.0). Attached runs only;
    #: absent or false keeps F5's plan-only answer, so existing callers are unchanged.
    submit: bool = False


class BranchRequest(BaseModel):
    """``POST /runs/{run_id}/checkpoints/{step}/branch``.

    ``run_id`` is **required**: a branch is a new run and R-02.23's whole mechanism is that
    the new ``run_id`` is what makes it one. Generating one here would mean the operator
    could not name the thing they just created.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1, max_length=64)
    optimizer_config_hash: str | None = None


def _run_log_path(run_id: str) -> Path:
    from training.run_log import run_log_path

    return run_log_path(run_log_root(), run_id)


def _manifest_response(run_id: str, step: int) -> dict[str, Any]:
    directory = checkpoint_dir(run_id, step)
    doc = manifest_module.read(directory)
    return {
        # Read from disk, because the manifest is the authority (ADR-004). The index is
        # not consulted for this answer at all.
        "manifest": doc,
        "manifest_sha256": manifest_module.digest(doc),
        # FR-005.16: the provider, and WHICH SOURCE SAID SO. The label is the load-bearing
        # part - a step-record provider presented as the manifest's would be a claim the
        # artifact does not make.
        "provider_sources": provider_source.provider_sources(doc, log_path=_run_log_path(run_id)),
    }


@router.post(
    "/runs/{run_id}/checkpoints",
    status_code=202,
    summary="Request a save at the next step boundary",
)
def request_checkpoint(
    run_id: str, session: Annotated[Session, Depends(session_dependency)]
) -> dict[str, Any]:
    """Set the save-request flag. **202, because the save has not happened yet.**

    The loop honours it at the next step **boundary** - a save never interrupts a step - and
    it is never a Celery control message: every worker is ``--pool=solo -c 1``, so a busy
    worker never reads the control queue, which is exactly the worker a save request is for
    (ADR-006, ADR-007). Two requests for one boundary **collapse into one save** (E-12).
    """
    require_run(session, run_id)
    pending = requests.request_save(session, run_id)
    return {"run_id": run_id, "accepted": True, "pending_save": pending}


@router.get("/checkpoints", summary="Every indexed checkpoint, across runs")
def list_all_checkpoints(
    session: Annotated[Session, Depends(session_dependency)],
    run_id: Annotated[str | None, Query()] = None,
    domain: Annotated[str | None, Query()] = None,
    base_model: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> dict[str, Any]:
    """Index-backed, bounded by ``limit``. **One operator, so no cursor** (PPRD section 10).

    Served through :func:`services.reindex.list_checkpoints`, which is the seam foundation
    4.4's equivalence test compares. An endpoint building its own body from the ORM row
    would be **outside** that guarantee - the one thing worth knowing before wiring a route
    over this table.
    """
    rows = list_checkpoints(session, run_id=run_id)
    if domain is not None:
        rows = [r for r in rows if r["domain"] == domain]
    if base_model is not None:
        rows = [r for r in rows if r["base_model"] == base_model]
    return {"checkpoints": rows[:limit], "count": len(rows[:limit]), "total": len(rows)}


@router.get("/runs/{run_id}/checkpoints", summary="One run's checkpoints, newest first")
def list_run_checkpoints(
    run_id: str, session: Annotated[Session, Depends(session_dependency)]
) -> dict[str, Any]:
    """The run's checkpoints and **what ``latest`` resolves to**.

    ``latest`` comes from the symlink on disk rather than from the newest row: the pointer
    is what a copied directory carries and what a consumer follows, so reporting the index's
    opinion of it could disagree with the artifact.
    """
    require_run(session, run_id)
    rows = sorted(list_checkpoints(session, run_id=run_id), key=lambda r: r["step"], reverse=True)
    try:
        latest = resolve_latest(run_id)
    except Exception:  # a run with no checkpoint yet is normal, not an error
        latest = None
    return {
        "run_id": run_id,
        "checkpoints": rows,
        "latest_step": latest,
        # From disk, so a run whose index was dropped still reports its artifacts honestly.
        "steps_on_disk": [step for step, _ in iter_step_dirs(run_id)],
    }


@router.get(
    "/runs/{run_id}/checkpoints/{step}",
    summary="One checkpoint's manifest, read from disk",
)
def get_checkpoint(
    run_id: str, step: int, session: Annotated[Session, Depends(session_dependency)]
) -> dict[str, Any]:
    """The manifest as written, plus the labelled provider join (FR-005.16)."""
    require_run(session, run_id)
    return {"run_id": run_id, "step": step, **_manifest_response(run_id, step)}


@router.post(
    "/runs/{run_id}/checkpoints/{step}/verify",
    summary="Recompute weights_sha256 and compare",
)
def verify_checkpoint(
    run_id: str, step: int, session: Annotated[Session, Depends(session_dependency)]
) -> dict[str, Any]:
    """Recompute the digest over the bytes and compare it to the manifest.

    **Never repairs a manifest to match the bytes it found** (FR-005.27): the manifest is
    the record of what was written and the bytes are what is being questioned. A mismatch
    refuses as ``checkpoint_hash_mismatch`` naming both digests, having changed nothing.
    """
    require_run(session, run_id)
    directory = checkpoint_dir(run_id, step)
    doc = manifest_module.read(directory)
    expected = str(doc["weights_sha256"])
    actual = verify_weights(directory, expected)
    return {
        "run_id": run_id,
        "step": step,
        "match": True,
        "expected": expected,
        "actual": actual,
    }


@router.get(
    "/runs/{run_id}/checkpoints/{step}/lineage",
    summary="The parent chain, walked through manifests",
)
def get_lineage(
    run_id: str,
    step: int,
    session: Annotated[Session, Depends(session_dependency)],
    limit: Annotated[int, Query(ge=1, le=4096)] = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Follow ``parent`` by name to the root, or to where the record stops.

    ``terminated`` is ``root``, ``truncated`` or ``limit`` - three facts, never collapsed.
    **F9 computes indent depth from this chain**; no depth value is shipped, because depth
    is a property of a walk rather than of a node.
    """
    require_run(session, run_id)
    return {"run_id": run_id, "step": step, **walk(run_id, step, limit=limit).as_dict()}


@router.post("/runs/{run_id}/continue", summary="Continue a run from a checkpoint")
@requires_approval_when_agent(
    "hotupdate_attach",
    when=approval_gates.continue_submits,
    validate=approval_gates.validate_continue,
    summary=approval_gates.summarise_continue,
)
def continue_run(
    run_id: str,
    session: Annotated[Session, Depends(session_dependency)],
    response: Response,
    body: Annotated[ContinueRequest | None, Body()] = None,
) -> dict[str, Any]:
    """Resolve, **verify**, decide on trainer state - all before anything is submitted.

    Every refusal happens here, before a GPU lease is taken and before step 1 (F1's
    preflight rule, extended with F5's cases). Without ``submit`` this route returns the plan.

    **With ``submit: true`` it submits** (D52, F11 FTASKS 18.0; attached runs only) - through
    :func:`services.run_continue.submit_continue`, which reuses F6's ``run_launch.launch``
    (validator, definition record, lease, enqueue). So there is still **no second submission
    path, no lease logic and no GPU resolution here** (ADR-008): the route asks, the one
    launcher does.
    """
    status = require_run(session, run_id)
    # No body and an empty body are the same request: continue from `latest`. Built here
    # rather than as a default argument, because a call in a default is evaluated once at
    # import and shared by every request.
    options = body if body is not None else ContinueRequest()
    if options.submit:
        # THE SUBMISSION (D52). Its own module, because it is F1/F6's submission path reused
        # for a new attempt - the validator, the definition record, the lease, the enqueue -
        # and this route stays the place an operator asks. 202: the run is queued, not done.
        from services import run_continue

        submitted = run_continue.submit_continue(
            session,
            run_id,
            step=options.step,
            configured_config_hash=options.optimizer_config_hash,
        )
        response.status_code = 202
        return {**submitted, "run_status": "pending"}
    plan = resume_module.continue_plan(
        run_id,
        step=options.step,
        configured_config_hash=options.optimizer_config_hash,
    )
    log.info(
        "continue planned",
        extra={
            "code": "continue_planned",
            "run_id": run_id,
            "step": plan.step,
            "optimizer_state": plan.optimizer_state,
            "reason": plan.reason,
        },
    )
    return {"plan": plan.as_dict(), "run_status": status}


@router.post(
    "/runs/{run_id}/checkpoints/{step}/branch",
    summary="Branch a new run from any checkpoint",
)
def branch_run(
    run_id: str,
    step: int,
    body: Annotated[BranchRequest, Body()],
    session: Annotated[Session, Depends(session_dependency)],
) -> dict[str, Any]:
    """A new ``run_id``, ``parent`` = this checkpoint's name, ``base_model`` carried forward.

    A branch is **always** ``optimizer.state = "fresh"`` by construction, because trainer
    state is addressed by ``run_id`` and this run has a new one (R-02.15, FR-005.38). So the
    loss discontinuity R-02.22 exists to prevent is unavoidable on every branch as the
    requirements stand; it is recorded and logged, and no copy-forward of another run's
    trainer state is invented, because no requirement authorises one (AMB-2).
    """
    require_run(session, run_id)
    plan = resume_module.branch_plan(
        run_id,
        step,
        new_run_id=body.run_id,
        configured_config_hash=body.optimizer_config_hash,
    )
    log.info(
        "branch planned",
        extra={
            "code": "branch_planned",
            "run_id": body.run_id,
            "branched_from": plan.parent,
            "step": plan.step,
        },
    )
    return {"plan": plan.as_dict()}
