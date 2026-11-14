"""The chain orchestrator: create, advance, halt  (R-02.27; FR-006.39 - .46).

## A thin orchestrator with no privileged path  (FR-006.46, AMB-5)

BRD §9 asks whether chained runs are *"a first-release feature or a script on top of single
runs"* and does not answer. So this module does nothing a sequence of manual branches could
not do: it calls **the same** validator (:func:`services.run_definition.validate`, through
:func:`services.run_launch.launch`), **the same** checkpoint resolution
(:func:`services.checkpoints.resume.branch_plan`) and **the same** stop mechanism
(:func:`services.stop_service.request_stop`). It adds no second loader, no second submit
path and no second stop flag. Deleting this module, its two tables, the sweeper and the
four routes removes a feature; it does not unpick the loop.

## The order of the three checks before a hop, and why it is this order

1. **The advance filter** (:func:`schemas.chain.advance_filter`) - FR-006.43's third layer,
   **M-8's target**. First because it is the only one of the three whose wrong answer is a
   *correctness* failure rather than an outage: advancing past a floor breach starts the
   next stage from the checkpoint whose general capability had just collapsed and pushes it
   further in the same direction.
2. **The chain's lease** (E-17, FR-006.44). Before the checkpoint resolution, because a
   chain that has lost its card cannot start anything and reading a 2.4 GB manifest to find
   that out is work for nothing.
3. **The checkpoint** (E-16, FR-006.40). Last, because it is the only one that touches the
   filesystem.

## There is exactly one advance path  (FR-006.45)

:func:`sweep`, from ``miforge.maintenance.chain_sweep`` on the ``maintenance`` queue.
**The finishing stage's own task does not enqueue its successor**: a worker that died
between the terminal transition and the enqueue would stall the chain forever, and ADR-007's
rule is that the record is the protocol. Two advance paths would be two answers and a
double-start race.

## The double-start guard is a compare-and-swap, and that is a CORRECTION

F6's task 7.7 names ``uq_mf_chain_stages_chain_stage`` as the guard, with an
``IntegrityError`` read as *"already started"*. **That cannot be the guard at start time,
and the tables say so themselves**: ``mf_chain_stages.definition`` is ``NOT NULL`` and
carries R-02.53's definition *"validated at CHAIN CREATION ... so a curriculum's fourth
stage cannot be discovered invalid three hours in"*, and the ``started_iff_run`` CHECK
models a stage row that exists and has **not** started (``run_id`` and ``started_at`` both
null). So every stage's row already exists when a sweep arrives, no insert happens, and
nothing can raise an integrity error.

What the unique constraint does do is make the stage *plan* single-valued - a second row for
one ``(chain_id, stage_index)`` is impossible - and that is tested. The start-time guard is
:func:`_claim_stage`'s conditional ``UPDATE ... WHERE run_id IS NULL``, whose row count is
``0`` for the loser. It has **no window either**, for the same reason the constraint would
not: the test is in the ``WHERE`` clause, evaluated by the database under the row lock,
rather than in a ``SELECT`` this process hoped was still true.

*An unreachable ``except IntegrityError`` branch is deliberately NOT written here. F8's
duplicate check was deleted rather than kept "for safety" on exactly this ground: a guard
that cannot fire reads as protection while protecting nothing.*

## A later stage declares ``load`` and the chain supplies the path

``RunDefinition.model_source`` is required, and R-02.27 *determines* a hop's source - *"from
the last saved checkpoint"*. So a stage after the first must declare ``mode: load``, refused
by name at creation if it does not, and the chain replaces only the ``path``. The operator's
own stage definition stays verbatim in ``mf_chain_stages.definition``; what reaches
``launch`` is what the **chain** submitted for that run, which is what that run's
``definition_submitted`` then means. Replacing a ``create`` or an ``attach`` source would be
a silent override of the thing the operator chose.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import sqlalchemy as sa

from core.errors import Refusal, refuse
from schemas.chain import ChainSpec, advance_filter, assert_advanceable, halt_reason
from schemas.run_definition import RunDefinition
from services.chains.tables import CHAIN_TERMINAL, CHAINS, STAGES
from workers.run_record import RUNS, TERMINAL_STATES, as_aware, rowcount_of, utcnow

log = logging.getLogger(__name__)

#: How the chain appears in ``mf_leases.holder_run_id``. **Prefixed**, because that column
#: has no foreign key and a bare chain id there would read as a run id that matches no run
#: row - so a ``gpu_lease_held`` refusal would name a holder nobody could look up. FR-006.44
#: puts the lease on the *chain*, and this is what makes that visible in the refusal.
LEASE_HOLDER_PREFIX = "chain:"

#: The run statuses a stage must reach before the chain looks at it again.
_STAGE_TERMINAL = TERMINAL_STATES


def lease_holder(chain_id: str) -> str:
    """The ``holder_run_id`` a chain's lease carries. One spelling, three readers."""
    return f"{LEASE_HOLDER_PREFIX}{chain_id}"


# ===================================================================== creating a chain


def create(
    session: Any,
    spec: ChainSpec,
    *,
    chain_id: str | None = None,
    launch: Any | None = None,
    log_root: Path | None = None,
    now: datetime | None = None,
    **spies: Any,
) -> dict[str, Any]:
    """Validate **every** stage, record the plan, lease the card, start stage 0.

    FR-006.39: every stage is validated **now**, not at stage start, so a curriculum's
    fourth stage cannot be discovered invalid three hours in. That is the whole reason
    ``mf_chain_stages.definition`` exists and is ``NOT NULL``.

    Args:
        session: an open session. **The caller commits.**
        spec: the chain, at least two stages (a "chain" of one is a run).
        chain_id: supplied for a test; minted otherwise.
        launch: :func:`services.run_launch.launch`, injectable for a test.
        log_root: run-log root, forwarded to ``launch``.
        now: injectable clock.
        **spies: forwarded to the validator - F7's gate, F10's resolver, F1's budget hook
            and the optional-service check, so a test asserts payload and call count.

    Returns:
        :func:`detail`'s payload for the created chain.

    Raises:
        Refusal: ``chain_stage_invalid`` carrying **every** bad stage with **every** one of
            its failures. One refusal for the whole curriculum, not one per stage, for the
            same reason FR-006.59 collects a single run's failures: an operator fixing a
            configuration wants the list.
    """
    moment = now or utcnow()
    identity = chain_id or uuid4().hex
    launch_fn = launch if launch is not None else _live_launch()

    prepared, resolved_stages = _validate_every_stage(session, spec, **spies)

    session.execute(
        sa.insert(CHAINS).values(
            id=identity,
            name=spec.name,
            status="pending",
            current_stage=0,
            created_at=moment,
            updated_at=moment,
        )
    )
    for index, (stage, definition) in enumerate(zip(spec.stages, prepared, strict=True)):
        # FR-006.43's SECOND layer, at the insert. The `Literal` already excluded the three
        # and `advance_filter` re-asserts it at decision time; this catches a configuration
        # that arrived as plain strings - from JSON, from a migration, from a later edit
        # that widened the type.
        assert_advanceable(stage.advance_on)
        session.execute(
            sa.insert(STAGES).values(
                chain_id=identity,
                stage_index=index,
                definition=definition.model_dump(mode="json"),
                advance_on=list(stage.advance_on),
            )
        )
    session.flush()

    # ---- the chain's lease, from stage 0's own resolved budget -------------- FR-006.44
    # Through the first stage's validated budget rather than a second device resolution:
    # ADR-008 allows exactly one place to turn a request into a card, and `_live_budget`
    # already went there. `None` on a machine with no card (ADR-023), and the column stays
    # null, which is an assertion rather than an omission.
    budget = resolved_stages[0].get("budget") or {}
    gpu_uuid = budget.get("gpu_uuid")
    if gpu_uuid is not None:
        from core.gpu import Device
        from workers import lease as lease_module

        # The device was already resolved, ONCE, by the budget step - `device=` exists for
        # exactly this caller, as `lease.acquire`'s own docstring says: *"so a device is
        # resolved once per submit, not so one can be constructed by hand."* A second
        # `resolve_device` here would be a second place that turns a request into a card,
        # which ADR-008 forbids on the grounds that the rule is only true if there is one.
        held = lease_module.acquire(
            session,
            lease_holder(identity),
            prepared[0].gpu_request,
            device=Device(uuid=str(gpu_uuid), name=str(budget.get("gpu_name") or ""), index=0),
            now=moment,
        )
        gpu_uuid = held.gpu_uuid
        session.execute(sa.update(CHAINS).where(CHAINS.c.id == identity).values(gpu_uuid=gpu_uuid))

    log.info(
        "chain created",
        extra={
            "code": "chain_created",
            "chain_id": identity,
            "stages": len(spec.stages),
            "gpu_uuid": gpu_uuid,
        },
    )

    # ---- stage 0 starts here, and ONLY stage 0 -----------------------------------------
    # A chain that waited for the first sweep window to begin would look broken for a
    # minute after a 202. Advancing is the sweep's and only the sweep's (FR-006.45); the
    # first stage needs no advance decision and no checkpoint - its definition names its
    # own source.
    row = _stage_row(session, identity, 0)
    assert row is not None  # just inserted
    _start_stage(
        session,
        chain_id=identity,
        stage=row,
        model_source=None,
        launch_fn=launch_fn,
        log_root=log_root,
        now=moment,
        **spies,
    )
    session.execute(
        sa.update(CHAINS)
        .where(CHAINS.c.id == identity)
        .values(status="running", current_stage=0, updated_at=moment)
    )
    session.flush()
    return detail(session, identity)


def _validate_every_stage(
    session: Any, spec: ChainSpec, **spies: Any
) -> tuple[list[RunDefinition], list[dict[str, Any]]]:
    """Every stage through the one validator. Returns the definitions and their resolutions."""
    from pydantic import ValidationError

    from services import run_definition as validator_module

    definitions: list[RunDefinition] = []
    resolutions: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []

    for index, stage in enumerate(spec.stages):
        body = dict(stage.definition)
        if body.get("run_id") is not None:
            # A chain mints a run id per hop (FR-006.41). A stage naming its own would make
            # two stages able to collide on one run id, and the second would silently
            # UPDATE the first's row.
            bad.append(
                {
                    "stage_index": index,
                    "failures": [
                        {
                            "code": "chain_stage_invalid",
                            "message": (
                                "a chain stage must not name its own run_id: a hop is a "
                                "branch and the chain mints a new one"
                            ),
                            "field": "run_id",
                        }
                    ],
                }
            )
            continue
        try:
            definition = RunDefinition.model_validate(body)
        except ValidationError as exc:
            bad.append(
                {
                    "stage_index": index,
                    "failures": [
                        {
                            "code": "chain_stage_invalid",
                            "message": str(error["msg"]),
                            "field": ".".join(str(part) for part in error["loc"]),
                        }
                        for error in exc.errors()
                    ],
                }
            )
            continue

        if index > 0 and str(definition.model_source.get("mode") or "load") != "load":
            # See the module docstring: a hop's source is DETERMINED by R-02.27, so a later
            # stage declaring `create` or `attach` is a stage the chain cannot start -
            # refused rather than silently overridden.
            bad.append(
                {
                    "stage_index": index,
                    "failures": [
                        {
                            "code": "chain_stage_invalid",
                            "message": (
                                "a stage after the first starts from the previous stage's "
                                "last saved checkpoint, so its model_source must "
                                "declare mode 'load'; the chain supplies the path. Refused "
                                "rather than overridden: replacing the source you chose "
                                "would train something you did not ask for."
                            ),
                            "field": "model_source.mode",
                        }
                    ],
                }
            )
            continue

        outcome = validator_module.validate(definition, session=session, **spies)
        if not outcome.ok:
            payload = outcome.payload()
            bad.append({"stage_index": index, "failures": payload["failures"]})
            continue
        definitions.append(definition)
        resolutions.append(dict(outcome.resolved))

    if bad:
        raise refuse(
            "chain_stage_invalid",
            f"{len(bad)} of this chain's {len(spec.stages)} stages cannot be run as "
            f"configured. Every stage is validated now rather than at stage start, so a "
            f"curriculum's fourth stage is not discovered invalid three hours in. Fix the "
            f"stages listed and resubmit.",
            stages=bad,
            stage_count=len(spec.stages),
        )
    return definitions, resolutions


# ===================================================================== starting a stage


def _claim_stage(session: Any, stage_id: int, run_id: str, moment: datetime) -> bool:
    """Claim an unstarted stage for ``run_id``. ``False`` means **already started**.

    **The double-start guard** (FR-006.45, E-18). A conditional ``UPDATE`` with
    ``run_id IS NULL`` in the ``WHERE`` clause: the test is evaluated by the database under
    the row lock, so two sweeps racing produce one winner and one row count of ``0``. See
    the module docstring for why this is a compare-and-swap rather than the unique
    constraint task 7.7 names.
    """
    result = session.execute(
        sa.update(STAGES)
        .where(STAGES.c.id == stage_id, STAGES.c.run_id.is_(None))
        .values(run_id=run_id, started_at=moment)
    )
    return rowcount_of(result) == 1


def _start_stage(
    session: Any,
    *,
    chain_id: str,
    stage: sa.Row[Any],
    model_source: dict[str, Any] | None,
    launch_fn: Any,
    log_root: Path | None,
    now: datetime,
    **spies: Any,
) -> str | None:
    """Claim the stage and submit its run. ``None`` when another sweep claimed it first."""
    run_id = uuid4().hex
    if not _claim_stage(session, int(stage.id), run_id, now):
        log.info(
            "chain stage already started",
            extra={
                "code": "chain_stage_already_started",
                "chain_id": chain_id,
                "stage_index": int(stage.stage_index),
            },
        )
        return None

    body = dict(stage.definition)
    body["run_id"] = run_id
    if model_source is not None:
        body["model_source"] = model_source
    definition = RunDefinition(**body)

    launch_fn(
        definition,
        session=session,
        log_root=log_root,
        lease=_chain_lease(chain_id),
        **spies,
    )
    # The run's place in the curriculum. Written here rather than threaded through
    # `launch`: F6 owns both sides and `mf_runs.chain_membership_is_whole` requires the
    # pair to move together, which one statement guarantees and two do not.
    session.execute(
        sa.update(RUNS)
        .where(RUNS.c.id == run_id)
        .values(chain_id=chain_id, stage_index=int(stage.stage_index))
    )
    session.flush()
    log.info(
        "chain stage started",
        extra={
            "code": "chain_stage_started",
            "chain_id": chain_id,
            "stage_index": int(stage.stage_index),
            "run_id": run_id,
        },
    )
    return run_id


def _chain_lease(chain_id: str) -> Callable[..., str | None]:
    """The lease hook ``launch`` calls - **renew the chain's, never take a second**.

    FR-006.44: the chain holds the card for its whole duration. A stage taking a lease of
    its own would be refused ``gpu_lease_held`` by the chain's own lease, and releasing the
    chain's first would risk losing the card to another run between the two - *the
    curriculum would then stall at stage two of four with nothing able to say why.*
    """

    def hook(session: Any, run_id: str, gpu_request: str, gpu_uuid: str) -> str | None:
        from workers import lease as lease_module

        holder = lease_module.holder_of(session, gpu_uuid)
        expected = lease_holder(chain_id)
        if holder != expected:
            raise refuse(
                "chain_lease_lost",
                f"The chain no longer holds device {gpu_uuid} - it is held by "
                f"{holder!r}. The chain is halted rather than re-acquiring: a blind "
                f"re-acquire could land on a DIFFERENT card mid-curriculum, which silently "
                f"changes the device every later comparison was taken on. "
                f"Continue from the last stage's checkpoint "
                f"on a card you choose.",
                chain_id=chain_id,
                gpu_uuid=gpu_uuid,
                holder_run_id=holder,
                expected_holder=expected,
            )
        lease_module.renew(session, expected, gpu_uuid)
        return gpu_uuid

    return hook


def _live_launch() -> Any:
    from services import run_launch

    return run_launch.launch


# ===================================================================== advancing a chain


def candidates(session: Any) -> list[str]:
    """Chain ids a sweep should look at: every non-terminal chain.

    Deliberately **not** narrowed to "chains whose current stage is terminal" in SQL. The
    join would have to reach ``mf_runs`` from a table on a different ``MetaData`` and would
    encode the advance condition in two places - here and in :func:`advance_one` - which is
    the second-source-of-truth shape this project has paid for repeatedly. A sweep looks at
    a handful of rows once a minute; the query is not the cost.
    """
    rows = session.execute(
        sa.select(CHAINS.c.id)
        .where(CHAINS.c.status.notin_(CHAIN_TERMINAL))
        .order_by(CHAINS.c.created_at)
    ).all()
    return [str(row.id) for row in rows]


def advance_one(
    session: Any,
    chain_id: str,
    *,
    launch: Any | None = None,
    log_root: Path | None = None,
    checkpoint_root: Path | None = None,
    now: datetime | None = None,
    **spies: Any,
) -> dict[str, Any]:
    """Decide one chain's next move and take it. **The whole advance decision.**

    Returns:
        ``{"chain_id", "action", "reason", ...}``. ``action`` is one of ``none``,
        ``started``, ``completed``, ``halted`` - **always present**, because a missing key
        is indistinguishable from a healthy one.
    """
    moment = now or utcnow()
    launch_fn = launch if launch is not None else _live_launch()

    chain = _chain_row(session, chain_id, for_update=True)
    if chain is None:
        raise refuse(
            "chain_not_found",
            f"No chain with id {chain_id!r}. Check the identifier.",
            chain_id=chain_id,
        )
    if chain.status in CHAIN_TERMINAL:
        return _outcome(chain_id, "none", "chain_terminal", status=str(chain.status))

    index = int(chain.current_stage)
    stage = _stage_row(session, chain_id, index)
    if stage is None:
        # A chain whose plan is missing the stage it says it is on. Refused rather than
        # repaired: the plan is the record, and inventing a stage would invent a curriculum.
        raise refuse(
            "chain_stage_missing",
            f"Chain {chain_id!r} says it is on stage {index} and has no such stage row. "
            f"The plan is the record; nothing is invented here. Inspect the chain.",
            chain_id=chain_id,
            stage_index=index,
        )

    if stage.run_id is None:
        # AC-006.32's recovery shape: chain progress lives in the database, so a sweeper
        # with no memory of what was running starts the stage the rows say is current.
        started = _start_stage(
            session,
            chain_id=chain_id,
            stage=stage,
            # ``index - 1``, because the stage that FINISHED is the one before the current
            # one. This read ``index``, which resolved the unstarted stage's own (null) run
            # id and refused ``checkpoint_not_found`` for run ``None`` - found by the
            # recovery test rather than by reading, and it would have left a chain whose
            # process died between the chain row moving and the stage being claimed
            # unrecoverable by any sweep, with the sweep's own error counter as the only
            # sign.
            model_source=(
                None
                if index == 0
                else _hop_source(session, chain_id, index - 1, root=checkpoint_root)
            ),
            launch_fn=launch_fn,
            log_root=log_root,
            now=moment,
            **spies,
        )
        if started is None:
            return _outcome(chain_id, "none", "stage_already_started", stage_index=index)
        _touch(session, chain_id, moment)
        return _outcome(
            chain_id, "started", "stage_was_unstarted", stage_index=index, run_id=started
        )

    run = _run_row(session, str(stage.run_id))
    if run is None or str(run.status) not in _STAGE_TERMINAL:
        return _outcome(
            chain_id,
            "none",
            "stage_running",
            stage_index=index,
            run_id=str(stage.run_id),
            run_status=None if run is None else str(run.status),
        )

    reason = _effective_reason(str(run.status), run.stop_reason)

    # ---- 1. THE ADVANCE FILTER. M-8's target --------------------------------- FR-006.43
    if not advance_filter(reason, list(stage.advance_on or ())):
        code = halt_reason(reason)
        return halt(
            session,
            chain_id,
            code,
            detail=(
                f"Stage {index} (run {stage.run_id}) ended on {reason!r}, which this "
                f"stage's advance set {sorted(stage.advance_on or ())} does not admit. "
                f"A floor breach, an operator stop and a failed stage never advance a "
                f"chain, in any configuration."
            ),
            now=moment,
            stage_index=index,
            stop_reason=reason,
            advance_on=sorted(stage.advance_on or ()),
        )

    _finish_stage(session, int(stage.id), moment)

    if index + 1 >= _stage_count(session, chain_id):
        session.execute(
            sa.update(CHAINS)
            .where(CHAINS.c.id == chain_id)
            .values(status="completed", finished_at=moment, updated_at=moment)
        )
        session.flush()
        log.info(
            "chain completed",
            extra={"code": "chain_completed", "chain_id": chain_id, "stages": index + 1},
        )
        return _outcome(chain_id, "completed", reason, stage_index=index)

    # ---- 2. THE CHAIN'S LEASE, before any filesystem work --------- FR-006.44, E-17
    if chain.gpu_uuid is not None:
        from workers import lease as lease_module

        holder = lease_module.holder_of(session, str(chain.gpu_uuid))
        if holder != lease_holder(chain_id):
            return halt(
                session,
                chain_id,
                "chain_lease_lost",
                detail=(
                    f"The chain's lease on {chain.gpu_uuid} is held by {holder!r}, not by "
                    f"this chain. Halted rather than re-acquired: a blind re-acquire could "
                    f"land on a different card mid-curriculum."
                ),
                now=moment,
                stage_index=index,
                gpu_uuid=str(chain.gpu_uuid),
                holder_run_id=holder,
            )

    # ---- 3. THE CHECKPOINT, through F5 -------------------------- FR-006.40, .41, E-16
    next_stage = _stage_row(session, chain_id, index + 1)
    assert next_stage is not None  # the count check above
    try:
        source = _hop_source(session, chain_id, index, root=checkpoint_root)
    except Refusal as exc:
        if exc.code != "checkpoint_not_found":
            # A manifest that does not validate or a hash that does not verify is a disk
            # problem an operator can fix. Halting - which is terminal - would discard a
            # recoverable curriculum, so it is raised for the sweep to log and left for the
            # next sweep. Only "there is no checkpoint at all" is E-16's halt.
            raise
        return halt(
            session,
            chain_id,
            "chain_no_checkpoint",
            detail=(
                f"Stage {index} (run {stage.run_id}) produced no saved checkpoint, so there "
                f"is nothing for stage {index + 1} to start from. A chain hands each stage "
                f"the previous stage's checkpoint; starting the next stage from the base "
                f"model would discard the "
                f"completed stage without saying so. Give the stage a save cadence and run "
                f"it again."
            ),
            now=moment,
            stage_index=index,
            run_id=str(stage.run_id),
        )

    started = _start_stage(
        session,
        chain_id=chain_id,
        stage=next_stage,
        model_source=source,
        launch_fn=launch_fn,
        log_root=log_root,
        now=moment,
        **spies,
    )
    if started is None:
        return _outcome(chain_id, "none", "stage_already_started", stage_index=index + 1)

    session.execute(
        sa.update(CHAINS)
        .where(CHAINS.c.id == chain_id)
        .values(current_stage=index + 1, status="running", updated_at=moment)
    )
    session.flush()
    log.info(
        "chain advanced",
        extra={
            "code": "chain_advanced",
            "chain_id": chain_id,
            "from_stage": index,
            "to_stage": index + 1,
            "stop_reason": reason,
            "run_id": started,
        },
    )
    return _outcome(
        chain_id, "started", reason, stage_index=index + 1, run_id=started, source=source
    )


def _hop_source(
    session: Any, chain_id: str, finished_index: int, *, root: Path | None
) -> dict[str, Any]:
    """The next stage's ``model_source``, resolved **through F5**  (FR-006.40, .41).

    :func:`services.checkpoints.resume.branch_plan` is the one resolution: it verifies the
    manifest and the weights hash before anything is submitted, carries ``base_model``
    forward from the loaded checkpoint's own manifest - so a lineage walk does not terminate
    one hop short - and reports ``optimizer.state = "fresh"`` by construction, because
    trainer state is addressed by ``run_id`` and a hop takes a new one (FR-006.42, DEF-6).

    **F6 adds no second loader and no copy-forward of trainer state.** No requirement
    authorises one, so a curriculum carries R-02.22's loss discontinuity once per stage;
    that is DEF-6, recorded rather than fixed here.
    """
    from services.checkpoints.paths import resolve_latest
    from services.checkpoints.resume import branch_plan

    stage = _stage_row(session, chain_id, finished_index)
    assert stage is not None
    previous_run = str(stage.run_id)
    step = resolve_latest(previous_run, root=root)
    plan = branch_plan(previous_run, step, new_run_id=uuid4().hex, root=root)
    log.info(
        "chain hop resolved its checkpoint",
        extra={
            "code": "chain_hop_resolved",
            "chain_id": chain_id,
            "from_run_id": previous_run,
            "step": plan.step,
            "parent": plan.parent,
            "base_model": plan.base_model,
            # FR-006.42 / AC-006.30, logged because it is the cost a curriculum pays per
            # stage and nothing else would say it happened.
            "optimizer_state": plan.optimizer_state,
        },
    )
    return {"mode": "load", "path": str(plan.directory)}


def _effective_reason(status: str, stop_reason: Any) -> str:
    """The reason the advance filter sees, from a terminal run's row.

    ``failed`` is **not** one of R-02.25's five - it has an error code, not a stop reason -
    and :data:`schemas.chain.NEVER_ADVANCES` names it for exactly this conversion. A
    ``completed`` run with no recorded reason is treated the same way: a stage that cannot
    say why it stopped cannot be the basis for starting another.
    """
    if status == "failed":
        return "failed"
    if stop_reason is None:
        return "failed"
    return str(stop_reason)


# ===================================================================== halting, stopping


def halt(
    session: Any,
    chain_id: str,
    code: str,
    *,
    detail: str | None = None,
    now: datetime | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Halt the chain, **naming why**  (``halted_names_its_code``).

    ``halted`` is distinct from ``completed``: a curriculum that stopped early because a
    stage hit its retention floor did not finish, and reporting it as complete would be the
    silently-wrong answer ADR-013 forbids.
    """
    moment = now or utcnow()
    session.execute(
        sa.update(CHAINS)
        .where(CHAINS.c.id == chain_id)
        .values(
            status="halted",
            halt_code=code,
            halt_detail=detail,
            finished_at=moment,
            updated_at=moment,
        )
    )
    session.flush()
    log.warning(
        "chain halted",
        extra={"code": code, "chain_id": chain_id, "detail": detail, **extra},
    )
    return _outcome(chain_id, "halted", code, detail=detail, **extra)


def stop_chain(
    session: Any, chain_id: str, *, label: str | None = None, now: datetime | None = None
) -> dict[str, Any]:
    """``POST /chains/{id}/stop`` - halt the chain and stop its running stage.

    **Through the same mechanism as a single run** (FR-006.24, AMB-3): F6 adds no second
    stop route, no second flag and no second mechanism, so the running stage is stopped by
    :func:`services.stop_service.request_stop` exactly as an operator stopping that run
    alone would. The chain is halted with ``chain_halted_on_operator_stop`` - the code
    :func:`schemas.chain.halt_reason` already gives for an operator-stopped stage, rather
    than a second name for one event.

    Raises:
        Refusal: ``chain_not_found``; ``chain_not_stoppable`` on a chain that already
            finished - a success for a stop that stopped nothing would tell the operator
            they had stopped a curriculum that had already ended.
    """
    from services import stop_service

    moment = now or utcnow()
    chain = _chain_row(session, chain_id, for_update=True)
    if chain is None:
        raise refuse(
            "chain_not_found",
            f"No chain with id {chain_id!r}. Check the identifier.",
            chain_id=chain_id,
        )
    if chain.status in CHAIN_TERMINAL:
        raise refuse(
            "chain_not_stoppable",
            f"Chain {chain_id!r} is already {chain.status}, so there is nothing to stop. "
            f"Create a new chain, or branch from the last stage's checkpoint.",
            chain_id=chain_id,
            status=str(chain.status),
            halt_code=None if chain.halt_code is None else str(chain.halt_code),
        )

    stage = _stage_row(session, chain_id, int(chain.current_stage))
    stopped: dict[str, Any] | None = None
    if stage is not None and stage.run_id is not None:
        try:
            stopped = stop_service.request_stop(session, str(stage.run_id), label=label)
        except Refusal as exc:
            if exc.code != "run_not_stoppable":
                raise
            # The stage had already finished. That is not a reason to refuse the chain's
            # stop - the chain is still live and still needs halting - and it is reported
            # rather than swallowed silently.
            stopped = {"run_id": str(stage.run_id), "already_terminal": True}

    outcome = halt(
        session,
        chain_id,
        "chain_halted_on_operator_stop",
        detail=(
            f"An operator stopped this chain. Stage {chain.current_stage} was asked to stop "
            f"through the same cooperative mechanism a single run uses, so it stops at its "
            f"next step boundary."
        ),
        now=moment,
        stage_index=int(chain.current_stage),
        requested_by=label,
    )
    outcome["stage_stop"] = stopped
    return outcome


# ===================================================================== the sweep


def sweep(
    session_scope: Callable[[], AbstractContextManager[Any]],
    *,
    launch: Any | None = None,
    log_root: Path | None = None,
    checkpoint_root: Path | None = None,
    now: datetime | None = None,
    **spies: Any,
) -> dict[str, Any]:
    """**The one advance path**  (FR-006.45). One transaction per chain.

    Args:
        session_scope: a ``session_scope``-shaped factory. **A factory, not a session**,
            because one chain's failure must not take the sweep with it: a broken chain gets
            its own transaction rolled back and the next chain gets a clean one. That is the
            lesson ``services/reindex.py`` records in its own words - one subsystem killing
            the whole operation - and it applies to a sweep for the same reason.

    Returns:
        ``{"looked_at", "advanced", "halted", "completed", "errors", "actions"}``. Every key
        present, counts rather than booleans, so a sweep that did nothing is
        distinguishable from a sweep that could not run.
    """
    with session_scope() as session:
        ids = candidates(session)

    report: dict[str, Any] = {
        "looked_at": len(ids),
        "advanced": 0,
        "halted": 0,
        "completed": 0,
        "errors": 0,
        "actions": [],
    }
    for chain_id in ids:
        try:
            with session_scope() as session:
                outcome = advance_one(
                    session,
                    chain_id,
                    launch=launch,
                    log_root=log_root,
                    checkpoint_root=checkpoint_root,
                    now=now,
                    **spies,
                )
        # One chain must not take the sweep with it - see the docstring.
        except Exception:
            report["errors"] += 1
            # Logged with the chain named and the chain LEFT NON-TERMINAL on purpose. An
            # unexpected error is not evidence that a curriculum should be discarded, and
            # `halted` is terminal; the next sweep retries and an operator can stop it.
            log.exception(
                "chain sweep could not advance one chain",
                extra={"code": "chain_sweep_error", "chain_id": chain_id},
            )
            continue
        action = str(outcome["action"])
        if action == "started":
            report["advanced"] += 1
        elif action == "halted":
            report["halted"] += 1
        elif action == "completed":
            report["completed"] += 1
        report["actions"].append(outcome)
    log.info(
        "chain sweep",
        extra={
            "code": "chain_sweep",
            **{key: value for key, value in report.items() if key != "actions"},
        },
    )
    return report


# ===================================================================== reads


def listing(session: Any, *, limit: int = 100) -> dict[str, Any]:
    """``GET /chains`` - newest first, with each chain's position and outcome."""
    rows = session.execute(
        sa.select(CHAINS).order_by(CHAINS.c.created_at.desc()).limit(limit)
    ).all()
    return {"chains": [_chain_payload(session, row) for row in rows]}


def detail(session: Any, chain_id: str) -> dict[str, Any]:
    """``GET /chains/{id}`` - per-stage outcome **and the advance decision**.

    The advance decision is reported for every finished stage, computed by the same
    :func:`schemas.chain.advance_filter` the sweep uses - not a second rendering of the
    rule. A screen that showed one answer while the sweep took another would be worse than
    no screen.
    """
    chain = _chain_row(session, chain_id)
    if chain is None:
        raise refuse(
            "chain_not_found",
            f"No chain with id {chain_id!r}. Check the identifier.",
            chain_id=chain_id,
        )
    payload = _chain_payload(session, chain)
    payload["stages"] = [_stage_payload(session, row) for row in _stage_rows(session, chain_id)]
    return payload


def _chain_payload(session: Any, row: sa.Row[Any]) -> dict[str, Any]:
    return {
        "chain_id": str(row.id),
        "name": str(row.name),
        "status": str(row.status),
        "current_stage": int(row.current_stage),
        "stage_count": _stage_count(session, str(row.id)),
        "gpu_uuid": None if row.gpu_uuid is None else str(row.gpu_uuid),
        "halt_code": None if row.halt_code is None else str(row.halt_code),
        "halt_detail": None if row.halt_detail is None else str(row.halt_detail),
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
        "finished_at": _iso(row.finished_at),
    }


def _stage_payload(session: Any, row: sa.Row[Any]) -> dict[str, Any]:
    advance_on = sorted(row.advance_on or ())
    run: sa.Row[Any] | None = None if row.run_id is None else _run_row(session, str(row.run_id))
    status = None if run is None else str(run.status)
    raw_reason = None if run is None else run.stop_reason
    stop_reason = None if raw_reason is None else str(raw_reason)

    decision: dict[str, Any] | None = None
    if status is not None and status in _STAGE_TERMINAL:
        reason = _effective_reason(status, stop_reason)
        advances = advance_filter(reason, advance_on)
        decision = {
            "stop_reason": reason,
            "advances": advances,
            # Named even when it advances, so a reader sees WHICH rule answered.
            "halt_code": None if advances else halt_reason(reason),
        }
    return {
        "stage_index": int(row.stage_index),
        "run_id": None if row.run_id is None else str(row.run_id),
        "advance_on": advance_on,
        "started_at": _iso(row.started_at),
        "finished_at": _iso(row.finished_at),
        "run_status": status,
        "advance_decision": decision,
    }


# ===================================================================== row helpers


def _chain_row(session: Any, chain_id: str, *, for_update: bool = False) -> sa.Row[Any] | None:
    statement = sa.select(CHAINS).where(CHAINS.c.id == chain_id)
    if for_update:
        # `with_for_update` is a no-op on SQLite and the real thing on PostgreSQL. Both are
        # correct here: SQLite's single writer gives the same exclusion by construction.
        statement = statement.with_for_update()
    row: sa.Row[Any] | None = session.execute(statement).one_or_none()
    return row


def _stage_row(session: Any, chain_id: str, index: int) -> sa.Row[Any] | None:
    row: sa.Row[Any] | None = session.execute(
        sa.select(STAGES).where(STAGES.c.chain_id == chain_id, STAGES.c.stage_index == index)
    ).one_or_none()
    return row


def _stage_rows(session: Any, chain_id: str) -> Sequence[sa.Row[Any]]:
    rows: Sequence[sa.Row[Any]] = session.execute(
        sa.select(STAGES).where(STAGES.c.chain_id == chain_id).order_by(STAGES.c.stage_index)
    ).all()
    return rows


def _stage_count(session: Any, chain_id: str) -> int:
    return int(
        session.execute(
            sa.select(sa.func.count()).select_from(STAGES).where(STAGES.c.chain_id == chain_id)
        ).scalar_one()
    )


def _run_row(session: Any, run_id: str) -> sa.Row[Any] | None:
    session.expire_all()
    row: sa.Row[Any] | None = session.execute(
        sa.select(RUNS.c.id, RUNS.c.status, RUNS.c.stop_reason, RUNS.c.failure_code).where(
            RUNS.c.id == run_id
        )
    ).one_or_none()
    return row


def _finish_stage(session: Any, stage_id: int, moment: datetime) -> None:
    session.execute(
        sa.update(STAGES)
        .where(STAGES.c.id == stage_id, STAGES.c.finished_at.is_(None))
        .values(finished_at=moment)
    )


def _touch(session: Any, chain_id: str, moment: datetime) -> None:
    session.execute(
        sa.update(CHAINS).where(CHAINS.c.id == chain_id).values(status="running", updated_at=moment)
    )


def _outcome(chain_id: str, action: str, reason: str | None, **extra: Any) -> dict[str, Any]:
    return {"chain_id": chain_id, "action": action, "reason": reason, **extra}


def _iso(value: Any) -> str | None:
    aware = as_aware(value)
    return None if aware is None else aware.isoformat()


__all__ = [
    "LEASE_HOLDER_PREFIX",
    "advance_one",
    "candidates",
    "create",
    "detail",
    "halt",
    "lease_holder",
    "listing",
    "start_next",
    "stop_chain",
    "sweep",
]


def start_next(
    session: Any,
    chain_id: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """FR-006.40's named entry point. An alias for :func:`advance_one`.

    The task list and the FTDD both name ``start_next``; one function with two names would
    be two places a rule could change, so this *is* :func:`advance_one` and the advance
    decision is not duplicated. Kept because four documents cite the name.
    """
    return advance_one(session, chain_id, **kwargs)


# ------------------------------------------------------------------ purge (2026-10-20)

#: A chain in one of these states still needs every run it contains or names.
_ACTIVE_CHAIN_STATES: tuple[str, ...] = ("pending", "running")


def purge_blockers(
    session: Any, kind: str, item_id: str, mentions: Callable[[Any], bool]
) -> list[tuple[str, str, dict[str, Any]]]:
    """An ACTIVE chain blocks the purge of a run it contains or names, or of a part a stage
    uses. Registered with :mod:`services.purge.blockers` by this package's ``__init__``, so
    the purge service never imports chain code (FR-006.46)."""
    found: list[tuple[str, str, dict[str, Any]]] = []
    chains = session.execute(
        sa.select(CHAINS.c.id, CHAINS.c.name, CHAINS.c.status).where(
            CHAINS.c.status.in_(_ACTIVE_CHAIN_STATES)
        )
    ).all()
    for chain in chains:
        stages = session.execute(
            sa.select(STAGES.c.stage_index, STAGES.c.run_id, STAGES.c.definition).where(
                STAGES.c.chain_id == chain.id
            )
        ).all()
        for stage in stages:
            if (kind == "run" and stage.run_id == item_id) or mentions(stage.definition):
                found.append(
                    (
                        "chain_active",
                        f"Chain {chain.name} ({chain.id}) is {chain.status} and its stage "
                        f"{stage.stage_index} is, names or uses this {kind}.",
                        {"chain_id": str(chain.id)},
                    )
                )
                break
    return found
