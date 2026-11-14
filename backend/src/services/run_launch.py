"""Defining and starting a run - **R-02.53's whole surface**  (FR-006.57-.65).

## The order, and the order is the requirement

::

    run_definition.validate(definition)   ALLOCATES NOTHING     FR-006.57, .58
      1. structural (Pydantic, at the edge)
      2. F7 require_valid_pack()                         <-- M-9
      3. part resolvability (re-resolved, uncached)
      4. cadences, the non-empty condition set, each condition, the refresh rule
      5. the computed budget vs free VRAM, BEFORE any allocation   <-- M-14
      6. optional-service availability, named if absent
    then, and only then:
      the projected footprint                                   FR-006.53
      the definition record + hash, DURABLE                      <-- M-16
      the mf_runs row, `pending`                                 FR-006.63
      F1: the GPU lease                                          step 7, F1's, LAST
      enqueue miforge.train.run_step_loop                        -> `train`

**Steps 1-6 allocate nothing and this function's validator call takes no lease** -
``test_definition_allocates_nothing.py`` asserts that on the lease spy's call count,
because *a test that only checks the response passes against a validator that took and
released a lease.*

**The definition record is durable before the enqueue** (FR-006.61, FR-006.63). **M-16**
writes it after: a worker can then start a run whose definition is not yet durable, so a
crash in that window leaves a running run nothing can state the shape of.

## There is exactly one path to a run that trains

F1's ``POST /api/v1/runs`` predates R-02.53 and submits the *acquisition half* with no
pack, no stop condition and no scoring - which is precisely the shape R-02.53 forbids. It
is left in place and is not a second gate, because
:func:`training.tasks.run_step_loop` **refuses by name** (``run_definition_missing``) a run
whose definition was never recorded. So the ungated route can create a row and **cannot
produce a run that trains**, and the gate sits on the one path that can. Fail closed where
correctness is at stake (ADR-013).
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

import sqlalchemy as sa

from core.canonical import sha256_of
from core.errors import refuse
from packs.registry import row_id as pack_row_id
from schemas.run_definition import RunDefinition
from services import footprint as footprint_module
from services import run_definition as validator_module
from training import run_log
from workers.run_record import RUNS, utcnow

log = logging.getLogger(__name__)

#: The ``definition`` block's discriminator in ``step-record-v1.json``. F6 writes this block
#: and the schema says so in its own description.
DEFINITION_BLOCK = "definition"


def validate_only(definition: RunDefinition, *, session: Any, **spies: Any) -> dict[str, Any]:
    """``POST /runs/validate`` - **the same validator, read-only**  (FR-006.62, M-15).

    Allocates nothing, creates no row, writes no run-log entry. Returns the resolved
    summary or FR-006.59's complete failure list, **as a 200 either way**: asking whether a
    definition is valid is a successful question, and the refusals fire where an action is
    attempted (the same shape F7 uses for pack validity).

    **There is exactly one implementation and the create path calls it.** A second, laxer
    copy behind the submit button is the drift this project pays for, and a validator the
    create path does not actually call is the unreachable-capability failure ADR-016 #1
    exists to catch.
    """
    outcome = validator_module.validate(definition, session=session, **spies)
    payload = outcome.payload()
    payload["footprint"] = _footprint_of(definition, outcome).payload()
    return payload


def launch(
    definition: RunDefinition,
    *,
    session: Any,
    enqueue: Any | None = None,
    lease: Any | None = None,
    log_root: Any = None,
    extra_resolved: dict[str, Any] | None = None,
    **spies: Any,
) -> dict[str, Any]:
    """Validate, record, lease, enqueue. Returns the accepted run's summary.

    Args:
        definition: R-02.53's definition, structurally valid already.
        session: the request-scoped session. **The caller commits** - which is what makes
            the row and the index entry appear together or not at all.
        enqueue: override the enqueue call, for tests. ``None`` sends the real task.
        lease: override the lease acquisition, for the allocates-nothing test's spy.
        log_root: where run logs live. ``None`` uses the configured root.
        extra_resolved: blocks merged into ``resolved`` before it is hashed and recorded -
            the continue route's ``resume`` record (D52), which is how a resume's
            discontinuity reaches the run log's definition block. Keys the validator
            produced are never overwritten.

    Raises:
        Refusal: ``run_definition_invalid`` carrying **every** failure (FR-006.59).
            Nothing was allocated and no row was created on that path.
    """
    # ---- 1-6. THE VALIDATOR. One call, and M-15 deletes it ------------- FR-006.62
    outcome = validator_module.validate(definition, session=session, **spies)
    outcome.raise_if_invalid()

    run_id = definition.run_id or uuid4().hex
    resolved = dict(outcome.resolved)
    for key, value in (extra_resolved or {}).items():
        if key in resolved:
            raise refuse(
                "run_definition_resolved_key_taken",
                f"{key!r} is already a key of the resolved definition; a caller's extra block "
                f"may add to the record but never replace what the validator resolved.",
                key=key,
            )
        resolved[key] = value

    # ---- the projected footprint, reported and enforcing nothing -------- FR-006.53
    projection = _footprint_of(definition, outcome)
    resolved["footprint"] = projection.payload()

    # ---- F4 reads its own block from `definition_resolved[SCORING_KEY]` -------------
    # Written here so F4's evaluator finds what F6 recorded. The key is imported from F4
    # rather than respelled: two spellings of it is a scoring block the evaluator never
    # finds, and the symptom is a run with no scores and no error.
    resolved[validator_module.SCORING_KEY] = _scoring_block(definition, resolved)

    submitted = definition.model_dump(mode="json")
    # THE HASH. `DefinitionBlock.definition_sha256` is described by the contract as the
    # "canonical hash of `resolved`", so that is what is hashed - the contract wins over
    # this feature's own FR-006.61, which calls it the submitted definition's identity.
    # Both identities are kept: the submitted one rides inside `resolved` as
    # `submitted_sha256`, which `resolved` being a free-form object allows. The discrepancy
    # is reported rather than resolved by editing a vendored schema.
    definition_sha256 = sha256_of(resolved)

    # ---- the definition record, DURABLE BEFORE THE ENQUEUE ------- FR-006.61, .63, M-16
    _record_definition(
        session,
        run_id=run_id,
        submitted=submitted,
        resolved=resolved,
        definition_sha256=definition_sha256,
        run_type=definition.run_type,
        gpu_request=definition.gpu_request,
        # `mf_runs.pack_id` references `mf_packs.id`, which is one row per VERSION -
        # `registry.row_id(name, version)`, e.g. `notary-md@1` - and `runs_using` and the
        # delete guard both read it that way. Until 2026-10-18 this wrote the bare pack id,
        # so every run defined from a pack failed its own insert on the foreign key (a 500)
        # after passing the whole validator. No test reached it: the launch tests use loose
        # parts or a stubbed gate. Found by the first live run of the New run form's
        # pickers (F9 task 14.0); pinned by `test_define_from_a_committed_pack.py`.
        pack_id=(
            pack_row_id(definition.pack.pack_id, definition.pack.version)
            if definition.pack is not None
            else None
        ),
        log_root=log_root,
    )

    # ---- step 7: F1's device resolution and the GPU lease, LAST ----------- FR-006.58
    # Last because it is the first step that takes something away from another run. The
    # budget check already ran, at step 5, against free VRAM - so a run that does not fit
    # never takes the only card to find out.
    gpu_uuid = _acquire_lease(session, run_id, definition, resolved, lease)
    if gpu_uuid is not None:
        session.execute(sa.update(RUNS).where(RUNS.c.id == run_id).values(gpu_uuid=gpu_uuid))

    # ---- COMMIT, THEN ENQUEUE ---------------------------------------------------------
    # The run row, its definition and its lease must be VISIBLE to the worker before the
    # message can reach it. The request session committed in the dependency's teardown -
    # AFTER this returned - so a warm worker read the run before it existed, refused, and
    # acked the message away: the run sat `pending` forever with nothing queued. A cold
    # worker hid it (importing torch took longer than the commit). Found live 2026-10-17,
    # F11 window 2: the second run of the session never started.
    session.commit()

    # ---- the enqueue ---------------------------------------------------------------
    if enqueue is None:
        from training.tasks import run_step_loop

        enqueue = run_step_loop.delay
    enqueue(run_id)

    log.info(
        "run defined and accepted",
        extra={
            "code": "run_defined",
            "run_id": run_id,
            "run_type": definition.run_type,
            "definition_sha256": definition_sha256,
            "stop_conditions": sorted(definition.enabled_conditions()),
            "gpu_uuid": gpu_uuid,
            "seed": resolved.get("seed"),
            "projected_footprint": projection.rendered(),
        },
    )
    return {
        "run_id": run_id,
        "status": "pending",
        "definition_sha256": definition_sha256,
        "resolved": resolved,
        "footprint": projection.payload(),
        "may_be_more_to_fix": outcome.may_be_more,
    }


def read_definition(session: Any, run_id: str, *, log_root: Any = None) -> dict[str, Any]:
    """``GET /runs/{run_id}/definition`` - **read from the run log**  (FR-006.61).

    The log is authoritative (ADR-004) and ``mf_runs``' three definition columns are an
    index over it, so this reads the log and falls back to the index only when the log file
    is absent - reporting which source answered, so a disagreement is visible rather than
    smoothed over.

    Raises:
        Refusal: ``run_not_found``; ``run_definition_missing`` for a run created by a path
            that recorded none (F1's pre-R-02.53 ``POST /runs``).
    """
    row = session.execute(
        sa.select(
            RUNS.c.id,
            RUNS.c.definition_submitted,
            RUNS.c.definition_resolved,
            RUNS.c.definition_sha256,
        ).where(RUNS.c.id == run_id)
    ).one_or_none()
    if row is None:
        raise refuse(
            "run_not_found",
            f"No run with id {run_id!r}. Check the identifier.",
            run_id=run_id,
        )
    record = dict(row._mapping)

    path = _log_path(run_id, log_root)
    if path.is_file():
        blocks = list(run_log.read_blocks(path, block=DEFINITION_BLOCK))
        if blocks:
            block = blocks[-1]
            return {
                "run_id": run_id,
                "source": "run_log",
                "submitted": block["submitted"],
                "resolved": block["resolved"],
                "definition_sha256": block["definition_sha256"],
                # A disagreement between the authoritative log and the index is reported,
                # not reconciled silently: the manifest/log wins and the disagreement is
                # logged (ADR-004, foundation 4.5).
                "index_agrees": record["definition_sha256"] == block["definition_sha256"],
            }

    if record["definition_resolved"] is None:
        raise refuse(
            "run_definition_missing",
            f"Run {run_id!r} has no recorded definition. A run needs one before it "
            f"is enqueued, so this row was created by a path that predates it - "
            f"POST /api/v1/runs, which submits the acquisition half only. Define the run "
            f"through POST /api/v1/runs/define.",
            run_id=run_id,
        )
    return {
        "run_id": run_id,
        "source": "index",
        "submitted": record["definition_submitted"],
        "resolved": record["definition_resolved"],
        "definition_sha256": record["definition_sha256"],
        "index_agrees": True,
    }


# --------------------------------------------------------------------------- internals


def _record_definition(
    session: Any,
    *,
    run_id: str,
    submitted: dict[str, Any],
    resolved: dict[str, Any],
    definition_sha256: str,
    run_type: str,
    gpu_request: str,
    pack_id: str | None,
    log_root: Any,
) -> None:
    """The run log first (authoritative), then the row (the index)  (ADR-004, M-16).

    In that order, and the order is the point: the log is the record and the row is a cache
    over it, so a crash between the two leaves the authoritative copy present and the index
    rebuildable - never the reverse.
    """
    path = _log_path(run_id, log_root)
    run_log.append(
        path,
        {
            "block": DEFINITION_BLOCK,
            "run_id": run_id,
            # Null, and ONLY on this block: it is written before step 0 exists, which the
            # schema says in so many words.
            "step": None,
            "ts": utcnow().isoformat(),
            "submitted": submitted,
            "resolved": resolved,
            "definition_sha256": definition_sha256,
        },
    )

    exists = session.execute(sa.select(RUNS.c.id).where(RUNS.c.id == run_id)).one_or_none()
    values = {
        "definition_submitted": submitted,
        "definition_resolved": resolved,
        "definition_sha256": definition_sha256,
        "updated_at": utcnow(),
    }
    if exists is None:
        session.execute(
            sa.insert(RUNS).values(
                id=run_id,
                # `pending`. **Settled 2026-10-14 and `queued` stops existing**: it is
                # what the CHECK permits and the column defaults to, what `RUN_STATUSES`,
                # the janitor and every test fixture already say, and the project's rule
                # is that the constraint is the authority. FR-006.13's `queued` is a
                # documentation defect, reported - F9 found the cost of leaving both,
                # which was a freshly submitted run rendering as "pending (unrecognised)".
                status="pending",
                run_type=run_type,
                gpu_request=gpu_request,
                pack_id=pack_id,
                **values,
            )
        )
    else:
        session.execute(sa.update(RUNS).where(RUNS.c.id == run_id).values(**values))
    # Flushed so the row and its FK are real before the lease is taken against it - a lease
    # for a run the database does not have yet would fail on the constraint, after the
    # definition was already durable.
    session.flush()


def _acquire_lease(
    session: Any,
    run_id: str,
    definition: RunDefinition,
    resolved: dict[str, Any],
    lease: Any | None,
) -> str | None:
    """F1's lease, through F1's own module. ``None`` on a machine with no card (ADR-023)."""
    budget = resolved.get("budget") or {}
    gpu_uuid = budget.get("gpu_uuid")
    if gpu_uuid is None:
        return None
    if lease is not None:
        held = lease(session, run_id, definition.gpu_request, gpu_uuid)
        return None if held is None else str(held)

    from core.gpu import Device
    from workers import lease as lease_module

    device = Device(uuid=str(gpu_uuid), name=str(budget.get("gpu_name") or ""), index=0)
    return lease_module.acquire(session, run_id, definition.gpu_request, device=device).gpu_uuid


def _footprint_of(
    definition: RunDefinition, outcome: validator_module.Outcome
) -> footprint_module.Footprint:
    """The projection, from the ``step_count`` total when there is one (FR-006.53)."""
    enabled = outcome.resolved.get("loop", {}).get("stop_conditions") or {}
    budget = outcome.resolved.get("budget") or {}
    return footprint_module.project(
        max_steps=footprint_module.steps_for_footprint(enabled, fallback=0),
        save_every=definition.cadences.save_every,
        param_count=budget.get("param_count"),
        optimizer_state_precision=str(
            definition.run_configuration.get("optimizer_state_precision") or "8bit"
        ),
    )


def _scoring_block(definition: RunDefinition, resolved: dict[str, Any]) -> dict[str, Any]:
    """F4's block, assembled from the definition and the pack's resolved slots.

    ``pack_retention_floor`` and ``retention_floor_override`` are passed as the **two
    layers F4's ``resolve_floor`` already models**, so the layer that supplied the floor is
    recorded by F4 rather than decided here. F6 picks no floor.
    """
    pack = resolved.get("pack") or {}
    block = dict(definition.scoring)
    block.setdefault("pack_retention_floor", pack.get("retention_floor"))
    if definition.retention_floor_override is not None:
        block["retention_floor_override"] = definition.retention_floor_override
    return block


def _log_path(run_id: str, log_root: Any) -> Any:
    from training.tasks import run_log_root

    return run_log.run_log_path(run_log_root(log_root), run_id)


__all__ = [
    "DEFINITION_BLOCK",
    "last_logged_step",
    "launch",
    "read_definition",
    "rebuild_definition_index",
    "terminal_outcome",
    "validate_only",
]


def rebuild_definition_index(session: Any, *, log_root: Any, run_id: str) -> int:
    """Rebuild one run's three definition columns from the **run log**  (ADR-004; task 9.3).

    The log is authoritative and ``mf_runs.definition_submitted`` / ``_resolved`` /
    ``_sha256`` are an index over its ``definition`` block, so they are reconstructible -
    which is what ADR-004 means by *"the database is a rebuildable index"* and what makes a
    lost or corrupted row recoverable without losing what the run was asked to do.

    Returns:
        ``1`` when a definition block was found and the row updated, ``0`` when the run has
        no log or no definition block.

    **It is wired into :func:`services.reindex._rebuild_run_log_indexes`, and that wiring is
    the point.** ``mf_steps``, ``mf_batches`` and ``mf_scores`` each shipped with a
    rebuilder **nothing called** - three features independently reported the same one-line
    gap, each rightly declining to edit a file that was not theirs, which is exactly how
    they accumulated. A fourth would be the same defect with the lesson already written
    down, so this one is called from the one command, and
    ``test_definition_reindex_wiring.py`` asserts the command **iterates** it rather than
    that it exists - *assert the wiring, not the existence.*

    **Three columns and not the row.** ``mf_runs`` is a *record* table (its status, its
    heartbeat and the gap between its beats are nowhere in any log), so ``drop_index`` does
    not empty it and this does not recreate a missing row. It repairs the three columns that
    **are** an index, and leaves the record alone.
    """
    path = run_log.run_log_path(log_root, run_id)
    if not path.is_file():
        return 0
    blocks = list(run_log.read_blocks(path, block=DEFINITION_BLOCK))
    if not blocks:
        return 0
    block = blocks[-1]
    result = session.execute(
        sa.update(RUNS)
        .where(RUNS.c.id == run_id)
        .values(
            definition_submitted=block["submitted"],
            definition_resolved=block["resolved"],
            definition_sha256=block["definition_sha256"],
            updated_at=utcnow(),
        )
    )
    rebuilt = 1 if (getattr(result, "rowcount", 0) or 0) > 0 else 0
    if rebuilt:
        log.info(
            "definition index rebuilt from the run log",
            extra={
                "code": "definition_index_rebuilt",
                "run_id": run_id,
                "definition_sha256": block["definition_sha256"],
            },
        )
    return rebuilt


def last_logged_step(run_id: str, *, log_root: Any = None) -> int | None:
    """The highest step in the run's **authoritative log**, or ``None``  (ADR-004).

    ``mf_runs.last_step`` has no production writer, so ``GET /runs/{id}`` served ``null``
    for every run - including one that trained 650 steps (found live, F11 window 2,
    2026-10-18) - while the log a directory away held every step. The ``mf_steps`` index is
    not the answer either: it is committed with the heartbeat, so a crash loses up to a
    minute of rows; the log is written FIRST (``StepWriter``), before anything else.
    """
    path = _log_path(run_id, log_root)
    if not path.is_file():
        return None
    steps = [int(block["step"]) for block in run_log.read_blocks(path, block="step")]
    return max(steps) if steps else None


def terminal_outcome(run_id: str, *, log_root: Any = None) -> dict[str, Any]:
    """What ended this run, read from the **authoritative run log**  (FR-006.28, ADR-004).

    ~~``mf_runs`` has **no ``stop_reason`` column** - the DDL is owed and is in F6's feature
    report~~ - **stale since migration ``0016``, which added ``stop_reason``,
    ``stop_detail``, ``round`` and four more as F6's index**; the chain sweep reads that
    column directly. **The code below is still right and deliberately unchanged:** this is
    read from the last ``boundary`` block rather than from the row. That
    is not a workaround: the log is the record and the row is an index over it, and the
    reason is a **function of the block** rather than a value missing from it. The block
    carries every condition's ``triggered``, and
    :data:`training.stop.precedence.PRECEDENCE` is deterministic, so the single-valued
    reason is derivable and ``all_fired`` with it.

    *Reported rather than smuggled:* ``BoundaryBlock`` is closed to unknown keys, so F6
    cannot add a ``stop_reason`` field to it, and nothing is hidden in
    ``cadence.reason`` to compensate.

    Returns:
        ``{"stop_reason", "stop_detail", "all_fired", "round"}``. **Every key is present**,
        with ``None`` or ``[]`` where the run has not ended - *a missing key is
        indistinguishable from a healthy one.*
    """
    from training.stop import precedence
    from training.stop.base import Verdict

    empty: dict[str, Any] = {
        "stop_reason": None,
        "stop_detail": None,
        "all_fired": [],
        "round": None,
    }
    path = _log_path(run_id, log_root)
    if not path.is_file():
        return empty
    blocks = list(run_log.read_blocks(path, block="boundary"))
    if not blocks:
        return empty

    final = blocks[-1]
    verdicts = [
        Verdict(
            condition=str(entry["condition"]),
            triggered=bool(entry["triggered"]),
            observed_at_step=int(entry["observed_at_step"]),
            detail=entry.get("detail"),
        )
        for entry in final.get("stop_conditions") or ()
    ]
    first = precedence.first_fired(verdicts)
    return {
        "stop_reason": None if first is None else first.condition,
        "stop_detail": None if first is None else first.detail,
        "all_fired": list(precedence.fired(verdicts)),
        "round": final.get("round"),
    }
