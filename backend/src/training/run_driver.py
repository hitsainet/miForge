"""The production driver: from a recorded definition to a terminal run  (F6).

:func:`training.tasks.run_step_loop` is the Celery task; this module is everything it does,
as a plain function so the whole path is testable without a broker.

## What this module closes

Three things were built, tested, documented and **had no production caller** until this
file existed. By ADR-016 #1 that means none of them was shipped:

* **F8's run types.** Nothing called ``runtypes.resolve()`` on the submit path.
  :mod:`services.run_definition`'s budget step calls ``RunType.preflight``, and **the loop
  calls** ``prepare`` **and** ``save`` **- since 2026-10-16, and not before.**

  ⚠ **This bullet used to read "and this module calls ``prepare`` and ``save``", and BOTH
  halves were false**; the correction then said so and left the defect open. ~~Reported
  rather than fixed here - routing the loop through ``RunType`` is F8's change, not a line
  in this module.~~ **CLOSED**: the dispatch is three calls in
  :func:`training.tasks.drive_run` - ``resolve`` at the top, ``prepare`` before the
  optimizer, ``save`` at the save point - and this module supplies the two values it owns,
  ``run_type`` and ``adapter``, off the recorded definition. What the gap cost while it was
  open, kept because it is why the dispatch is where it is:

  1. ``RunType.prepare`` is what returns the :class:`~training.runtypes.base.TrainablePlan`
     - ``all_parameters`` for ``change``/``create``, and the adapter's
     ``assert_only_adapter_requires_grad``. Unwired, an **adapter run optimised every
     parameter**, because ``optimizer_module.build`` was handed ``acquired.module`` whole,
     so R-02.43's whole economy did not happen. That was the expensive half. It is why
     ``prepare`` runs **before** the optimizer is built rather than beside the acquisition:
     the freeze has to precede the parameter collection.
  2. ``RunType.save`` is what picks the artifact kind. Unwired, an adapter run's save point
     wrote a **full checkpoint** through F5's writer rather than a PEFT directory through
     ``adapter_save``, so the artifact R-02.44 defines - and acceptance criterion 15 is
     about - was not what a run produced. The ``test``/``retention`` provenance owed at the
     same seam stays closed: both writers reach it through F8's one join
     (:func:`training.runtypes.change.joined_scores`) rather than a projection each.

  **The controls** (ADR-016 #1): ``tests/integration/test_runtype_dispatch.py`` asserts an
  adapter run's optimizer covers the adapter's parameters **alone** and that its save point
  publishes a PEFT directory, and mutating either dispatch line out turns it red; the
  ``change`` path's checkpoint-with-tokenizer-at-root is pinned by
  ``test_save_wiring.py``, which is unchanged by the routing.
* **F7's pack gate.** ``require_valid_pack`` is step 2 of the validator.
* **The weights save's two missing values.** ``training/tasks.py`` wrote trainer state and
  **skipped the checkpoint**, logging ``checkpoint_save_skipped_no_provenance``, whenever
  it lacked ``signal_source`` (F2's) or ``domain`` (F6's, off R-02.53's definition). Both
  come off the **recorded definition** here, so the skip no longer fires on this path -
  and ``ManifestFields`` still gives neither a default, so the one thing that cannot happen
  is a fabricated provenance claim in a published artifact.

## A run with no recorded definition is refused, by name

R-02.53 requires a definition before the enqueue. A row created by F1's pre-R-02.53
``POST /api/v1/runs`` has none, and this driver **refuses** it (``run_definition_missing``)
rather than running it unbounded on the node's only card. That refusal is what makes the
ungated submit path unable to produce a run that trains, so there is exactly one path to a
running run and the gate sits on it.

* **F3's review-queue port.** ``bind_review_queue`` had no production caller, so a
  ``human_review`` selector refused ``midataworks_not_configured`` whatever was configured.
  The binding is :func:`clients.midataworks_review.wire`'s; the **run scope**
  is this module's, because the queue is addressed by ``run_id`` and nothing else on the
  path holds one for the whole run.

## The terminal transition happens **once**, here

The loop returns what happened and transitions nothing (so a boundary cannot half-finish a
run). This module marks the row terminal, **then** releases the lease (FR-006.29: a lease
has an expiry and a run row does not), **then** announces (FR-006.28: durable before
announced).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from core.errors import Refusal, refuse
from schemas.run_definition import RunDefinition
from services.selectors.plugins import human_review
from training import loop, run_state, tasks
from training.acquisition.base import AcquisitionSpec
from training.run_config import RunConfig
from training.step_record import rebuild_step_index
from workers.cancellation import CancellationToken
from workers.heartbeat import Heartbeat
from workers.run_record import RUNS, TERMINAL_STATES, require_run

log = logging.getLogger(__name__)


def drive_recorded_run(
    run_id: str,
    *,
    session: Any,
    session_scope: Any = None,
    max_steps: int | None = None,
    emit: Any = None,
    log_root: Path | None = None,
    trainer_state_root: Path | None = None,
    batch_source: Any = None,
    objectives: Any = None,
    evaluate: Any = None,
) -> dict[str, Any]:
    """Read the definition, drive the loop, transition the row. The whole production path.

    Args:
        run_id: the run. Its definition must already be recorded (FR-006.63).
        session: an open session.
        session_scope: a factory yielding a **new** session per call. Handed to the
            operator-stop condition, which must defeat this session's snapshot (M-2).
        batch_source: injected for tests; ``None`` resolves F2's source from the
            definition's ``signal_source`` block.
        evaluate: injected for tests; ``None`` uses F4's ``evaluate_run``.

    Raises:
        Refusal: ``run_definition_missing`` for a run with no recorded definition;
            ``run_not_runnable`` for a row that is already terminal (E-19).
    """
    definition, resolved = _recorded_definition(session, run_id)

    # E-19: a loop finding its run row already terminal stops immediately WITHOUT stepping,
    # and does not move the row. The heartbeat refuses to move a terminal row anyway
    # (FR-006.11) and F6 adds no path around that; this is the earlier, louder half.
    status = require_run(session, run_id)
    if status in TERMINAL_STATES:
        raise refuse(
            "run_not_runnable",
            f"Run {run_id!r} is already {status}, so the loop will not step it. A terminal "
            f"row is a historical record; continue from its last checkpoint instead.",
            run_id=run_id,
            status=status,
        )

    enabled = definition.enabled_conditions()
    ceiling = _ceiling(enabled, max_steps)
    signal_source = str(definition.signal_source.get("source") or "")
    domain = resolved.get("domain")
    seed = resolved.get("seed")

    # ⚠ THE LEASED CARD'S UUID IS ON THE ROW, NOT IN THE DEFINITION — wired 2026-10-16.
    # `definition.gpu_request` is whatever was SUBMITTED, usually `"auto"`; the card actually
    # leased at submit is `mf_runs.gpu_uuid`. This same function already reads that column
    # twice below — to RELEASE the lease — so the value was in hand all along and acquisition
    # was handed `"auto"` anyway.
    #
    # What that cost, before placement existed to notice: nothing in `backend/src/` moved the
    # model onto a card at all, so a run held a lease and trained on the CPU in silence. With
    # placement wired but the UUID still absent, a multi-card host must refuse `gpu_not_pinned`
    # rather than guess — taking index 0 would put two runs on one card with both leases
    # claiming exclusivity and both runs' numbers wrong. ADR-008: **pin by UUID, never index**,
    # and the index is re-derived only at the moment of placement.
    leased_gpu_uuid = session.execute(
        sa.select(RUNS.c.gpu_uuid).where(RUNS.c.id == run_id)
    ).scalar_one_or_none()
    spec = AcquisitionSpec(
        **{
            **definition.model_source,
            "seed": seed,
            # The lease wins over the submitted request: `"auto"` is a request to CHOOSE, and
            # by the time a worker runs, the choice has already been made and recorded.
            "gpu_request": leased_gpu_uuid or definition.gpu_request,
        }
    )
    cfg = RunConfig(**{**definition.run_configuration, "seed": seed})

    run_state.mark_running(session, run_id)
    session.commit()

    heartbeat = Heartbeat(run_id)
    # THE FIRST BEAT, AT LOOP ENTRY, BEFORE ANY WORK (workers.heartbeat.Heartbeat.start).
    # The janitor declines to reap a `running` row whose heartbeat was never written - it
    # cannot tell a worker that died at once from one two seconds into its first model load -
    # and `start()` is what keeps that case empty. It had no production caller until
    # 2026-10-17, so a run that died during acquisition stayed `running` forever and held the
    # GPU lease until it expired (found live, F11 window 2).
    heartbeat.start(session)
    session.commit()
    cancel = CancellationToken(run_id, session_scope=session_scope)
    # ⚠ THE REVIEW-QUEUE SCOPE WRAPS THE WHOLE RUN, INCLUDING THE SOURCE PRE-FLIGHT.
    #
    # The `human_review` selector's port is bound process-wide by
    # `clients.midataworks_review.wire`, but the queue itself is per-run: the
    # queue id lands on *this* run's binding row and the decisions are read back through it.
    # `review_run` is what tells the bound port which run it is speaking for, and without it
    # every queue operation refuses `review_queue_run_unknown` by name.
    #
    # It starts **before** `_resolve_source`, not merely before `drive_run`:
    # `source.preflight(spec)` resolves the run's selector configuration, and
    # `services.selectors.resolve.resolve` is what calls `HumanReviewSelector.prepare` - so a
    # scope opened after the source was built would refuse at run start, which reads like a
    # misconfigured queue rather than like missing wiring.
    #
    # The scope lives in `services/selectors/` and NOT in `clients/`: `training/**` may not
    # import `clients.*` (AC20's `test_F1_has_no_outbound_sibling_call_site`), and the port
    # F10 binds there is handed a callable that reads this scope rather than importing it.
    #
    # The body below is NOT extracted into a helper, deliberately:
    # `test_run_loop_reachability.py` walks this function's AST for the `tasks.drive_run`
    # call, so moving it one frame out would make that reachability test raise
    # `StopIteration` instead of checking the payload. Observed, not assumed.
    with human_review.review_run(run_id):
        source = (
            batch_source
            if batch_source is not None
            else _resolve_source(
                definition,
                run_id=run_id,
                session=session,
                emit=emit,
                log_root=log_root,
                resolved=resolved,
            )
        )

        try:
            result = tasks.drive_run(
                run_id=run_id,
                spec=spec,
                cfg=cfg,
                batch_source=source,
                objectives=objectives,
                session=session,
                session_scope=session_scope,
                cancel=cancel,
                heartbeat=heartbeat,
                emit=emit,
                max_steps=ceiling,
                save_every=definition.cadences.save_every,
                log_root=log_root,
                trainer_state_root=trainer_state_root,
                # THE TWO VALUES THE SAVE WAS MISSING. Off the recorded definition, never
                # defaulted: `domain=None` is an assertion that the run is not domain-scoped
                # (legitimate for general pretraining), and `signal_source` is the batch's
                # provenance, which has no value that would not be a fabricated claim.
                domain=domain,
                signal_source=signal_source,
                # F8'S DISPATCH, OFF THE RECORDED DEFINITION (R-02.41, R-02.43). The name,
                # not a resolved object: `drive_run` resolves it through F8's one registry,
                # which refuses an unknown name listing the three - and the submit path
                # already resolved the same string to run `preflight`, so a definition that
                # reaches a worker has had its run type checked twice by the same registry
                # rather than once by a registry and once by a literal.
                run_type=definition.run_type,
                adapter=definition.adapter,
                # F6's loop, fully on.
                stop_conditions=enabled,
                test_every=definition.cadences.test_every,
                retention_every=definition.cadences.retention_every,
                refresh_every=definition.cadences.refresh_every,
                refresh_mode=None if definition.refresh is None else definition.refresh.mode,
                refresh_rule=None if definition.refresh is None else definition.refresh.rule,
                evaluate=evaluate if evaluate is not None else _evaluator(run_id, session, emit),
                evaluate_step_zero=True,
                run_end_save=True,
                start_step=_resume_step(run_id, log_root),
            )
        except Refusal as exc:
            _finalize_failed(
                session,
                run_id,
                code=exc.code,
                message=failure_text(exc),
                emit=emit,
                log_root=log_root,
            )
            raise
        except BaseException as exc:
            _finalize_failed(
                session,
                run_id,
                code="run_loop_error",
                message=f"{type(exc).__name__}: {exc}",
                emit=emit,
                log_root=log_root,
            )
            raise

        return _finalize(session, run_id, result, emit=emit)


# ------------------------------------------------------------------- the terminal halves


def _finalize(session: Any, run_id: str, result: tasks.RunResult, *, emit: Any) -> dict[str, Any]:
    """Mark terminal, release the lease, announce. **In that order** (FR-006.28, .29)."""
    from training.stop.base import STOP_REASONS

    reason = result.stopped_because
    stop_reason = reason if reason in STOP_REASONS else None
    if reason == loop.CANCELLED:
        # A cancellation IS an operator stop: the flag is `cancel_requested_at` and the
        # condition that reads it is `operator_stop`. One vocabulary, so the terminal row
        # and the stop condition cannot name the same event two ways.
        stop_reason = "operator_stop"

    if stop_reason is None:
        # The step budget ran out with no condition firing. A run whose ONLY bound was the
        # driver's ceiling is refused at submit (FR-006.17), so reaching here means the
        # ceiling was tighter than the configured `step_count` - recorded as `step_count`
        # with the detail saying which bound bit, rather than inventing a sixth reason.
        stop_reason = "step_count"

    run_state.mark_completed(session, run_id, stop_reason=stop_reason, detail=result.stop_detail)
    _record_last_step(session, run_id, result.last_step)
    gpu_uuid = session.execute(
        sa.select(RUNS.c.gpu_uuid).where(RUNS.c.id == run_id)
    ).scalar_one_or_none()
    session.commit()

    run_state.release_lease(session, run_id, gpu_uuid)
    session.commit()

    # ANNOUNCED LAST, outside the durable writes. A socket failure cannot roll back a
    # recorded terminal state.
    run_state.announce_terminal(
        emit,
        run_id,
        status=run_state.STATE_COMPLETED,
        stop_reason=stop_reason,
        detail=result.stop_detail,
    )
    return {
        "run_id": run_id,
        "status": run_state.STATE_COMPLETED,
        "stop_reason": stop_reason,
        "stop_detail": result.stop_detail,
        "all_fired": list(result.all_fired),
        "steps_completed": result.steps_completed,
        "last_step": result.last_step,
        "rounds": result.rounds,
        "checkpoints": result.checkpoints,
        "trainer_states": result.trainer_states,
    }


def failure_text(exc: Any) -> str:
    """The refusal's message AND its details, for the run row.

    Only the code reached the row an operator can read until 2026-10-17, and the worker's
    own log lines are not in the container log - so a `weights_rebuild_failed` in the
    cluster said nothing about which tensors or why (F11 window 2). Bounded, and the
    details never carry a secret: refusals are built to be shown.
    """
    import json

    details = getattr(exc, "details", None)
    text = str(exc)
    if details:
        text += " | details: " + json.dumps(details, default=str, sort_keys=True)
    return text[:8000]


def _finalize_failed(
    session: Any,
    run_id: str,
    *,
    code: str,
    message: str,
    emit: Any,
    log_root: Path | None = None,
) -> None:
    """Mark ``failed``, restore the step index from the log, release the lease. **No run-end
    save** (E-9).

    The loop re-raises rather than returning, so the run-end save is skipped by
    construction - there is no flag here for a later edit to get wrong.
    """
    session.rollback()
    run_state.mark_failed(session, run_id, code=code, message=message)
    # THE ROLLBACK ABOVE TAKES EVERY UNCOMMITTED mf_steps ROW WITH IT - up to a minute of
    # steps, since the index commits with the heartbeat. The run log was written FIRST and
    # holds every completed step (ADR-004), so the index is rebuilt from it here. A failed
    # rebuild is logged and never replaces the run's own failure (found live, F11 window 2:
    # a run that trained 650 steps kept none of them in the index).
    try:
        with session.begin_nested():
            rebuilt = rebuild_step_index(
                session, log_root=tasks.run_log_root(log_root), run_id=run_id
            )
        log.info(
            "step index rebuilt from the run log after a failure",
            extra={"code": "step_index_rebuilt", "run_id": run_id, "steps": rebuilt},
        )
    except Exception as exc:  # never let the rebuild replace the run's own failure
        log.warning(
            "step index not rebuilt after a failure; the run log still holds every step",
            extra={"code": "step_index_rebuild_failed", "run_id": run_id, "error": str(exc)},
        )
    _record_last_step(session, run_id, _resume_step(run_id, log_root) or None)
    gpu_uuid = session.execute(
        sa.select(RUNS.c.gpu_uuid).where(RUNS.c.id == run_id)
    ).scalar_one_or_none()
    session.commit()
    run_state.release_lease(session, run_id, gpu_uuid)
    session.commit()
    run_state.announce_terminal(
        emit,
        run_id,
        status=run_state.STATE_FAILED,
        stop_reason=None,
        error_code=code,
    )


def _record_last_step(session: Any, run_id: str, step: int | None) -> None:
    """The run row's ``last_step`` at the terminal transition, from what actually happened.

    ``mf_runs.last_step`` had no production writer, so ``GET /api/v1/runs`` - the list the
    run dropdown reads - showed ``null`` for every run (found live 2026-10-18). ``None``
    writes nothing: no step is not step 0.
    """
    if step:
        session.execute(sa.update(RUNS).where(RUNS.c.id == run_id).values(last_step=int(step)))


# ----------------------------------------------------------------------------- readers


def _recorded_definition(session: Any, run_id: str) -> tuple[RunDefinition, dict[str, Any]]:
    """The definition as recorded, through the run log, with the index as the fallback.

    FR-006.64: **the worker reads the recorded definition, not the request**, and re-reads
    nothing it was told at submit except the stop flags. In particular the pack is **not**
    re-gated per boundary: F7's gate runs at start, and a run already holding parameters
    cannot un-start. Re-gating would also put an F7 resolution on the per-step path against
    FR-006.50's 5 % budget.
    """
    from services import run_launch

    record = run_launch.read_definition(session, run_id)
    submitted = record["submitted"]
    if not isinstance(submitted, Mapping):  # pragma: no cover - the writer guarantees it
        raise refuse(
            "run_definition_missing",
            f"Run {run_id!r}'s recorded definition is not an object.",
            run_id=run_id,
        )
    return RunDefinition(**dict(submitted)), dict(record["resolved"] or {})


def _ceiling(enabled: Mapping[str, Mapping[str, Any]], max_steps: int | None) -> int:
    """The driver's hard ceiling.

    The ``step_count`` condition's total when one is configured, else ``max_steps``, else a
    large bound that the other conditions are expected to end before. **A run with no
    condition at all is refused at submit** (FR-006.17), so this never bounds an unbounded
    run - it bounds a run whose only conditions are open-ended ones (a floor, a plateau, an
    operator stop), which is a legitimate shape.
    """
    from training.stop.base import StopReason

    total = (enabled.get(StopReason.STEP_COUNT.value) or {}).get("total")
    if isinstance(total, int):
        return total if max_steps is None else min(total, max_steps)
    if max_steps is not None:
        return max_steps
    return DEFAULT_CEILING


#: The ceiling for a run bounded only by open-ended conditions. Large enough not to be the
#: thing that ends such a run, finite because an unbounded ``while`` on the node's only card
#: is the failure FR-006.17 exists to prevent and a driver should not be able to express it.
DEFAULT_CEILING = 1_000_000


def _resume_step(run_id: str, log_root: Path | None) -> int:
    """The last completed step, from the **authoritative run log**  (FR-006.14, AC6).

    Re-derived rather than remembered: the loop never reads its position from worker
    memory, so a restart continues from the last recorded step with no gap and no repeat.
    Read from the log rather than from ``mf_steps`` because the index is rebuildable and a
    reindex in flight could make a run resume somewhere else.
    """
    from training.step_record import StepWriter

    writer = StepWriter(run_id=run_id, log_root=tasks.run_log_root(log_root))
    return writer.last_recorded_step() or 0


def source_block(definition: RunDefinition, resolved: dict[str, Any]) -> dict[str, Any]:
    """The source's spec: its own block, plus the run's parts **of the kind it consumes**.

    A definition names its material at the TOP level (``parts``, or a ``pack`` the validator
    resolves) and the source's block names only the source. Until 2026-10-17 the driver
    built the spec from the block alone, so every parts- or pack-based run that validated
    clean refused ``signal_source_no_part`` in the worker (found by tracing a real load run
    end to end, F11 window 2). The resolved list is the validator's (part ids it checked);
    test and retention sets are F4's and are not handed to the source.
    """
    from training.signals import parts_reader

    block = dict(definition.signal_source)
    if block.get("parts"):
        return block  # stated explicitly on the source: kept as given
    wanted = parts_reader.SOURCE_PART_KIND.get(str(block.get("source") or ""))
    given = {p.get("part_id"): p for p in (definition.parts or []) if isinstance(p, dict)}
    entries = (resolved.get("parts") or {}).get("parts") or []
    parts = []
    for entry in entries:
        if entry.get("kind") != wanted or not entry.get("part_id"):
            continue
        part = {"kind": entry["kind"], "part_id": entry["part_id"]}
        for key in ("splits", "columns"):
            if given.get(entry["part_id"], {}).get(key):
                part[key] = given[entry["part_id"]][key]
        parts.append(part)
    if parts:
        block["parts"] = parts
    return block


def _resolve_source(
    definition: RunDefinition,
    *,
    run_id: str,
    session: Any,
    emit: Any,
    log_root: Path | None,
    resolved: dict[str, Any] | None = None,
) -> Any:
    """F2's source, from the definition's own block, through F2's **one** registry.

    Resolved by name through :func:`training.signals.registry.get` - there is no second
    lookup path and no module-path resolution, which is what makes *"is it registered?"*
    testable.
    """
    from training import signals
    from training.signals.base import RunContext, SourceSpec

    signals.ensure_registered()
    spec = SourceSpec(**source_block(definition, resolved or {}))
    factory = signals.registry.get(spec.source)
    source = factory(
        RunContext(
            run_id=run_id,
            session=session,
            log_root=tasks.run_log_root(log_root),
            emit=emit,
        )
    )
    source.preflight(spec)
    return source


def _evaluator(run_id: str, session: Any, emit: Any) -> Any:
    """F4's evaluation, bound to this run. Returns ``None`` when the run has no scoring.

    The refusal F4 raises for an unconfigured run (``run_scoring_not_configured``) is
    **swallowed here and nowhere else**: a run may legitimately have no test or retention
    set, and the loop must not fail such a run at step 0. Every other refusal propagates -
    E-11's rule is that *a run whose floor cannot be measured is a run with no floor*, so an
    evaluation that fails for any other reason stops the run.
    """

    def evaluate(step: int) -> Any:
        from services.scoring import entrypoint as scoring

        try:
            return scoring.evaluate_run(run_id=run_id, step=step, session=session, emit=emit)
        except Refusal as exc:
            if exc.code == "run_scoring_not_configured":
                log.info(
                    "no scoring configured for this run; no set score is recorded",
                    extra={
                        "code": "run_scoring_not_configured",
                        "run_id": run_id,
                        "step": step,
                    },
                )
                return None
            raise

    return evaluate


__all__ = ["DEFAULT_CEILING", "drive_recorded_run"]
