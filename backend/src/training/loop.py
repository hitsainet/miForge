"""**The only ``repeat`` in the product**  (R-02.24; FR-006.1-.9, .12, .14).

R-02.24 is six lines of pseudocode. This module is what those six lines cost once they have
to survive a worker restart, a solo-pool Celery worker that cannot be signalled, a janitor
that reaps anything quiet, an hours-long run on the node's only GPU, and an operator who
presses stop.

``backend/tests/unit/test_one_repeat.py`` asserts structurally that there is exactly one
step loop in ``backend/src/`` and that it is here (AC-006.1). **F1's ``drive_run`` delegates
to this function** rather than keeping its own ``for``: F1's own TDD says *"F6 owns the
repeating"*, and its task tests are kept as the regression for that delegation.

## Nothing else is in the loop

Every operation inside a turn belongs to a feature that is already written - F1 takes the
step, F2 assembles the batch, F4 evaluates, F5 saves. F6 decides *when*, *how often*, and
*when to stop*. A loop that starts computing, scoring or writing is where the orchestrator
turns back into the monolith the architecture was decomposed to avoid.

In particular the loop **never calls** ``source.next()``: F1's ``step.run_one`` pulls the
batch, exactly once per boundary. **Mutation M-12** adds a second call, and 3.3's
call-count test must go red - two pulls per step means every other batch is trained on
nothing and the step series looks perfect.

## Step 0 is evaluated before the first step  (FR-006.6, M-11)

**R-02.24's loop body does not contain this**: its ``test()`` follows ``step(batch)``, so
with steps numbered from 1 no evaluation ever occurs at step 0 - while F4 FR-004.14 requires
a step-0 evaluation and **acceptance criterion 12 names the step-0 test score as a creation
run's baseline explicitly.** Reported as **DEF-1**. And no save happens at step 0:
``0 % n == 0`` for every ``n``, so a naive cadence check would checkpoint untrained weights.

## A save at run end, and **not** after a failure  (FR-006.8, E-9, M-13)

R-02.20 requires a save *"on the configured cadence **and at run end**"*; R-02.24's loop
body contains no run-end save, so the obligation lands here. The loop saves at run end for
**every stop-condition termination**, including a floor breach and an operator stop - the
artifact tells the truth, it is not hidden to make the run look better (FR-006.9). It does
**not** save after a ``failed`` termination: the last completed save is the recovery point,
and a save after an exception writes an artifact from a state nothing verified.

When run end lands on a cadence boundary, F5's FR-005.6 idempotence makes the second save a
no-op if the hash matches. **F6 adds no de-duplication of its own** (S-2): a second rule
about when a save is really a save would be a second answer, and F5's is the one that reads
the manifest.

## The loop is restartable and re-derives its position  (FR-006.14, AC6)

The step counter and the round are the only mutable members of :class:`LoopState`, and both
are re-derivable at resume - the step from the authoritative run log through F1's
``StepWriter.last_recorded_step``, the round from the step and ``refresh_every``. The loop
never reads its position from worker memory.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from training.boundary import BoundaryContext, BoundaryOutcome, run_boundary
from training.refresh.base import RefreshOutcome

log = logging.getLogger(__name__)

#: What the loop reports when it ran out of its step budget without any stop condition
#: firing. **Not a stop reason** - R-02.25's set is closed at five and ``step_count`` is the
#: condition an operator enables; this is the internal bound F1's ``drive_run`` passes so a
#: driver terminates at all. A run whose only bound is this one is refused at submit
#: (FR-006.17).
BUDGET_EXHAUSTED = "step_budget_reached"
CANCELLED = "cancelled"


@dataclass
class LoopState:
    """The loop's only mutable state. Two counters, both re-derivable at resume."""

    step: int = 0
    round: int = 0
    #: The plateau window for the current round. Cleared at a refresh (FR-006.22).
    window_losses: list[float] = field(default_factory=list)
    window_steps: list[int] = field(default_factory=list)
    window_health: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopResult:
    """What a driven run produced. A summary, never the record itself (ADR-003)."""

    run_id: str
    steps_completed: int
    last_step: int | None
    #: One of R-02.25's five, or :data:`BUDGET_EXHAUSTED` / :data:`CANCELLED`.
    stopped_because: str
    stop_detail: str | None = None
    #: Every condition that fired at the terminating boundary, in precedence order. Kept
    #: because FR-006.27 records all of them and only the first becomes ``stop_reason``.
    all_fired: tuple[str, ...] = ()
    rounds: int = 0
    boundaries: list[BoundaryOutcome] = field(default_factory=list)
    run_end_saved: bool = False


def run(
    ctx: BoundaryContext,
    *,
    max_steps: int,
    start_step: int = 0,
    evaluate_step_zero: bool = True,
    run_end_save: bool = True,
) -> LoopResult:
    """Repeat until a stop condition fires. **The one ``repeat``.**

    Args:
        ctx: the boundary context - every collaborator injected.
        max_steps: the step budget this driver will not exceed. F6's ``step_count``
            condition is what an operator configures; this is the hard ceiling that makes a
            driver terminate at all, and a run whose only bound is this one is refused at
            submit.
        start_step: the step already completed when the loop begins. Non-zero on a
            continue, read from the authoritative run log by the caller (FR-006.14).
        evaluate_step_zero: FR-006.6's baseline. ``False`` only on a resume, where step 0
            already has one and re-evaluating it would append a second baseline at a
            different parameter set.
        run_end_save: FR-006.8. ``False`` when the caller has no save to make - a run
            without a checkpoint-capable provenance, say.

    Returns:
        A :class:`LoopResult`. **The run row is not transitioned here**: that is
        :mod:`training.run_state`'s, called once by the task, so a boundary cannot
        half-finish a run.
    """
    state = LoopState(step=start_step, round=_round_of(start_step, ctx.refresh_every))
    result = LoopResult(
        run_id=ctx.run_id,
        steps_completed=0,
        last_step=start_step or None,
        stopped_because=BUDGET_EXHAUSTED,
    )

    # ---- the heartbeat, unconditionally, BEFORE any work --------------------- ADR-007
    # A `running` row whose heartbeat_at is NULL is ambiguous - a worker that died a
    # millisecond after claiming the run, or one two seconds into a model load - and the
    # janitor REPORTS such a row rather than reaping it. Beating first is what keeps that
    # ambiguous case empty in practice.
    if ctx.heartbeat is not None and ctx.session is not None:
        ctx.heartbeat.start(ctx.session)

    # ---- step 0: the baseline, before the first step ---------------- FR-006.6, M-11
    # `0 % n == 0` for every n, so this is NOT a cadence hit - it is an explicit, named
    # evaluation, and no save accompanies it. Removing this line is M-11 and 3.4's test
    # must go red: without it a creation run's first data point is already post-update
    # with nothing to compare against (acceptance criterion 12).
    if evaluate_step_zero and start_step == 0 and ctx.evaluate is not None:
        ctx.evaluate(0)

    try:
        while state.step < max_steps:
            next_step = state.step + 1
            outcome = run_boundary(
                _with_window(ctx, state),
                step=next_step,
                round_number=state.round,
            )
            state.step = next_step
            result.steps_completed += 1
            result.last_step = next_step
            result.boundaries.append(outcome)

            if outcome.loss is not None:
                state.window_losses.append(outcome.loss)
                state.window_steps.append(next_step)

            if outcome.refreshed is not None:
                _on_refresh(state, outcome.refreshed)

            if outcome.stop_reason is not None:
                result.stopped_because = outcome.stop_reason
                result.stop_detail = outcome.stop_detail
                result.all_fired = outcome.fired
                break
    except BaseException as exc:  # includes RunCancelled, which is a BaseException
        if type(exc).__name__ != "RunCancelled":
            # E-9. **RE-RAISED, and that is the mechanism rather than a flag.** The
            # run-end save is below this block, so a failure skips it *by construction*:
            # there is no `if not failed` for a later edit to get wrong. The caller turns
            # the raise into a `failed` transition, and the last completed save stays the
            # recovery point - a save after an exception would write an artifact from a
            # state nothing verified.
            log.error(
                "run loop terminated by an exception; no run-end save",
                extra={
                    "code": "run_loop_failed",
                    "run_id": ctx.run_id,
                    "step": state.step,
                    "error": type(exc).__name__,
                },
            )
            raise
        # Cancelled: landed on a boundary, so the step had not started and nothing is
        # recorded half (ADR-007). A run-end save DOES follow - FR-006.8 requires one for
        # every stop-condition termination, and an operator stop is one. The terminal
        # transition is the caller's.
        result.stopped_because = CANCELLED

    result.rounds = state.round + 1

    # ---- the run-end save, for every stop-condition termination ----- FR-006.8, M-13
    # Including a floor breach (FR-006.9) and an operator stop. NOT after a failure - that
    # path returned above.
    if run_end_save and ctx.save is not None and result.last_step:
        ctx.save(result.last_step)
        result.run_end_saved = True

    log.info(
        "run loop finished",
        extra={
            "code": "run_loop_finished",
            "run_id": ctx.run_id,
            "steps_completed": result.steps_completed,
            "last_step": result.last_step,
            "stopped_because": result.stopped_because,
            "all_fired": list(result.all_fired),
            "rounds": result.rounds,
        },
    )
    return result


def _with_window(ctx: BoundaryContext, state: LoopState) -> BoundaryContext:
    """Hand the boundary the current window. The loop owns the window; the condition reads it."""
    ctx.window_losses = tuple(state.window_losses)
    ctx.window_steps = tuple(state.window_steps)
    ctx.window_health = dict(state.window_health)
    return ctx


def _on_refresh(state: LoopState, refreshed: RefreshOutcome) -> None:
    """Increment the round and **clear the plateau window**  (FR-006.22, FR-006.37, M-10).

    The window is cleared here, in the loop, rather than inside the plateau condition: a
    refresh changes the batch distribution, so losses either side of it are not comparable,
    and a detector spanning the boundary measures the refresh rather than the model **in
    either direction** - a harder refreshed set raises the loss and resets a plateau that
    had genuinely arrived, an easier one lowers it and hides one.

    Clearing it here means a window **cannot** span a refresh by construction. M-10 lets it,
    and 5.6's window-reset test must go red.
    """
    state.round = refreshed.round
    state.window_losses.clear()
    state.window_steps.clear()
    state.window_health.clear()
    log.info(
        "plateau window reset at a refresh",
        extra={
            "code": "plateau_window_reset",
            "round": refreshed.round,
            "mode": refreshed.mode,
            "rule_sha256": refreshed.rule_sha256,
        },
    )


def _round_of(step: int, refresh_every: int | None) -> int:
    """The round a given step belongs to  (FR-006.37, AMB-4).

    **A "round" is ``refresh_every`` completed steps.** "Round" is defined nowhere in
    BRD-02 - verified; it appears in §1's prose and in acceptance criterion 3's *"over three
    rounds"* and is never given a length - so F6 defines it, which is what makes criterion
    3's "three rounds" a checkable statement. With no refresh rule a run has exactly one
    round, numbered 0.
    """
    if not refresh_every:
        return 0
    return int(step) // int(refresh_every)


def window_health_from(batch: Any) -> Mapping[str, Any]:
    """F3's named signals off a batch, for the plateau verdict's detail (FR-006.23).

    Read through ``TrainingBatch.health.signal_names`` - F3's own record, carried verbatim
    by F2 - rather than recomputed. **Nothing here stops a run**; the signals are attached
    so a plateau stop is interpretable, because a flat loss on a ``reward`` run can mean the
    model converged **or** that a zero advantage applied numerically nothing for a hundred
    steps, with the same curve.
    """
    health = getattr(batch, "health", None)
    names = getattr(health, "signal_names", ()) if health is not None else ()
    return {str(name): True for name in names}


__all__ = [
    "BUDGET_EXHAUSTED",
    "CANCELLED",
    "LoopResult",
    "LoopState",
    "run",
    "window_health_from",
]
