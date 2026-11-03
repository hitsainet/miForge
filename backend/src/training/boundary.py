"""One boundary, in a fixed order, with the heartbeat around every long phase.

FR-006.7's order, and **the order is the requirement**:

1. the step (F1);
2. the test and retention evaluation, if due (F4);
3. the save, if due (F5);
4. the stop-condition evaluation (F6);
5. the refresh, if due (F6).

## Evaluation precedes save, and swapping them is a silent defect  (M-7)

R-02.18 puts the step's test results into the checkpoint manifest, and F5 FR-005.15 writes
the ``ScoreRecord``s F4 supplies **verbatim or ``null``**, where *"``null`` means 'no test
ran at this step', which is a different statement from 'a test ran and scored nothing'."*
If the save ran first, a save landing on a coincident test boundary would write
``test: null`` for a step that **was** tested - a conforming, validating, permanently wrong
manifest. Nothing raises, nothing looks odd, and the artifact lies forever.

## The heartbeat brackets every long phase, not only the gaps between them  (M-3)

The database heartbeat is throttled to 60 s - **a minimum interval between writes, not a
maximum gap.** The loop's own sub-operations exceed it: a test-set pass is generation-bound
over hundreds of prompts, a self-generation batch is the same shape, and a 2.4 GB save is
seconds to tens of seconds (ADR-024). F4, F2 and F5 each advance the heartbeat during their
own passes; **F6's obligation is the one nobody else can hold** - that no loop-owned
sub-operation lacks an advance, including the boundary's own bookkeeping and the refresh.

ADR-007's named precedent is miStudio reaping a live 5.8-hour job as "worker lost". This is
that failure reproduced structurally, which is why **M-3** removes these advances and 4.7's
janitor test must go red.

## Durable first, announced second  (FR-006.12, M-17)

The boundary block goes to the authoritative run log, and **only then** is ``boundary``
emitted into ``miforge/run/<id>``. The emission is outside the durable write's ``try``, so a
socket failure cannot roll back a recorded boundary - *nothing about a run's progress exists
only on the socket*, which is ADR-007's rule and the reason it exists.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.errors import Refusal
from training import refresh as refresh_registry
from training import run_log
from training import stop as stop_registry
from training.refresh.base import RefreshContext, RefreshOutcome
from training.run_events import EVENT_BOUNDARY, EVENT_REFRESH
from training.stop.base import StopContext, Verdict

log = logging.getLogger(__name__)

#: The phase names used in ``BoundaryBlock.phase_timings_ms``. Enumerated so the budget
#: benchmark (FR-006.50) and the record agree about what ``loop_overhead`` excludes.
PHASES: tuple[str, ...] = ("step", "test", "save", "stop", "refresh", "loop_overhead")


@dataclass
class BoundaryContext:
    """What one boundary needs. Every collaborator injected, none imported here.

    The three callables are the other features' work, handed in as functions so this module
    orchestrates without reaching inside any of them (FR-006.1: *nothing else is in the
    loop*). ``take_step`` in particular is **F1's whole turn** - ``run_one`` plus the step
    record - and the loop calls it **once** per boundary and never calls
    ``source.next()`` itself (M-12).
    """

    run_id: str
    session: Any = None
    session_scope: Any = None
    heartbeat: Any = None
    emit: Any = None
    log_path: Path | None = None

    #: F1's one turn. Returns whatever F1 returns; the loop reads only ``loss``.
    take_step: Callable[[int], Any] | None = None
    #: F4's evaluation at this step, or ``None`` when the run has no scoring configured.
    evaluate: Callable[[int], Any] | None = None
    #: F5's save (trainer state plus the weights checkpoint) at this step.
    save: Callable[[int], Any] | None = None
    #: Whether an operator asked for a save, **consumed** at this boundary (F5 FR-005.1).
    #: Consulted at the save phase so the two reasons to save - the cadence and a request -
    #: collapse into ONE save at one boundary, which is F1's existing behaviour and the
    #: reason this is a hook rather than a second phase.
    save_requested: Callable[[int], Any] | None = None
    #: Rebuild the signal source after a refresh. The refresh constructs no source.
    apply_refresh: Callable[[RefreshOutcome], None] | None = None

    stop_conditions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    test_every: int = 1
    retention_every: int = 1
    save_every: int | None = None
    refresh_every: int | None = None
    refresh_mode: str | None = None
    refresh_rule: Mapping[str, Any] = field(default_factory=dict)
    #: The live signal source, for a ``template_seeds`` refresh. Read for seeds only.
    source: Any = None

    # ---- the plateau window, owned and mutated by the loop -------------- FR-006.22
    #: The losses of the current round, oldest first. **The loop clears this at a refresh**,
    #: which is what makes a plateau window unable to span one *by construction* rather than
    #: by the plateau condition remembering to check (M-10).
    window_losses: tuple[float, ...] = ()
    window_steps: tuple[int, ...] = ()
    #: F3's named signals observed over the window. Attached to a plateau verdict
    #: (FR-006.23) and **acted on by nothing**: F3 FR-003.10 states the division - *"F3
    #: measures and names; it does not stop a run"* - and no requirement makes a degenerate
    #: selector a stop condition.
    window_health: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class BoundaryOutcome:
    """What one boundary did. The boundary record's content, plus the loop's decisions."""

    step: int
    round: int
    verdicts: list[Verdict] = field(default_factory=list)
    fired: tuple[str, ...] = ()
    stop_reason: str | None = None
    stop_detail: str | None = None
    evaluated: bool = False
    saved: bool = False
    refreshed: RefreshOutcome | None = None
    loss: float | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    heartbeat_writes: int = 0
    longest_gap_ms: float | None = None
    #: The order the phases actually happened in. Not telemetry: it is what lets the
    #: ordering test assert the sequence rather than the result (M-7).
    effects: list[str] = field(default_factory=list)


def run_boundary(ctx: BoundaryContext, *, step: int, round_number: int) -> BoundaryOutcome:
    """One boundary in FR-006.7's order. Returns what happened; transitions nothing.

    The run row's terminal transition is :mod:`training.run_state`'s, called by the loop
    once, so a boundary cannot half-finish a run.
    """
    outcome = BoundaryOutcome(step=step, round=round_number)
    beat = _Beat(ctx)

    # ---- 1. THE STEP (F1) -------------------------------------------------- FR-006.2
    with _Phase(outcome, "step"):
        if ctx.take_step is not None:
            result = ctx.take_step(step)
            outcome.loss = _loss_of(result)
            outcome.effects.append("step")
    beat.advance()

    # ---- 2. THE EVALUATION, IF DUE (F4) ------------------- FR-006.7, M-7, R-02.18
    # BEFORE THE SAVE. Swapping 2 and 3 writes `test: null` into the manifest of a step
    # that WAS tested - a manifest that validates and is permanently wrong.
    if _due(step, ctx.test_every) or _due(step, ctx.retention_every):
        with _Phase(outcome, "test"):
            if ctx.evaluate is not None:
                ctx.evaluate(step)
                outcome.evaluated = True
                outcome.effects.append("test")
        beat.advance()

    # ---- 3. THE SAVE, IF DUE (F5) ------------------------------------------- FR-006.7
    # Two reasons, checked at the SAME boundary and collapsing into one save: the cadence,
    # and an operator's request. Checked here and nowhere else, because a save landing
    # mid-step would publish a parameter set no step record describes.
    on_cadence = _due(step, ctx.save_every)
    requested = ctx.save_requested(step) if ctx.save_requested is not None else None
    if on_cadence or requested:
        with _Phase(outcome, "save"):
            if ctx.save is not None:
                ctx.save(step)
                outcome.saved = True
                outcome.effects.append("save")
        beat.advance()

    # ---- 4. THE STOP-CONDITION EVALUATION (F6) ----------------------- FR-006.16, M-1
    # EVERY enabled condition, EVERY boundary, no short-circuit.
    with _Phase(outcome, "stop"):
        # THE WINDOW INCLUDES THE STEP JUST TAKEN. The loop owns the window and appends to
        # it after the boundary returns, so the context it handed in stops one step short -
        # and the loss of the step just completed is the most relevant observation a
        # plateau detector has. Without this the detector lags by exactly one step, which
        # on a short window is the difference between firing and not.
        #
        # *Found by a test, not by reading: the window-reset test was green on the broken
        # ordering and only went red once the arithmetic was stated as a number.*
        losses = ctx.window_losses
        steps = ctx.window_steps
        if outcome.loss is not None:
            losses = (*losses, outcome.loss)
            steps = (*steps, step)
        stop_ctx = StopContext(
            run_id=ctx.run_id,
            step=step,
            round=round_number,
            session=ctx.session,
            session_scope=ctx.session_scope,
            log_path=ctx.log_path,
            window_losses=losses,
            window_steps=steps,
            window_health=ctx.window_health,
        )
        outcome.verdicts = stop_registry.evaluate_all(ctx.stop_conditions, stop_ctx)
        outcome.fired = stop_registry.precedence.fired(outcome.verdicts)
        first = stop_registry.precedence.first_fired(outcome.verdicts)
        if first is not None:
            outcome.stop_reason = first.condition
            outcome.stop_detail = first.detail
        outcome.effects.append("stop")

    # ---- 5. THE REFRESH, IF DUE AND NOT STOPPING (F6) ---------------- FR-006.38, 6.7
    # LAST, and skipped when the run is stopping: a refresh recorded on a run that then
    # ended would say the prompt set rotated for a round that never happened.
    if (
        outcome.stop_reason is None
        and ctx.refresh_mode is not None
        and ctx.refresh_every is not None
        and _due(step, ctx.refresh_every)
    ):
        with _Phase(outcome, "refresh"):
            outcome.refreshed = _refresh(ctx, step=step, round_number=round_number + 1)
            outcome.effects.append("refresh")
        beat.advance()

    outcome.heartbeat_writes = beat.writes
    outcome.longest_gap_ms = beat.longest_gap_ms
    _record(ctx, outcome)
    return outcome


# ------------------------------------------------------------------------- the refresh


def _refresh(ctx: BoundaryContext, *, step: int, round_number: int) -> RefreshOutcome:
    """Run the refresh and hand the result to the loop's source rebuilder.

    A refusal propagates: FR-006.38 makes a failed refresh a **``failed`` termination**, not
    a sixth stop condition, and the loop's own handler turns the raise into that transition.
    Catching it here and carrying on would make the run log record a refresh that did not
    happen and every batch after it carry the wrong part reference.
    """
    assert ctx.refresh_mode is not None  # guarded by the caller
    refresh_ctx = RefreshContext(
        run_id=ctx.run_id,
        step=step,
        round=round_number,
        session=ctx.session,
        source=ctx.source,
    )
    outcome = refresh_registry.run(refresh_ctx, ctx.refresh_mode, dict(ctx.refresh_rule))
    if ctx.apply_refresh is not None:
        ctx.apply_refresh(outcome)

    # FR-006.36's record. **The run log has no `refresh` block** - `step-record-v1.json`
    # declares five blocks (step, evaluation, boundary, definition, batch) and
    # `BoundaryBlock` is closed to unknown keys, so there is nowhere in the authoritative
    # log for the mode, the rule hash and the seed counts to go. Reported as owed
    # foundation 3.3 work with the exact JSON; F6 states the need and defines no block.
    # Until then the record is the structured log (ADR-021) plus the round number on the
    # boundary block, and every batch after this carries the new part reference (F2's).
    log.info(
        "prompt set refreshed",
        extra={
            "code": "prompt_refresh_recorded",
            "run_id": ctx.run_id,
            "step": step,
            **outcome.payload(),
        },
    )
    if ctx.emit is not None:
        _emit(ctx, EVENT_REFRESH, {"run_id": ctx.run_id, "step": step, **outcome.payload()})
    return outcome


# ------------------------------------------------------------------------- the record


def _record(ctx: BoundaryContext, outcome: BoundaryOutcome) -> None:
    """Write the boundary block, **then** emit (FR-006.12).

    The write is validated against ``step-record-v1.json`` by
    :func:`training.run_log.append`, so a block with a key the contract does not declare is
    refused at the writer rather than discovered by a reader that skips it.
    """
    if ctx.log_path is not None:
        run_log.append(ctx.log_path, _block(ctx, outcome))
        outcome.effects.append("log")

    # OUTSIDE the durable write, and not in the same `try`. A socket failure must not be
    # able to roll back a recorded boundary.
    if ctx.emit is not None:
        _emit(
            ctx,
            EVENT_BOUNDARY,
            {
                "run_id": ctx.run_id,
                "step": outcome.step,
                "round": outcome.round,
                "loss": outcome.loss,
                "evaluated": outcome.evaluated,
                "saved": outcome.saved,
                "stop_reason": outcome.stop_reason,
                "all_fired": list(outcome.fired),
                "stop_conditions": [v.as_evaluation() for v in outcome.verdicts],
                "phase_timings_ms": dict(outcome.timings_ms),
            },
        )
        outcome.effects.append("socket")


def _block(ctx: BoundaryContext, outcome: BoundaryOutcome) -> dict[str, Any]:
    """The contract's ``BoundaryBlock``, and nothing else.

    ``additionalProperties`` is ``false`` there, so ``stop_reason`` and ``all_fired`` -
    which F6 would like to record - have no field. They are **derivable**: the array carries
    every condition's ``triggered``, and :data:`training.stop.precedence.PRECEDENCE` is
    deterministic, so the reason is a function of the block rather than a value missing from
    it. Reported as owed contract work; nothing is smuggled into ``cadence.reason``.
    """
    return {
        "block": "boundary",
        "run_id": ctx.run_id,
        "step": outcome.step,
        "ts": datetime.now(UTC).isoformat(),
        "round": outcome.round,
        "stop_conditions": [verdict.as_evaluation() for verdict in outcome.verdicts],
        "phase_timings_ms": {name: round(value, 3) for name, value in outcome.timings_ms.items()},
        "cadence": {
            "next_test_at_step": _next(outcome.step, ctx.test_every),
            "next_save_at_step": _next(outcome.step, ctx.save_every),
            "reason": outcome.stop_reason or ("refresh" if outcome.refreshed is not None else None),
        },
        "heartbeat_gap_ms": outcome.longest_gap_ms,
    }


def _emit(ctx: BoundaryContext, event: str, payload: Mapping[str, Any]) -> None:
    """Emit, and **swallow a transport failure loudly**.

    The socket is a projection (ADR-007). A Redis outage must not fail a run whose work is
    already durable, and it must not be silent either - otherwise the first symptom is a
    screen that stopped updating for a reason nobody recorded.
    """
    try:
        ctx.emit(ctx.run_id, event, dict(payload))
    except Exception as exc:
        log.warning(
            "run event not delivered; the record is durable",
            extra={
                "code": "run_event_not_delivered",
                "run_id": ctx.run_id,
                "event": event,
                "error": type(exc).__name__,
            },
        )


# --------------------------------------------------------------------------- helpers


class _Beat:
    """Advances the heartbeat and measures the gaps it leaves  (M-3, FR-006.10, OQ-5).

    The gap is measured on the wall clock between *attempts*, not between writes, because
    the 60 s throttle means most attempts do not write - and the number the janitor's
    threshold needs is how long the loop went without one.
    """

    def __init__(self, ctx: BoundaryContext) -> None:
        self._ctx = ctx
        self._last = time.monotonic()
        self.writes = 0
        self.longest_gap_ms: float | None = None

    def advance(self) -> None:
        now = time.monotonic()
        gap = (now - self._last) * 1000.0
        self.longest_gap_ms = gap if self.longest_gap_ms is None else max(self.longest_gap_ms, gap)
        self._last = now
        if self._ctx.heartbeat is None or self._ctx.session is None:
            return
        try:
            if self._ctx.heartbeat.touch(self._ctx.session):
                self.writes += 1
                # Durable at once - see training.step_record (commit when the beat is written).
                self._ctx.session.commit()
        except Refusal as exc:
            # FR-006.11: the heartbeat REFUSES to move a terminal row and F6 adds no path
            # around it. A refusal here is recorded and not worked around.
            log.warning(
                "heartbeat refused mid-boundary",
                extra={"code": exc.code, "run_id": self._ctx.run_id},
            )


class _Phase:
    """Time one phase into ``outcome.timings_ms``. A context manager so an exception still
    records the time the phase cost before it raised."""

    def __init__(self, outcome: BoundaryOutcome, phase: str) -> None:
        self._outcome = outcome
        self._phase = phase

    def __enter__(self) -> _Phase:
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        elapsed = (time.perf_counter() - self._started) * 1000.0
        self._outcome.timings_ms[self._phase] = (
            self._outcome.timings_ms.get(self._phase, 0.0) + elapsed
        )


def _due(step: int, every: int | None) -> bool:
    """Whether a cadence of ``every`` falls on ``step``.

    ``step >= 1`` is part of the test: ``0 % n == 0`` for every ``n``, so without it a
    cadence would fire at step 0 - which for the save would checkpoint untrained weights
    (task 3.4). Step 0's evaluation is the loop's explicit baseline, not a cadence hit.
    """
    return every is not None and every >= 1 and step >= 1 and step % every == 0


def _next(step: int, every: int | None) -> int | None:
    """The next step this cadence will fall on, or ``None`` when it has none."""
    if every is None or every < 1:
        return None
    return ((step // every) + 1) * every


def _loss_of(result: Any) -> float | None:
    """F1's step outcome's loss, or ``None``. Read defensively because the loop reads
    **only** this one field off F1's outcome and must not depend on its shape."""
    value = getattr(result, "loss", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - a non-numeric loss is F1's refusal
        return None


__all__ = [
    "PHASES",
    "BoundaryContext",
    "BoundaryOutcome",
    "run_boundary",
]
