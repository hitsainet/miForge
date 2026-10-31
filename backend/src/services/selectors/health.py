"""Selection health, and the three named signals  (FR-003.10; R-02.9, rationale from R-02.39).

**Selection health is a first-class output, not a log line.** This feature exists because
both of selection's failure modes are silent:

* **a selector that accepts everything is not selecting** - the run trains on raw output
  and drifts toward whatever the model already produces most readily; loss falls, the test
  score does not move, nothing errors;
* **a selector that rejects everything stalls the run** - batches never fill, and the run
  looks slow rather than broken;
* **a grader with zero variance is the same failure in scalar form** - a group-relative
  policy objective (R-02.7, GRPO) computes an advantage from the spread of rewards within a
  group, so a constant reward gives zero advantage for every member and the update is
  numerically nothing. The run burns GPU hours, records a loss, and learns nothing.

So each gets a **name**, recorded on the health record and emitted on the run's room.
:data:`SIGNAL_ACCEPTS_EVERYTHING` and friends are what F6 stops on (R-02.25) and what F7
and F9 render - a UI keyed on prose breaks on a wording change.

**F3 measures and names; it does not stop a run.** Stop conditions are F6's; the authoring
warning is F7's.

## Detection lives here, not in a plugin

A plugin sees one completion; degeneracy is a property of a **batch**. Putting the detection
in a plugin is how it ends up implemented five times and tested once.

## Two sentences, not one

The accept/reject sentence is verbatim from R-02.39, so the test compares strings. The
grader sentence is **new copy, not from any requirement**, and is marked as such - because
the accept/reject sentence does not describe the grader case at all (a grader with no spread
is not accepting or rejecting anything), and *a signal shown under a sentence that does not
describe it is worse than no sentence: the operator reads the wrong failure.*
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Sequence
from statistics import fmean, pvariance
from typing import Any

from schemas.selector import Verdict, VerdictValue, is_admitted
from services.selectors.settings import SelectorSettings, get_selector_settings
from services.selectors.types import (
    SIGNAL_ACCEPTS_EVERYTHING,
    SIGNAL_NO_SPREAD,
    SIGNAL_REJECTS_EVERYTHING,
    GraderStats,
    SelectionHealth,
    SelectionSignal,
)

log = logging.getLogger(__name__)

#: Strong references to in-flight socket emits. ``asyncio`` holds only a weak reference to
#: a task, so a fire-and-forget emit can be collected before it runs.
_PENDING_EMITS: set[Any] = set()

#: **Verbatim from R-02.39**, which BRD-02 quotes as the rationale for requiring a dry-run.
#: Supplied here once so F7 and F9 render one sentence rather than inventing two; the test
#: compares the string, which is only possible because it is fixed.
SELECTION_DEGENERATE_SENTENCE = (
    "A selector that accepts everything is not selecting; one that rejects everything "
    "will stall the run."
)

#: **New copy, not from any requirement**, and marked as such. The sentence above does not
#: describe this case: a grader with no spread is not accepting or rejecting anything.
GRADER_NO_SPREAD_SENTENCE = (
    "A grader whose scores have no spread gives every candidate the same advantage; the "
    "step runs and learns nothing from it."
)


def score_stats(values: Sequence[float]) -> GraderStats | None:
    """min / max / mean / variance in one place. ``None`` for an empty sequence.

    ``pvariance`` - the population variance - because the batch *is* the population whose
    spread the objective uses, not a sample from a larger one. With a single value the
    variance is 0.0, which is true and is exactly the degenerate case; the ``n`` floor is
    what stops a 1-of-1 batch from raising a signal on it.
    """
    if not values:
        return None
    numbers = [float(value) for value in values]
    return GraderStats(
        n=len(numbers),
        min=min(numbers),
        max=max(numbers),
        mean=fmean(numbers),
        variance=pvariance(numbers) if len(numbers) > 1 else 0.0,
    )


def compute_health(
    verdicts: Sequence[Verdict],
    *,
    settings: SelectorSettings | None = None,
    scores: Sequence[float] | None = None,
) -> SelectionHealth:
    """One pass over a batch's verdicts: counts, the rate **with ``n``**, and the signals.

    Args:
        verdicts: The batch's verdicts, in input order.
        settings: Injected so a test can set the ``n`` floor and the variance epsilon
            without touching the environment.
        scores: Grader scores, when the configuration has a grader form. Separate from the
            verdicts because a selector-only run has none, and an empty list and an absent
            one are different facts.
    """
    settings = settings or get_selector_settings()
    n = len(verdicts)
    accepted = sum(1 for verdict in verdicts if is_admitted(verdict))
    rejected = sum(1 for verdict in verdicts if verdict.value is VerdictValue.REJECT)
    undecided = sum(1 for verdict in verdicts if verdict.value is VerdictValue.UNDECIDED)

    # `is_admitted` rather than `value is ACCEPT`: it is the one place admission is decided
    # (foundation 3.1a), and counting it a second way here would be a second answer.
    histogram = Counter(verdict.label or "unlabelled" for verdict in verdicts)

    stats = score_stats(scores) if scores is not None else None
    signals = _signals(
        n=n,
        accepted=accepted,
        rejected=rejected,
        undecided=undecided,
        stats=stats,
        settings=settings,
    )
    return SelectionHealth(
        n=n,
        accepted=accepted,
        rejected=rejected,
        undecided=undecided,
        # The rate NEVER travels without `n` beside it (ADR-012). `Accepted 3 of 5`, never
        # a bare `60%`. A rate over an empty batch is 0.0 and the `n` says why.
        acceptance_rate=(accepted / n) if n else 0.0,
        reason_histogram=dict(sorted(histogram.items())),
        grader_stats=stats,
        signals=signals,
    )


def _signals(
    *,
    n: int,
    accepted: int,
    rejected: int,
    undecided: int,
    stats: GraderStats | None,
    settings: SelectorSettings,
) -> tuple[SelectionSignal, ...]:
    """The named degenerate outcomes, or none. Applies the ``n`` floor first."""
    floor = settings.selection_min_n_for_signals
    raised: list[SelectionSignal] = []

    if n >= floor:
        # A batch is not "accepting everything" if some of it is still pending: the
        # undecided candidates have not been accepted, so the claim would be false.
        if accepted == n and n > 0:
            raised.append(
                SelectionSignal(
                    name=SIGNAL_ACCEPTS_EVERYTHING,
                    message=SELECTION_DEGENERATE_SENTENCE,
                    details={"accepted": accepted, "n": n, "acceptance_rate": 1.0},
                )
            )
        elif rejected == n and n > 0:
            raised.append(
                SelectionSignal(
                    name=SIGNAL_REJECTS_EVERYTHING,
                    message=SELECTION_DEGENERATE_SENTENCE,
                    details={"rejected": rejected, "n": n, "acceptance_rate": 0.0},
                )
            )

    if (
        stats is not None
        and stats.n >= floor
        and stats.variance <= settings.grader_variance_epsilon
    ):
        raised.append(
            SelectionSignal(
                name=SIGNAL_NO_SPREAD,
                message=GRADER_NO_SPREAD_SENTENCE,
                details={
                    "variance": stats.variance,
                    "epsilon": settings.grader_variance_epsilon,
                    "n": stats.n,
                    "min": stats.min,
                    "max": stats.max,
                },
            )
        )

    del undecided  # counted on the record; not itself a degenerate signal
    return tuple(raised)


def signal_events(health: SelectionHealth) -> list[tuple[str, dict[str, Any]]]:
    """``(event name, payload)`` per signal. Pure, so the payload can be asserted.

    Separated from the emit so a test can assert the **payload and the call count** without
    a socket: ADR-016's rule is that *"was called" passes against a call sending the wrong
    arguments.*
    """
    return [
        (
            signal.name,
            {
                "signal": signal.name,
                "message": signal.message,
                "n": health.n,
                "accepted": health.accepted,
                "rejected": health.rejected,
                "undecided": health.undecided,
                "acceptance_rate": health.acceptance_rate,
                "details": signal.details,
            },
        )
        for signal in health.signals
    ]


def publish_signals(
    health: SelectionHealth,
    *,
    run_id: str | None,
    emit: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[str]:
    """Emit each signal and log one line each. Returns the names emitted.

    **The record is the truth and the socket is a convenience** (ADR-007): this is called
    *after* the health record exists, and a failure to emit does not lose the signal. That
    is the lesson behind miStudio reaping a live 5.8-hour job as "worker lost" - progress
    reported only over a socket is invisible to anything that was not listening.

    One structured line per signal, **none per verdict** (ADR-021): at batch scale a
    per-verdict line buries the signals sitting beside it.
    """
    emitted: list[str] = []
    for name, payload in signal_events(health):
        log.warning(
            "selection signal",
            extra={"code": name, "run_id": run_id, "details": payload["details"]},
        )
        if emit is not None:
            emit(name, payload)
        emitted.append(name)
    return emitted


def socket_emitter(run_id: str) -> Callable[[str, dict[str, Any]], None]:
    """A sync emitter onto ``miforge/run/<id>``, for a sync caller.

    :func:`api.sockets.emit_run_event` is a coroutine and this service is synchronous -
    deliberately so (the interface is sync, and every worker is ``--pool=solo -c 1``). So
    the coroutine is scheduled on a running loop when there is one and run to completion
    when there is not.

    **A failed emit is logged, never raised.** The record already holds the signal; losing
    the socket must not fail an evaluation whose verdicts are correct.
    """
    import asyncio

    def emit(event: str, payload: dict[str, Any]) -> None:
        from api.sockets import emit_run_event, publish_from_worker

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # NO LOOP: a Celery worker. The synchronous write-only publisher, never
            # `asyncio.run` per event (see `api.sockets.publish_from_worker`).
            try:
                publish_from_worker(run_id, event, payload)
            except Exception:  # transport failure, not a verdict
                log.warning("selection signal emit failed", extra={"code": event})
            return
        coroutine = emit_run_event(run_id, event, payload)
        # The reference is kept so the task is not garbage-collected mid-flight (RUF006):
        # a dropped task would lose the emit silently, and "silently" is the whole thing
        # this feature exists to avoid. The record already holds the signal either way.
        task = loop.create_task(coroutine)
        _PENDING_EMITS.add(task)
        task.add_done_callback(_PENDING_EMITS.discard)

    return emit
