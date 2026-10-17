"""The run lifecycle: ``pending -> running -> {completed, failed}``  (FR-006.13, .28, .29).

One function per transition, each asserting the state it transitions **from** in the
statement's own ``WHERE`` clause rather than in an ``if`` above it. The difference matters:
a check-then-act would race with a run finishing, and a run finishing is not a rare event -
it is what every run does.

## There is no ``paused`` state and no pause control  (AMB-2)

R-02.25 names *operator stop*, and **no requirement in BRD-02 defines pause, resume or a
held run** - verified: ``pause`` and ``paused`` do not occur in BRD-02 at all, and its one
use of ``held`` is the ordinary verb (*"6 s of held serving per step"* in §9's
device-placement note), never a run state. The mockup's Pause button and ``held`` run state
are **mockup-only scope**, reported as AMB-2, with the note that pause-and-resume is already
expressible from requirements that exist: an operator stop (R-02.25) followed by a continue
(R-02.22, F5 FR-005.28) is exactly that and needs no third state.

## ``pending``, not ``queued`` - **settled 2026-10-14, and ``queued`` stops existing**

FR-006.13 writes the first state as ``queued``; **``mf_runs.status``'s CHECK constraint says
``pending``** (``status_known``, foundation task 4.2), and ``workers.run_record`` re-exports
that set rather than restating it.

F9 found the cost of leaving both: a submitted run defaults to ``pending``, its pill map
carried ``queued``, and **the commonest state of a new run rendered as "pending
(unrecognised)"**. Two names for one state, each read as authoritative by a different layer.

**``pending`` wins, on three grounds and none of them is taste.** It is what the CHECK
constraint permits and the column defaults to, so nothing migrates and no row is rewritten.
It is what ``RUN_STATUSES``, ``workers.run_record.STATE_PENDING``, the janitor and
``tests/support/jobs.insert_run``'s default already say - and that default *alone* feeds
every run-creating test in the suite, which is how a one-word rename produced 123 collection
errors on 2026-10-14. And this project's own rule points there: *the constraint is the
authority and the document is the thing that is out of date.* ``queued`` would buy a
marginally clearer word for a migration, a column default change and a suite-wide rename.

So **FR-006.13's ``queued`` is a documentation defect**, reported for correction in F6's
FPRD/FTDD/FTID/FTASKS - F6 does not edit ``0xcc/``. This module imports the names from
:mod:`workers.run_record`, so there is no third spelling here either.

## A stop is durable before it is announced, and the lease is released last

FR-006.28 and FR-006.29. The order is **mark terminal, then release the lease**, and it is
the order because *a lease has an expiry and a run row does not*: a worker dying after the
terminal transition but before the release leaves a lease that frees the card by itself.
Reversed, a worker dying between the two leaves a non-terminal run with no lease, which the
janitor then has to disambiguate from a live run whose lease has lapsed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import sqlalchemy as sa

from core.errors import refuse
from training.run_events import EVENT_RUN_TERMINAL
from workers.run_record import (
    RUNS,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_PENDING,
    STATE_RUNNING,
    TERMINAL_STATES,
    require_run,
    rowcount_of,
    utcnow,
)

log = logging.getLogger(__name__)

# The four state names come from `workers.run_record`, which re-exports task 4.2's own set.
# NOT redefined here: a second copy of a state set is a second answer, and the one this
# module would get wrong is the one whose CHECK constraint spells it `pending`.

#: The legal transitions, as a table. One declaration, so the "every illegal transition is
#: refused" test enumerates from here rather than from whatever the reader remembers.
TRANSITIONS: Mapping[str, frozenset[str]] = {
    STATE_PENDING: frozenset({STATE_RUNNING, STATE_COMPLETED, STATE_FAILED}),
    STATE_RUNNING: frozenset({STATE_COMPLETED, STATE_FAILED}),
}

# STATE_PENDING -> COMPLETED is legal and is E-8: a stop requested on a queued run that has
# not started is HONOURED - the run goes terminal without ever acquiring the parameters, and
# the lease is released. Refusing it would leave the operator unable to cancel a queue.

assert (
    STATE_COMPLETED in TERMINAL_STATES and STATE_FAILED in TERMINAL_STATES
), "4.2's terminal set no longer contains this module's terminal states"


def mark_running(session: Any, run_id: str) -> bool:
    """``pending -> running``. Returns whether the row moved.

    ``False`` means the row was not ``pending``: already running (a restart, FR-006.14) or
    already terminal (E-19, the loop stops without stepping). The caller distinguishes
    them by reading the row; this function does not guess.
    """
    return _transition(session, run_id, to=STATE_RUNNING, values={})


def mark_completed(
    session: Any,
    run_id: str,
    *,
    stop_reason: str,
    detail: str | None = None,
) -> bool:
    """``running|pending -> completed``, carrying a ``stop_reason`` from R-02.25's five.

    ``stop_reason`` is **required**: FR-006.13 says a ``completed`` run carries one, and a
    terminal row that cannot say why it stopped is the shape of record an operator cannot
    act on.

    *The column it belongs in does not exist yet* - ``mf_runs`` has no ``stop_reason`` - so
    the reason's durable home is the **boundary block of the run log**, which is
    authoritative anyway (ADR-004) and from which it is derivable through
    :data:`training.stop.precedence.PRECEDENCE`. The DDL is in the feature report; the
    reason is never dropped.
    """
    from training.stop.base import STOP_REASONS

    if stop_reason not in STOP_REASONS:
        raise refuse(
            "stop_condition_unknown",
            f"{stop_reason!r} is not one of R-02.25's five stop reasons: "
            f"{list(STOP_REASONS)}. A completed run must say which of them ended it.",
            stop_reason=stop_reason,
            known=list(STOP_REASONS),
        )
    moved = _transition(
        session,
        run_id,
        to=STATE_COMPLETED,
        values={"finished_at": utcnow()},
    )
    if moved:
        log.info(
            "run completed",
            extra={
                "code": "run_completed",
                "run_id": run_id,
                "stop_reason": stop_reason,
                "stop_detail": detail,
            },
        )
    return moved


def mark_failed(
    session: Any,
    run_id: str,
    *,
    code: str,
    message: str | None = None,
) -> bool:
    """``running|pending -> failed``, carrying an error code (FR-006.13).

    No run-end save follows (FR-006.8, E-9): the last completed save is the recovery point,
    and a save after an exception writes an artifact from a state nothing verified.
    """
    moved = _transition(
        session,
        run_id,
        to=STATE_FAILED,
        values={
            "finished_at": utcnow(),
            "failure_code": code,
            "failure_message": message,
        },
    )
    if moved:
        log.error(
            "run failed",
            extra={"code": "run_failed", "run_id": run_id, "failure_code": code},
        )
    return moved


def _transition(session: Any, run_id: str, *, to: str, values: Mapping[str, Any]) -> bool:
    """One conditional ``UPDATE`` whose ``WHERE`` clause carries the policy.

    The legal source states are in the statement, so a row that changed between a read and
    this write is refused by the database rather than by a stale Python check. ``finished_at``
    is set in the same statement as ``status`` because ``mf_runs``' ``terminal_iff_finished``
    constraint pairs them - two statements would leave a window in which the row is illegal.
    """
    sources = sorted(state for state, allowed in TRANSITIONS.items() if to in allowed)
    if not sources:
        raise refuse(
            "run_state_transition_unknown",
            f"No legal transition reaches {to!r}. The lifecycle is "
            f"pending -> running -> {{completed, failed}} and nothing else (FR-006.13).",
            to=to,
        )
    status = require_run(session, run_id)
    result = session.execute(
        sa.update(RUNS)
        .where(RUNS.c.id == run_id)
        .where(RUNS.c.status.in_(sources))
        .values(status=to, updated_at=utcnow(), **dict(values))
    )
    moved = rowcount_of(result) > 0
    if not moved:
        log.warning(
            "run state transition refused",
            extra={
                "code": "run_state_transition_refused",
                "run_id": run_id,
                "from": status,
                "to": to,
                "legal_from": sources,
            },
        )
    return moved


def release_lease(session: Any, run_id: str, gpu_uuid: str | None) -> bool:
    """Release the card, **after** the terminal transition (FR-006.29).

    ``False`` when there is no card (a machine with none, ADR-023) or no live lease. Never
    raises on a missing lease: a crash between the terminal transition and this call leaves
    the lease to expire on its own, which is the whole reason the order is this way round.
    """
    if not gpu_uuid:
        return False
    from workers import lease as lease_module

    return lease_module.release(session, run_id, gpu_uuid)


def announce_terminal(
    emit: Any,
    run_id: str,
    *,
    status: str,
    stop_reason: str | None,
    detail: str | None = None,
    error_code: str | None = None,
) -> None:
    """Emit ``run_terminal`` - **after** the durable write, never in the same ``try``.

    FR-006.12 and FR-006.28: the socket is a projection that may be missed entirely, so a
    socket failure must not be able to roll back a recorded terminal state. The caller does
    the durable write first and calls this afterwards; a raise from here is logged and
    swallowed by the caller, not by this function, so the swallowing is visible at the call
    site.
    """
    if emit is None:
        return
    emit(
        run_id,
        EVENT_RUN_TERMINAL,
        {
            "run_id": run_id,
            "status": status,
            "stop_reason": stop_reason,
            "stop_detail": detail,
            "error_code": error_code,
        },
    )


__all__ = [
    "STATE_COMPLETED",
    "STATE_FAILED",
    "STATE_PENDING",
    "STATE_RUNNING",
    "TRANSITIONS",
    "announce_terminal",
    "mark_completed",
    "mark_failed",
    "mark_running",
    "release_lease",
]
