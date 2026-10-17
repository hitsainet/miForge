"""The three Socket.IO event names F6 emits - **declared once**  (FPRD §7; AC-006.45).

Three strings, in one place, and **no emit site carries an inline literal.**

## Why this file exists at all

Until 2026-10-13 this feature's documents named the room and three event *kinds* and no
strings, and **F9 had invented ``step | score | save | signal``** against them and then
withdrew the four names. Nothing in the project named them. So any subscription written
against either guess would have been **silently dead while looking connected**: the emitter
looks healthy, the subscriber looks connected, and nothing arrives. That failure is
invisible from both ends, which is why the names are constants and why the emitter test
asserts the **strings, the payload and the count** (M-17).

## What F6 does not define here

``score`` and ``retention_floor_breached`` are F4's and ``checkpoint_saved`` is F5's
(:data:`services.checkpoints.service.CHECKPOINT_SAVED_EVENT`). They share the same room and
are **cited, not redefined**: a second spelling of one event is the same defect one layer
down.

The room is the foundation's :func:`api.sockets.room_for` - ``miforge/run/<id>``. F6 adds no
room and no channel.

## The sync bridge, and why F6 had to build it

``api.sockets.emit_run_event`` is a **coroutine**, while F1's ``StepWriter``, F4's score
recorder, F5's save and F6's boundary all take a synchronous ``emit(run_id, event,
payload)`` - deliberately, because the interfaces are sync and every worker is
``--pool=solo -c 1``. Nothing in the codebase bridged the two at that arity: F3 has a
two-argument bridge bound to one run, and there was no three-argument one, because
``run_step_loop`` refused before it ever reached a call site that needed it.

**Handing those features the coroutine function directly would have been a silent no-op**:
``emit(...)`` would build a coroutine, nobody would await it, Python would warn at garbage
collection and no event would ever arrive. The emitter would look wired and the subscriber
would look connected - the same invisible-from-both-ends failure the named constants above
exist to prevent. :func:`run_emitter` is the bridge, modelled on F3's.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)

#: One per completed boundary: the step, the round, every condition's evaluation.
EVENT_BOUNDARY = "boundary"

#: One per prompt refresh (R-02.26): the mode, the rule hash, the new set's identity.
EVENT_REFRESH = "refresh"

#: One per terminal transition: the final state and the ``stop_reason`` or error code.
EVENT_RUN_TERMINAL = "run_terminal"

#: F6's three, as the reachability test enumerates them. A fourth event added without a
#: line here is an event no test asserts, which is how a subscriber goes quiet.
RUN_EVENTS: tuple[str, ...] = (EVENT_BOUNDARY, EVENT_REFRESH, EVENT_RUN_TERMINAL)

#: In-flight emit tasks, kept referenced so a scheduled task is not garbage-collected
#: mid-flight (RUF006). A dropped task loses the emit **silently**, and "silently" is the
#: whole thing this module exists to avoid; the record is durable either way.
_PENDING: set[asyncio.Task[None]] = set()


def run_emitter() -> Any:
    """A synchronous ``emit(run_id, event, payload)`` onto ``miforge/run/<id>``.

    Schedules the coroutine on a running loop when there is one and runs it to completion
    when there is not - the same two cases F3's ``socket_emitter`` handles, at the arity F1,
    F4, F5 and F6 all take.

    **A failed emit is logged, never raised.** The record already holds everything
    (ADR-007); losing the socket must not fail a run whose work is durable. It must not be
    *silent* either, or the first symptom is a screen that stopped updating for a reason
    nobody recorded.
    """

    def emit(run_id: str, event: str, payload: dict[str, Any]) -> None:
        from api.sockets import emit_run_event

        coroutine = emit_run_event(run_id, event, payload)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(coroutine)
            except Exception:  # pragma: no cover - transport failure, not a record failure
                coroutine.close()
                log.warning(
                    "run event not delivered; the record is durable",
                    extra={"code": "run_event_not_delivered", "run_id": run_id, "event": event},
                )
            return
        task = loop.create_task(coroutine)
        _PENDING.add(task)
        task.add_done_callback(_PENDING.discard)

    return emit


__all__ = [
    "EVENT_BOUNDARY",
    "EVENT_REFRESH",
    "EVENT_RUN_TERMINAL",
    "RUN_EVENTS",
    "run_emitter",
]
