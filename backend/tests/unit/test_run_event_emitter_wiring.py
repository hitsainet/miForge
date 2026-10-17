"""The production run-event emitter must be the SYNC BRIDGE, not the coroutine  (ADR-016 #1).

``api.sockets.emit_run_event`` is an ``async def``. F1's ``StepWriter``, F4's score
recorder, F5's save and F6's boundary all take a **synchronous** ``emit(run_id, event,
payload)`` and call it bare, discarding the result. So handing those features the coroutine
function is a **silent no-op**: ``emit(...)`` builds a coroutine, nobody awaits it, and no
event ever arrives. ``training/run_events.py`` exists entirely to prevent that, and its
module docstring says so - *"the emitter would look wired and the subscriber would look
connected"*, invisible from both ends.

It was wired the broken way. ``training.tasks._emitter`` returned ``emit_run_event``
itself, :func:`training.run_events.run_emitter` had **no production caller and no test at
all**, and the whole suite was green: every emit test injects its own recorder, so none of
them ever asked what production passes.

Found by ``0xcc/scripts/check_reachability.py``, which reports a public symbol that only
tests reference - ``run_emitter`` was referenced *nowhere*.

**This is the control.** Point ``_emitter`` back at ``emit_run_event`` and both tests below
must go red. They assert the payload, the room and the **call count**, not merely "was
called": a count is what distinguishes a delivered event from a coroutine that was built,
dropped and warned about at garbage collection.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from training import run_events, tasks


def test_the_production_emitter_is_not_a_coroutine_function() -> None:
    """The arity is right either way; only this tells the two apart."""
    emitter = tasks._emitter()
    assert emitter is not None, "the transport is importable in the test environment"
    assert not inspect.iscoroutinefunction(emitter), (
        "training.tasks._emitter() returned a coroutine function. Every emit site calls it "
        "bare and discards the result, so every run event would be silently lost. Return "
        "training.run_events.run_emitter() instead."
    )


def test_the_production_emitter_actually_delivers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Payload, room AND count - asserted through the real socket layer.

    ``sio.emit`` is itself a coroutine, so it records a call **only if something drove the
    coroutine to completion**. That is what makes this a delivery test rather than a type
    test: against the broken wiring the recorder stays empty.
    """
    from api import sockets

    seen: list[tuple[str, dict[str, Any], str | None]] = []

    async def _record(event: str, payload: dict[str, Any], room: str | None = None) -> None:
        seen.append((event, payload, room))

    monkeypatch.setattr(sockets.sio, "emit", _record)

    emitter = tasks._emitter()
    emitter("run-42", run_events.EVENT_BOUNDARY, {"step": 7, "reason": "test_threshold"})

    assert len(seen) == 1, (
        f"expected exactly one delivered event, got {len(seen)}. Zero means the coroutine "
        f"was built and never awaited - the silent no-op run_events.py exists to prevent."
    )
    event, payload, room = seen[0]
    assert event == run_events.EVENT_BOUNDARY
    assert payload == {"step": 7, "reason": "test_threshold"}
    assert room == sockets.room_for("run-42")


def test_the_bridge_has_a_production_caller() -> None:
    """Reachability, asserted on the live object rather than by grep.

    A grep proves a call exists; this proves the production path *is* the bridge. Identity,
    because `run_emitter()` returns a fresh closure each time - so the check is that the
    emitter behaves as the bridge does, which the delivery test above pins, plus that the
    module is reached at all.
    """
    emitter = tasks._emitter()
    assert emitter.__qualname__.startswith("run_emitter"), (
        f"production emitter is {emitter.__qualname__!r}, not run_events.run_emitter's "
        f"closure. The bridge is unwired again."
    )
