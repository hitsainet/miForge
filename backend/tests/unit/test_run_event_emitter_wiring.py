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


class _Recorder:
    """Stands in for the worker's write-only ``RedisManager``: records what it was asked to
    publish. Synchronous, like the real one - which is the point of the fix."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, dict[str, Any], str | None, str | None]] = []

    def emit(
        self, event: str, data: dict[str, Any], namespace: str | None = None, room: Any = None
    ) -> None:
        self.seen.append((event, data, namespace, room))


def test_the_production_emitter_actually_delivers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Payload, room AND count - through the publisher a worker actually uses.

    A worker has no event loop, so the bridge publishes through
    ``api.sockets.publish_from_worker``. Against the coroutine-function wiring the recorder
    stays empty; against the ``asyncio.run`` per event wiring it stays empty too, because
    that path goes through ``sio`` instead (see ``test_worker_emits_reach_redis.py`` for what
    it did against a real Redis).
    """
    from api import sockets

    recorder = _Recorder()
    monkeypatch.setattr(sockets, "_worker_publisher", lambda: recorder)

    emitter = tasks._emitter()
    for step in (7, 8, 9):
        emitter("run-42", run_events.EVENT_BOUNDARY, {"step": step, "reason": "test_threshold"})

    assert len(recorder.seen) == 3, (
        f"expected exactly three published events, got {len(recorder.seen)}. Zero means the "
        f"worker path is not the write-only publisher."
    )
    event, payload, namespace, room = recorder.seen[0]
    assert event == run_events.EVENT_BOUNDARY
    assert payload == {"step": 7, "reason": "test_threshold"}
    assert namespace == "/"
    assert room == sockets.room_for("run-42")
    assert [p["step"] for _, p, _, _ in recorder.seen] == [7, 8, 9]


def test_a_failed_publish_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The record is durable (ADR-007): a lost socket must not fail the run - and must not
    be silent either."""
    from api import sockets

    class Down:
        def emit(self, *args: Any, **kwargs: Any) -> None:
            raise ConnectionError("redis down")

    monkeypatch.setattr(sockets, "_worker_publisher", lambda: Down())
    with caplog.at_level("WARNING"):
        tasks._emitter()("run-42", run_events.EVENT_BOUNDARY, {"step": 1})
    assert any(getattr(r, "code", None) == "run_event_not_delivered" for r in caplog.records)


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
