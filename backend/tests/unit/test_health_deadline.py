"""Every external call on the readiness path has a client-side deadline, single flight (015
review round 2).

libpq has no read time-out, so a database peer that keeps acknowledging and never answers holds
a call past every server-side and TCP bound. ``api.health.call_with_deadline`` answers after
``PROBE_DEADLINE_SECONDS`` whatever the call is doing, and while one call of a kind is still
stuck the next one answers at once instead of starting another thread.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from api import health as health_module
from core.config import reset_settings
from services.legacy_names import store

DEADLINE = 0.3


@pytest.fixture(autouse=True)
def _short_deadline(monkeypatch: pytest.MonkeyPatch) -> Iterator[threading.Event]:
    release = threading.Event()
    monkeypatch.setattr(health_module, "PROBE_DEADLINE_SECONDS", DEADLINE)
    monkeypatch.setattr(health_module, "_in_flight", {})
    monkeypatch.setitem(health_module.DEPENDENCY_PROBES, "redis", lambda: {"status": "ok"})
    monkeypatch.setattr(store, "record", lambda *a, **k: 0)
    monkeypatch.setattr(store, "read_recent", lambda **_: store.StoredUses())
    reset_settings()
    yield release
    release.set()  # free any thread a test left blocked
    reset_settings()


def _timed() -> tuple[float, dict[str, Any]]:
    started = time.monotonic()
    body = health_module.health_payload()
    return time.monotonic() - started, body


def test_a_hung_probe_answers_within_the_deadline(
    monkeypatch: pytest.MonkeyPatch, _short_deadline: threading.Event
) -> None:
    monkeypatch.setitem(
        health_module.DEPENDENCY_PROBES, "postgres", lambda: _short_deadline.wait(30) and {}
    )
    elapsed, body = _timed()
    assert elapsed < DEADLINE * 5, f"health waited {elapsed:.2f} s on a hung probe"
    postgres = body["dependencies"]["postgres"]
    assert postgres["status"] == "unavailable" and "no answer within" in postgres["detail"]


def test_a_stuck_call_is_not_started_twice(
    monkeypatch: pytest.MonkeyPatch, _short_deadline: threading.Event
) -> None:
    calls: list[float] = []

    def hung() -> dict[str, Any]:
        calls.append(time.monotonic())
        _short_deadline.wait(30)
        return {"status": "ok", "detail": None}

    monkeypatch.setitem(health_module.DEPENDENCY_PROBES, "postgres", hung)
    _timed()
    elapsed, body = _timed()
    assert len(calls) == 1, "a second probe thread was started while the first was stuck"
    assert elapsed < DEADLINE * 2, "the second call waited although nothing was started"
    assert "has not returned" in body["dependencies"]["postgres"]["detail"]
    _short_deadline.set()
    deadline = time.monotonic() + 5
    while not health_module._in_flight["probe:postgres"].is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
    _, body = _timed()
    assert body["dependencies"]["postgres"]["status"] == "ok"
    assert len(calls) == 2


def test_a_hung_store_read_is_meta_error(
    monkeypatch: pytest.MonkeyPatch, _short_deadline: threading.Event
) -> None:
    monkeypatch.setitem(health_module.DEPENDENCY_PROBES, "postgres", lambda: {"status": "ok"})

    def hung(**_: Any) -> store.StoredUses:
        _short_deadline.wait(30)
        return store.StoredUses()

    monkeypatch.setattr(store, "read_recent", hung)
    elapsed, body = _timed()
    assert elapsed < DEADLINE * 5
    assert body["legacy_names_meta"]["error"] == "ProbeTimeoutError"


def test_a_hung_store_write_does_not_hold_health(
    monkeypatch: pytest.MonkeyPatch,
    _short_deadline: threading.Event,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setitem(health_module.DEPENDENCY_PROBES, "postgres", lambda: {"status": "ok"})
    monkeypatch.setattr(store, "record", lambda *a, **k: _short_deadline.wait(30) and 0)
    with caplog.at_level(logging.WARNING, logger="api.health"):
        elapsed, body = _timed()
    assert elapsed < DEADLINE * 5
    assert body["dependencies"]["postgres"]["status"] == "ok"
    assert any(r.__dict__.get("code") == "legacy_names_store_unavailable" for r in caplog.records)


def test_a_stuck_postgres_call_reads_unavailable_never_ok(
    monkeypatch: pytest.MonkeyPatch, _short_deadline: threading.Event
) -> None:
    """Single flight never hands back an earlier answer: while a call is stuck, Postgres reads
    unavailable on every health call, and a finished ``ok`` is not reused for a later failure."""
    answers = iter([{"status": "ok", "detail": None}])

    def probe() -> dict[str, Any]:
        try:
            return next(answers)
        except StopIteration:
            _short_deadline.wait(30)  # the database hangs from now on
            return {"status": "ok", "detail": None}

    monkeypatch.setitem(health_module.DEPENDENCY_PROBES, "postgres", probe)
    assert _timed()[1]["dependencies"]["postgres"]["status"] == "ok"
    for _ in range(3):
        _, body = _timed()
        assert body["dependencies"]["postgres"]["status"] == "unavailable", body["dependencies"]
