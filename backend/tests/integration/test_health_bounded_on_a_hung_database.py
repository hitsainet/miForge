"""``GET /api/health`` answers within seconds when PostgreSQL hangs (015 review, item 3).

The readiness probe calls it every 10 s, and since 015 it also writes and reads the legacy-name
store. A database that accepts connections and never answers must turn into "unavailable" and
``legacy_names_meta.error`` within the probe engine's bounds - never a probe that waits forever
while the next one piles on. The database here is a socket that accepts and then says nothing.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from core import legacy_names
from core.config import reset_settings
from db.session import PROBE_TIMEOUT_SECONDS, reset_engine
from services.legacy_names import store

#: Probe connect + store write + error-row write + store read, each bounded, plus slack.
BOUND_SECONDS = 6 * PROBE_TIMEOUT_SECONDS + 4


@pytest.fixture
def silent_database() -> Iterator[int]:
    """A TCP port that accepts connections and never sends a byte."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    held: list[socket.socket] = []
    stop = threading.Event()

    def accept() -> None:
        server.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = server.accept()
                held.append(conn)
            except OSError:
                continue

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield server.getsockname()[1]
    stop.set()
    thread.join(2)
    for conn in held:
        conn.close()
    server.close()


def _health_in_thread(results: list[Any]) -> threading.Thread:
    from api import health as health_module

    def run() -> None:
        started = time.monotonic()
        body = health_module.health_payload()
        results.append((time.monotonic() - started, body))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_a_hung_database_is_reported_within_the_bound(
    silent_database: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api import health as health_module

    monkeypatch.setenv(
        "DATABASE_URL", f"postgresql+psycopg://u:p@127.0.0.1:{silent_database}/miforge"
    )
    monkeypatch.setenv("MILLM_BASE_URL", "")  # a retired old name set: the write path runs too
    monkeypatch.setitem(health_module.DEPENDENCY_PROBES, "redis", lambda: {"status": "ok"})
    # The REAL Postgres probe, whatever an earlier test in this process left in the table,
    # and no in-flight record from an earlier test (order independence; CI runs a shard in
    # one worker).
    monkeypatch.setitem(health_module.DEPENDENCY_PROBES, "postgres", health_module._probe_postgres)
    health_module.reset_in_flight()
    reset_settings()
    reset_engine()
    store.reset_throttle()
    legacy_names.reset_logged()
    results: list[Any] = []
    try:
        threads = [_health_in_thread(results) for _ in range(3)]  # probes overlapping
        for thread in threads:
            thread.join(BOUND_SECONDS * 2)
        assert not any(t.is_alive() for t in threads), "a health call is still waiting"
    finally:
        reset_settings()
        reset_engine()
        store.reset_throttle()
    assert len(results) == 3
    for elapsed, body in results:
        assert elapsed < BOUND_SECONDS * 2, f"a health call took {elapsed:.1f} s"
        assert body["dependencies"]["postgres"]["status"] == "unavailable"
        assert body["legacy_names_meta"]["error"], "a hung store read must be meta.error"
        assert [i["old"] for i in body["legacy_names"]["retired_but_set"]] == ["MILLM_BASE_URL"]
