"""Review round 2 of 015 P0: the health path's probe engine, on the engine production uses.

Each test needs PostgreSQL (``MIFORGE_TEST_DATABASE_URL``, CI's ``postgres`` job): the bounds
under test are libpq connection options and the one-connection pool, both of which exist only
on the PostgreSQL branch of ``db.session.get_probe_engine``.

* A lock held on the legacy-name store must turn the store read into ``meta.error`` within
  the lock/statement bound, not hold the readiness probe for as long as the lock lives.
* An exhausted probe pool must answer within the pool wait, not SQLAlchemy's 30 s default.
* A pooled connection that has gone silent (a black-holed TCP path after a database node is
  lost) must not make a HEALTHY database read "unavailable" on every later probe.
"""

from __future__ import annotations

import select
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import Engine

from core import legacy_names
from core.config import reset_settings
from db.session import PROBE_TIMEOUT_SECONDS, get_probe_engine, reset_engine
from services.legacy_names import store
from services.legacy_names.tables import USES
from tests.support.db import migrated_engine


@pytest.fixture
def pg_engine(tmp_path: Path) -> Iterator[Engine]:
    engine = migrated_engine(tmp_path)
    if engine.dialect.name != "postgresql":
        engine.dispose()
        pytest.skip("the probe engine's bounds are PostgreSQL options (CI's postgres job)")
    yield engine
    engine.dispose()


def _configure(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    from api import health as health_module

    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("MILLM_BASE_URL", "")  # a retired old name set: the store write runs
    monkeypatch.delenv("MISERVER_BASE_URL", raising=False)
    monkeypatch.setitem(health_module.DEPENDENCY_PROBES, "redis", lambda: {"status": "ok"})
    reset_settings()
    reset_engine()
    store.reset_throttle()
    legacy_names.reset_logged()


def _health(timeout: float) -> tuple[float, dict[str, Any]] | None:
    from api import health as health_module

    out: list[tuple[float, dict[str, Any]]] = []

    def run() -> None:
        started = time.monotonic()
        body = health_module.health_payload()
        out.append((time.monotonic() - started, body))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    return out[0] if out else None


@pytest.fixture
def cleanup() -> Iterator[None]:
    yield
    reset_settings()
    reset_engine()
    store.reset_throttle()


def test_a_lock_on_the_store_is_bounded(
    pg_engine: Engine, monkeypatch: pytest.MonkeyPatch, cleanup: None
) -> None:
    """Pins the ``lock_timeout``/``statement_timeout`` options (mutation M12)."""
    _configure(monkeypatch, pg_engine.url.render_as_string(hide_password=False))
    with pg_engine.connect() as holder:
        holder.execute(sa.text(f"LOCK TABLE {USES.name} IN ACCESS EXCLUSIVE MODE"))
        answered = _health(4 * PROBE_TIMEOUT_SECONDS + 4)
        # While the lock is STILL held, the server must end the blocked store write on its
        # own (lock/statement time-out). Without it the write thread waits as long as the lock
        # lives and the client-side deadline is the only bound (control M12 survived on the
        # read alone, which store.recent() also bounds with SET LOCAL statement_timeout).
        from api import health as health_module

        write = health_module._in_flight.get("legacy_names_write")
        ended = write is not None and write.wait(2 * PROBE_TIMEOUT_SECONDS + 2)
        holder.rollback()
    assert ended, "the blocked store write was not ended by the server while the lock lived"
    assert answered is not None, "a lock on the store held the health answer past its bound"
    _, body = answered
    assert body["dependencies"]["postgres"]["status"] == "ok", body["dependencies"]
    assert body["legacy_names_meta"]["error"], "a store read blocked by a lock must be meta.error"
    # The SERVER ended the blocked read (lock/statement time-out, 2 s), before the client-side
    # deadline (5 s) had to: otherwise the stuck thread lives as long as the lock and every
    # later probe answers "still running". Control M12 (drop the options) survived until this.
    assert body["legacy_names_meta"]["error"] not in (
        "ProbeTimeoutError",
        "ProbeStillRunningError",
    ), body["legacy_names_meta"]


def test_an_exhausted_probe_pool_answers_within_the_wait(
    pg_engine: Engine, monkeypatch: pytest.MonkeyPatch, cleanup: None
) -> None:
    """Pins ``pool_timeout`` (mutation M10). Design-neutral: whether a busy pool reads as
    unavailable or the engine opens another connection, the answer must come back in time."""
    _configure(monkeypatch, pg_engine.url.render_as_string(hide_password=False))
    engine = get_probe_engine()
    size = getattr(engine.pool, "size", None)  # NullPool keeps nothing to exhaust
    held = [engine.connect() for _ in range(size() if callable(size) else 0)]
    try:
        answered = _health(4 * PROBE_TIMEOUT_SECONDS + 4)
    finally:
        for connection in held:
            connection.close()
    assert answered is not None, "health waited on the probe pool past its bound"


def test_reset_engine_discards_the_probe_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: None
) -> None:
    """Pins the probe-engine half of ``reset_engine`` (mutation M14): a test that resets the
    engines must not inherit the previous test's pooled probe connection. Any engine."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'probe.db'}")
    reset_settings()
    reset_engine()
    first = get_probe_engine()
    with first.connect() as connection:
        connection.execute(sa.text("SELECT 1"))
    assert first.pool.checkedin() == 1
    reset_engine()
    assert first.pool.checkedin() == 0, "reset_engine left the probe pool's connection open"
    assert get_probe_engine() is not first, "reset_engine kept the old probe engine"


class _FreezingProxy:
    """TCP relay to the test database whose EXISTING connections can be frozen (no byte
    relayed either way, sockets kept open) while new connections pass normally - a pooled
    connection whose path went dark while the database itself is healthy."""

    def __init__(self, host: str, port: int) -> None:
        self.upstream = (host, port)
        self.frozen: list[list[bool]] = []
        self.server = socket.socket()
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(32)
        self.port = self.server.getsockname()[1]
        self.stop = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def _relay(self, a: socket.socket, b: socket.socket, flag: list[bool]) -> None:
        try:
            while not self.stop.is_set():
                ready, _, _ = select.select([a], [], [], 0.05)
                if flag[0] or not ready:
                    continue
                data = a.recv(65536)
                if not data:
                    break
                b.sendall(data)
        except OSError:
            pass

    def _accept(self) -> None:
        self.server.settimeout(0.2)
        while not self.stop.is_set():
            try:
                client, _ = self.server.accept()
            except OSError:
                continue
            upstream = socket.create_connection(self.upstream)
            flag = [False]
            self.frozen.append(flag)
            threading.Thread(target=self._relay, args=(client, upstream, flag), daemon=True).start()
            threading.Thread(target=self._relay, args=(upstream, client, flag), daemon=True).start()

    def freeze_existing(self) -> None:
        for flag in self.frozen:
            flag[0] = True

    def close(self) -> None:
        self.stop.set()
        self.server.close()


def test_a_silent_pooled_connection_does_not_fail_a_healthy_database(
    pg_engine: Engine, monkeypatch: pytest.MonkeyPatch, cleanup: None
) -> None:
    """The review-round-2 finding: with ``pool_size=1, max_overflow=0`` one silent pooled
    connection starves every later probe - each reads "unavailable" (QueuePool TimeoutError)
    although the database answers every new connection."""
    url = pg_engine.url
    proxy = _FreezingProxy(url.host or "127.0.0.1", url.port or 5432)
    try:
        _configure(
            monkeypatch,
            url.set(host="127.0.0.1", port=proxy.port).render_as_string(hide_password=False),
        )
        warm = _health(4 * PROBE_TIMEOUT_SECONDS + 4)
        assert warm is not None and warm[1]["dependencies"]["postgres"]["status"] == "ok"
        proxy.freeze_existing()  # the pooled connection goes dark; the database does not
        _health(PROBE_TIMEOUT_SECONDS + 1)  # may hang on the dark connection; not judged
        later = [_health(4 * PROBE_TIMEOUT_SECONDS + 4) for _ in range(2)]
    finally:
        proxy.close()
    for answered in later:
        assert answered is not None, "a later health call hung"
        postgres = answered[1]["dependencies"]["postgres"]
        assert postgres["status"] == "ok", f"a healthy database read {postgres}"
