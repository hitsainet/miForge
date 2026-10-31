# Origin: miStudio (Onegaishimas/miStudio) backend/src/db/migration_safety.py @ 67b9ab1f
# Mode: adapt (docs/REUSE.md). Kept, with their reasons: the session guards set by set_config (no
# interpolated SQL), client_connection_check_interval on Postgres 14+, the blocker watch on a
# second connection, the deadline cancel, and SQLSTATE classification. Changed: SQLAlchemy (typed,
# so MyPy strict holds) instead of raw psycopg2 for the watch and the guards, so psycopg 3's
# `sqlstate` is read the same way; application_name is miforge-migrate; and env.py applies all of
# it ONLY on PostgreSQL, because the test suite and a laptop run migrations on SQLite.
"""Migrations that fail fast, say why, and never leave a waiter behind.

WHY THIS MODULE EXISTS (miStudio incident 2026-10-21, 05:05-05:16 ET)
---------------------------------------------------------------------
miStudio's API container ran a bare ``alembic upgrade`` at start. Its migration began with
``ALTER TABLE … ADD COLUMN`` (ACCESS EXCLUSIVE). A hung, two-day-old ``COPY`` held ACCESS SHARE on
the table, so the ALTER waited with no ``lock_timeout``, forever. Each killed attempt's ALTER STAYED
QUEUED in Postgres: a backend asleep on a lock does not notice its client is gone, so every orphan
kept its place in the lock queue, in front of every later reader of the table.

miForge's exposure is narrower and still real. The ``migrate`` init container of
``gitops/pilot/backend.yaml`` runs before any app container, so nothing can serve or work against an
old schema and no probe kills the wait — but with no ``lock_timeout`` a held lock left the pod in
``Init`` INDEFINITELY, with every container down and nothing in the log saying why.

What this module does for the migration session (``backend/alembic/env.py``, online mode,
PostgreSQL only):

* ``lock_timeout`` (``MIGRATION_LOCK_TIMEOUT``, default 15s): a lock wait errors out on its own.
* ``statement_timeout`` (``MIGRATION_STATEMENT_TIMEOUT``, default 150s): a statement that runs
  longer is cancelled and REPORTED. A migration that legitimately needs longer runs out of band with
  ``MIGRATION_STATEMENT_TIMEOUT=0``.
* ``idle_in_transaction_session_timeout`` (``MIGRATION_IDLE_TIMEOUT``, default 5min): a migration
  process that stops talking mid-transaction cannot hold its locks indefinitely.
* ``client_connection_check_interval`` (``MIGRATION_CLIENT_CHECK_INTERVAL``, default 2s, Postgres
  14+; production runs 15): a killed migration's backend notices the closed socket and aborts, even
  while it sleeps on a lock, so a killed init container does not leave its ALTER in the queue.
* :class:`BlockerWatch`: while the migration waits on a lock, a second connection reads
  ``pg_blocking_pids`` and ``pg_stat_activity`` and logs exactly who blocks it (pid, application,
  user, client, state, transaction age, query). On a lock timeout the last reading is in the error,
  with the ``pg_terminate_backend`` to run if the blocker is stale.
* A deadline (``MIGRATION_DEADLINE_AT``, set by ``python -m db.migrate``): past it, the watch
  cancels the migration's backend, so a run is bounded as a whole, not just per statement.

``python -m db.migrate upgrade`` (the init container's command) retries a lock timeout a bounded
number of times with backoff inside ``MIGRATION_DEADLINE`` (default 180s) and then EXITS NON-ZERO
with the blocker named. The kubelet restarts the init container with backoff, so a short-lived lock
is ridden out and a stale one is reported on every attempt. The deadline is pinned against the
Deployment's progress deadline by ``backend/tests/integration/test_migration_lock_safety.py``.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.pool import NullPool

# A child of the `alembic` logger, so alembic.ini's INFO-level `alembic` logger prints it.
logger = logging.getLogger("alembic.miforge.safety")

APPLICATION_NAME = "miforge-migrate"
WATCH_APPLICATION_NAME = "miforge-migrate-watch"

LOCK_NOT_AVAILABLE = "55P03"  # lock_timeout
QUERY_CANCELED = "57014"  # statement_timeout or pg_cancel_backend

DEFAULTS = {
    "MIGRATION_LOCK_TIMEOUT": "15s",
    "MIGRATION_STATEMENT_TIMEOUT": "150s",
    "MIGRATION_IDLE_TIMEOUT": "5min",
    "MIGRATION_CLIENT_CHECK_INTERVAL": "2s",
    "MIGRATION_ATTEMPTS": "4",
    "MIGRATION_RETRY_BACKOFF": "5s",
    "MIGRATION_DEADLINE": "180s",
}

#: Backoff never grows past this between two attempts.
MAX_BACKOFF_S = 30.0

_UNITS = {"us": 1e-6, "ms": 1e-3, "s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}
_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(us|ms|s|min|h|d)?\s*$")


def parse_duration(value: str) -> float:
    """Seconds in a Postgres duration (``15s``, ``4min``, ``500ms``; a bare number is MILLISECONDS,
    as Postgres reads it for these settings). ``0`` means "disabled"."""
    match = _DURATION.match(str(value))
    if not match:
        raise ValueError(f"not a duration: {value!r} (use e.g. 15s, 500ms, 4min)")
    number, unit = match.groups()
    return float(number) * _UNITS[unit or "ms"]


def _env(name: str) -> str:
    return os.environ.get(name, "").strip() or DEFAULTS[name]


@dataclass(frozen=True)
class MigrationSettings:
    lock_timeout: str
    statement_timeout: str
    idle_timeout: str
    client_check_interval: str
    attempts: int
    retry_backoff_s: float
    deadline_s: float

    @property
    def lock_timeout_s(self) -> float:
        return parse_duration(self.lock_timeout)

    @property
    def statement_timeout_s(self) -> float:
        return parse_duration(self.statement_timeout)

    def backoffs(self) -> list[float]:
        """The sleep before attempt 2, 3, … — doubling, capped at MAX_BACKOFF_S."""
        return [
            min(self.retry_backoff_s * 2**i, MAX_BACKOFF_S)
            for i in range(max(self.attempts - 1, 0))
        ]

    def lock_retry_budget_s(self) -> float:
        """Worst case for a run whose every attempt dies on its first lock wait."""
        return self.attempts * self.lock_timeout_s + sum(self.backoffs())


def settings_from_env() -> MigrationSettings:
    values = {name: _env(name) for name in DEFAULTS}
    for name in (
        "MIGRATION_LOCK_TIMEOUT",
        "MIGRATION_STATEMENT_TIMEOUT",
        "MIGRATION_IDLE_TIMEOUT",
        "MIGRATION_CLIENT_CHECK_INTERVAL",
        "MIGRATION_RETRY_BACKOFF",
        "MIGRATION_DEADLINE",
    ):
        parse_duration(values[name])  # refuse a malformed value here, with its name, not in SQL
    attempts = int(values["MIGRATION_ATTEMPTS"])
    if attempts < 1:
        raise ValueError(f"MIGRATION_ATTEMPTS must be >= 1, got {attempts}")
    return MigrationSettings(
        lock_timeout=values["MIGRATION_LOCK_TIMEOUT"],
        statement_timeout=values["MIGRATION_STATEMENT_TIMEOUT"],
        idle_timeout=values["MIGRATION_IDLE_TIMEOUT"],
        client_check_interval=values["MIGRATION_CLIENT_CHECK_INTERVAL"],
        attempts=attempts,
        retry_backoff_s=parse_duration(values["MIGRATION_RETRY_BACKOFF"]),
        deadline_s=parse_duration(values["MIGRATION_DEADLINE"]),
    )


# ----------------------------------------------------------------------------- session guards


@dataclass(frozen=True)
class SessionGuards:
    backend_pid: int
    applied: dict[str, str]


def apply_session_guards(
    connection: Connection, settings: MigrationSettings | None = None
) -> SessionGuards:
    """Set the migration session's timeouts (session-level, so they outlive this commit and hold
    for every transaction alembic opens). The transaction this opens is committed here."""
    settings = settings or settings_from_env()
    wanted = {
        "application_name": APPLICATION_NAME,
        "lock_timeout": settings.lock_timeout,
        "statement_timeout": settings.statement_timeout,
        "idle_in_transaction_session_timeout": settings.idle_timeout,
    }
    version = int(connection.execute(text("SHOW server_version_num")).scalar_one())
    if version >= 140000:
        wanted["client_connection_check_interval"] = settings.client_check_interval
    else:
        logger.warning(
            "Postgres %s has no client_connection_check_interval; a killed migration's "
            "backend is only reaped by lock_timeout",
            version,
        )
    applied: dict[str, str] = {}
    for name, value in wanted.items():
        # set_config, not an interpolated SET: the values come from the environment.
        applied[name] = str(
            connection.execute(
                text("SELECT set_config(:name, :value, false)"),
                {"name": name, "value": str(value)},
            ).scalar_one()
        )
    pid = int(connection.execute(text("SELECT pg_backend_pid()")).scalar_one())
    connection.commit()
    logger.info(
        "migration session pid=%s: lock_timeout=%s statement_timeout=%s "
        "idle_in_transaction_session_timeout=%s client_connection_check_interval=%s",
        pid,
        applied.get("lock_timeout", "n/a"),
        applied.get("statement_timeout", "n/a"),
        applied.get("idle_in_transaction_session_timeout", "n/a"),
        applied.get("client_connection_check_interval", "n/a"),
    )
    return SessionGuards(backend_pid=pid, applied=applied)


# ----------------------------------------------------------------------------- the blocker watch

_BLOCKERS_SQL = text(r"""
SELECT b.pid, b.application_name, b.usename, b.client_addr::text, b.state, b.backend_type,
       EXTRACT(EPOCH FROM now() - b.xact_start)::float,
       EXTRACT(EPOCH FROM now() - b.query_start)::float,
       left(regexp_replace(coalesce(b.query, ''), '\s+', ' ', 'g'), 300),
       b.wait_event_type
  FROM pg_stat_activity w
 CROSS JOIN LATERAL unnest(pg_blocking_pids(w.pid)) AS blocker(pid)
  JOIN pg_stat_activity b ON b.pid = blocker.pid
 WHERE w.pid = :pid
 ORDER BY b.xact_start NULLS LAST
""")


def _age(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    s = int(seconds)
    return f"{s // 86400}d{s % 86400 // 3600:02d}h{s % 3600 // 60:02d}m{s % 60:02d}s"


@dataclass(frozen=True)
class Blocker:
    pid: int
    application_name: str | None
    usename: str | None
    client_addr: str | None
    state: str | None
    backend_type: str | None
    xact_age_s: float | None
    query_age_s: float | None
    query: str
    waiting_on: str | None

    def describe(self) -> str:
        return (
            f"pid={self.pid} application_name={self.application_name!r} user={self.usename!r} "
            f"client={self.client_addr or 'local'} state={self.state!r} "
            f"xact_age={_age(self.xact_age_s)} query_age={_age(self.query_age_s)}"
            + (f" (itself waiting on {self.waiting_on})" if self.waiting_on else "")
            + f" query={self.query!r}"
        )


def watch_engine(url: str, application_name: str) -> Engine:
    """A one-connection engine for diagnostics: its own short statement_timeout, no pool."""
    return create_engine(
        url,
        poolclass=NullPool,
        connect_args={
            "application_name": application_name,
            "options": "-c statement_timeout=5000",
            "connect_timeout": 5,
        },
    )


@dataclass
class BlockerWatch:
    """Polls who blocks the migration's backend; cancels it past an optional deadline.

    Used as a context manager around ``context.run_migrations()``. A watch that cannot connect
    logs it and steps aside: it is diagnostics, never a reason for a migration to fail.
    """

    url: str
    backend_pid: int
    deadline_at: float | None = None
    poll_s: float = 0.5
    blockers: list[Blocker] = field(default_factory=list)
    cancelled_for_deadline: bool = False
    available: bool = False

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_logged: tuple[int, ...] = ()
        self._engine: Engine | None = None
        self._conn: Connection | None = None

    def __enter__(self) -> BlockerWatch:
        try:
            self._engine = watch_engine(self.url, WATCH_APPLICATION_NAME)
            self._conn = self._engine.connect().execution_options(isolation_level="AUTOCOMMIT")
            self.available = True
        except Exception as exc:  # diagnostics must not fail a migration
            logger.warning(
                "blocker watch unavailable (%s); a lock timeout will not name its blocker",
                type(exc).__name__,
            )
            return self
        self._thread = threading.Thread(
            target=self._run, name="migration-blocker-watch", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        try:
            if self._conn is not None:
                self._conn.close()
            if self._engine is not None:
                self._engine.dispose()
        except Exception:  # closing a diagnostics connection
            pass

    def read_blockers(self) -> list[Blocker]:
        assert self._conn is not None
        rows = self._conn.execute(_BLOCKERS_SQL, {"pid": self.backend_pid}).all()
        return [Blocker(*row) for row in rows]

    def _cancel(self) -> None:
        assert self._conn is not None
        self._conn.execute(text("SELECT pg_cancel_backend(:pid)"), {"pid": self.backend_pid})

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                if (
                    self.deadline_at is not None
                    and time.time() >= self.deadline_at
                    and not self.cancelled_for_deadline
                ):
                    self.cancelled_for_deadline = True
                    logger.error(
                        "migration deadline reached; cancelling the migration's backend pid=%s",
                        self.backend_pid,
                    )
                    self._cancel()
                blockers = self.read_blockers()
            except Exception as exc:  # a broken watch steps aside
                logger.warning("blocker watch stopped: %s", type(exc).__name__)
                return
            if blockers:
                self.blockers = blockers
                pids = tuple(b.pid for b in blockers)
                if pids != self._last_logged:
                    self._last_logged = pids
                    logger.warning(
                        "migration (pid=%s) is WAITING FOR A LOCK held or queued by:\n  %s",
                        self.backend_pid,
                        "\n  ".join(b.describe() for b in blockers),
                    )


# ----------------------------------------------------------------------------- errors


class MigrationSafetyError(RuntimeError):
    """A migration stopped by one of this module's guards. ``retryable`` says whether a later
    attempt can succeed without anyone intervening."""

    retryable = False


class MigrationLockTimeoutError(MigrationSafetyError):
    retryable = True

    def __init__(self, guards: SessionGuards, blockers: list[Blocker]):
        self.blockers = blockers
        if blockers:
            who = "\n  ".join(b.describe() for b in blockers)
            terminate = ", ".join(str(b.pid) for b in blockers)
            # An operator hint printed in the log, never executed.
            hint = f"SELECT pg_terminate_backend(pid) FROM unnest(ARRAY[{terminate}]) AS pid;"
            detail = f"Blocked by:\n  {who}\nIf a blocker is stale, end it: {hint}"
        else:
            detail = "The blocker could not be identified (the watch saw none before the timeout)."
        super().__init__(
            f"migration (pid={guards.backend_pid}) gave up waiting for a lock after "
            f"lock_timeout={guards.applied.get('lock_timeout')}. Nothing was applied: the "
            f"transaction rolled back and no statement is left waiting. {detail}"
        )


class MigrationStatementTimeoutError(MigrationSafetyError):
    def __init__(self, guards: SessionGuards):
        super().__init__(
            f"a migration statement ran longer than statement_timeout="
            f"{guards.applied.get('statement_timeout')} and was cancelled; nothing was applied. "
            f"If it legitimately needs longer, run it out of band: "
            f"MIGRATION_STATEMENT_TIMEOUT=0 python -m db.migrate upgrade"
        )


class MigrationDeadlineExceededError(MigrationSafetyError):
    def __init__(self, guards: SessionGuards, blockers: list[Blocker]):
        tail = (
            ("\n  last seen blocked by: " + "\n  ".join(b.describe() for b in blockers))
            if blockers
            else ""
        )
        super().__init__(
            f"migration (pid={guards.backend_pid}) passed its deadline (MIGRATION_DEADLINE) and "
            f"was cancelled; nothing was applied.{tail}"
        )


def sqlstate_of(exc: BaseException) -> str | None:
    """The SQLSTATE anywhere in an exception's chain (SQLAlchemy wraps psycopg's errors)."""
    seen: set[int] = set()
    stack: list[Any] = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        for attr in ("sqlstate", "pgcode"):
            code = getattr(current, attr, None)
            if isinstance(code, str) and code:
                return code
        stack.extend(
            [
                getattr(current, "orig", None),
                getattr(current, "__cause__", None),
                getattr(current, "__context__", None),
            ]
        )
    return None


def explain(
    exc: BaseException, guards: SessionGuards, watch: BlockerWatch
) -> MigrationSafetyError | None:
    """The guard error a database error stands for, or None when it is something else."""
    code = sqlstate_of(exc)
    error: MigrationSafetyError
    if code == LOCK_NOT_AVAILABLE:
        error = MigrationLockTimeoutError(guards, watch.blockers)
    elif code == QUERY_CANCELED and watch.cancelled_for_deadline:
        error = MigrationDeadlineExceededError(guards, watch.blockers)
    elif code == QUERY_CANCELED:
        error = MigrationStatementTimeoutError(guards)
    else:
        return None
    logger.error("%s", error)
    return error


def deadline_from_env() -> float | None:
    value = os.environ.get("MIGRATION_DEADLINE_AT", "").strip()
    return float(value) if value else None
