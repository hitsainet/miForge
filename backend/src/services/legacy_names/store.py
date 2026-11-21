"""Record old environment names in use; read them back for ``GET /api/health`` (FR-015.6).

Suite phase P7 removes the ``MILLM_*`` fallback only on seven days of evidence that no process
used one (FTID §3.3, CF-4, CF-18, CF-24, CF-25 m2). Three rules make that evidence honest:

- **Persisted, throttled.** Each process records its uses (API startup, every health call,
  every task, and a 60 s heartbeat in idle workers and beat). A row is written at most once a
  minute per process; ``last_seen`` survives restarts, and the health answer is the union of
  the live record and the rows of the last 24 hours.
- **A failure is evidence of nothing.** A failed read becomes ``legacy_names_meta.error``; a
  failed WRITE leaves an error row for its reporter, which the health answer reports until that
  reporter's next success. Neither ever shows as an empty list.
- **Names only.** Kind, old name, new name, state and times; never a value (ADR-020).

Never raises from :func:`record`: a store outage must not stop a task or the API.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session

from core.legacy_names import LegacyResolution, LegacyUse
from services.legacy_names.tables import USES, WRITE_ERRORS

log = logging.getLogger(__name__)

THROTTLE_SECONDS = 60.0
IN_USE_WINDOW = timedelta(hours=24)
WINDOW_HOURS = 24

Scope = Callable[[], AbstractContextManager[Session]]

_lock = threading.Lock()
#: ``(kind, old, state)`` -> monotonic time of this process's last successful write.
_last_written: dict[tuple[str, str, str], float] = {}


def _default_scope() -> AbstractContextManager[Session]:
    from db.session import session_scope

    return session_scope()


def api_reporter() -> str:
    return f"api@{socket.gethostname()}"


def beat_reporter() -> str:
    return f"beat@{socket.gethostname()}"


def as_utc(value: datetime) -> datetime:
    """SQLite returns naive datetimes; every time here is UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _stamp(value: datetime) -> str:
    return as_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def reset_throttle() -> None:
    """Forget this process's write times. For tests (and a simulated restart)."""
    with _lock:
        _last_written.clear()


def _insert(session: Session, table: sa.Table) -> Any:
    if session.get_bind().dialect.name == "postgresql":
        return postgresql.insert(table)
    return sqlite.insert(table)


def _upsert_uses(session: Session, uses: list[LegacyUse], now: datetime) -> int:
    for use in uses:
        statement = _insert(session, USES).values(
            kind=use.kind,
            old_name=use.old,
            state=use.state,
            new_name=use.new,
            first_seen=now,
            last_seen=now,
        )
        statement = statement.on_conflict_do_update(
            index_elements=["kind", "old_name", "state"],
            set_={
                "last_seen": statement.excluded.last_seen,
                "new_name": statement.excluded.new_name,
            },
        )
        session.execute(statement)
    return len(uses)


def _clear_error(session: Session, reporter: str, now: datetime) -> None:
    session.execute(
        sa.update(WRITE_ERRORS)
        .where(WRITE_ERRORS.c.reporter == reporter, WRITE_ERRORS.c.cleared_at.is_(None))
        .values(cleared_at=now)
    )


def report_write_failure(
    reporter: str, exc: BaseException, now: datetime, *, scope: Scope | None = None
) -> bool:
    """Upsert this reporter's error row in its own short transaction. Never raises."""
    try:
        with (scope or _default_scope)() as session:
            statement = _insert(session, WRITE_ERRORS).values(
                reporter=reporter[:128],
                error=type(exc).__name__[:64],
                failed_at=now,
                cleared_at=None,
            )
            statement = statement.on_conflict_do_update(
                index_elements=["reporter"],
                set_={
                    "error": statement.excluded.error,
                    "failed_at": statement.excluded.failed_at,
                    "cleared_at": None,
                },
            )
            session.execute(statement)
        return True
    except Exception as second:
        log.warning(
            "legacy-name write failure could not be recorded",
            extra={
                "code": "legacy_names_error_unrecordable",
                "reporter": reporter,
                "detail": type(second).__name__,
            },
        )
        return False


def record(
    resolution: LegacyResolution,
    *,
    reporter: str,
    now: datetime | None = None,
    clock: Callable[[], float] | None = None,
    scope: Scope | None = None,
) -> int:
    """Write this process's uses whose row was last written over a minute ago.

    Returns the number of rows written (0 when throttled, when nothing is in use - no session
    is opened then - or when the write failed). On failure the throttle is NOT advanced, so the
    next heartbeat retries within a minute, and the failure is recorded for the health answer.
    """
    tick = (clock or time.monotonic)()
    with _lock:
        due = [
            use
            for use in resolution.uses
            if tick - _last_written.get((use.kind, use.old, use.state), float("-inf"))
            >= THROTTLE_SECONDS
        ]
    if not due:
        return 0
    stamp = now or datetime.now(UTC)
    try:
        with (scope or _default_scope)() as session:
            written = _upsert_uses(session, due, stamp)
            _clear_error(session, reporter, stamp)
    except Exception as exc:
        log.warning(
            "could not record legacy environment names",
            extra={
                "code": "legacy_names_store_unavailable",
                "reporter": reporter,
                "detail": type(exc).__name__,
            },
        )
        report_write_failure(reporter, exc, stamp, scope=scope)
        return 0
    with _lock:
        for use in due:
            _last_written[(use.kind, use.old, use.state)] = tick
    return written


@dataclass(frozen=True)
class StoredUse:
    kind: str
    old: str
    new: str | None
    state: str
    last_seen: datetime


@dataclass(frozen=True)
class StoredUses:
    uses: tuple[StoredUse, ...] = ()
    #: ``(reporter, error type)`` of uncleared write failures in the window.
    write_errors: tuple[tuple[str, str], ...] = ()


def recent(session: Session, *, now: datetime, window: timedelta = IN_USE_WINDOW) -> StoredUses:
    """Rows seen within ``window`` of ``now``, and uncleared write failures in it."""
    since = now - window
    if session.get_bind().dialect.name == "postgresql":
        # A stuck read must not hold the readiness probe, which calls this every 10 s.
        session.execute(sa.text("SET LOCAL statement_timeout = 2000"))
    uses = session.execute(
        sa.select(
            USES.c.kind, USES.c.old_name, USES.c.new_name, USES.c.state, USES.c.last_seen
        ).where(USES.c.last_seen >= since)
    ).all()
    errors = session.execute(
        sa.select(WRITE_ERRORS.c.reporter, WRITE_ERRORS.c.error).where(
            WRITE_ERRORS.c.cleared_at.is_(None), WRITE_ERRORS.c.failed_at >= since
        )
    ).all()
    return StoredUses(
        uses=tuple(
            StoredUse(r.kind, r.old_name, r.new_name, r.state, as_utc(r.last_seen)) for r in uses
        ),
        write_errors=tuple((r.reporter, r.error) for r in errors),
    )


def read_recent(*, now: datetime | None = None, scope: Scope | None = None) -> StoredUses:
    with (scope or _default_scope)() as session:
        return recent(session, now=now or datetime.now(UTC))


#: The health block's four lists (CF-4); the served key for ``conflict`` uses is plural.
_LISTS = {
    "in_use": "in_use",
    "also_set": "also_set",
    "conflict": "conflicts",
    "retired_but_set": "retired_but_set",
}


def block(
    live: LegacyResolution, stored: StoredUses | None, *, now: datetime
) -> dict[str, list[dict[str, Any]]]:
    """The ``legacy_names`` block: exactly four lists, each sorted by ``old``.

    ``in_use`` entries carry ``kind``, ``old``, ``new`` and ``last_seen`` (a live use is seen
    now); the other three carry ``old`` and ``new``.
    """
    seen: dict[tuple[str, str, str], tuple[str | None, datetime]] = {}
    for use in live.uses:
        seen[(use.kind, use.old, use.state)] = (use.new, now)
    for row in stored.uses if stored else ():
        key = (row.kind, row.old, row.state)
        if key in seen:
            new, last_seen = seen[key]
            seen[key] = (new, max(last_seen, row.last_seen))
        else:
            seen[key] = (row.new, row.last_seen)
    out: dict[str, list[dict[str, Any]]] = {served: [] for served in _LISTS.values()}
    for (kind, old, state), (new, last_seen) in seen.items():
        served = _LISTS.get(state)
        if served is None:
            continue
        if state == "in_use":
            out[served].append(
                {"kind": kind, "old": old, "new": new, "last_seen": _stamp(last_seen)}
            )
        else:
            out[served].append({"old": old, "new": new})
    for listing in out.values():
        listing.sort(key=lambda item: str(item["old"]))
    return out


def meta(*, read_error: BaseException | None, stored: StoredUses | None) -> dict[str, Any]:
    """``legacy_names_meta``: always both keys; ``error`` names a failed read or write."""
    error: str | None = None
    if read_error is not None:
        error = type(read_error).__name__
    elif stored is not None and stored.write_errors:
        error = "; ".join(sorted(f"write failed in {r}: {e}" for r, e in stored.write_errors))
    return {"error": error, "in_use_window_hours": WINDOW_HOURS}


# ------------------------------------------------------------------------------- heartbeat


@dataclass
class Heartbeat:
    """One daemon thread per process recording every ``interval`` seconds until stopped."""

    resolution: LegacyResolution
    reporter: str
    interval: float
    sleep: Callable[[float], bool] | None = None
    record_fn: Callable[..., int] | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    ticks: int = 0

    def _loop(self) -> None:
        wait = self.sleep or self._stop.wait
        while not self._stop.is_set():
            (self.record_fn or record)(self.resolution, reporter=self.reporter)
            self.ticks += 1
            if wait(self.interval):
                break

    def start(self) -> Heartbeat:
        self.thread = threading.Thread(
            target=self._loop, name="legacy-names-heartbeat", daemon=True
        )
        self.thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout)


_heartbeat: Heartbeat | None = None


def start_heartbeat(
    resolution: LegacyResolution,
    *,
    reporter: str,
    interval: float = THROTTLE_SECONDS,
    sleep: Callable[[float], bool] | None = None,
    record_fn: Callable[..., int] | None = None,
) -> Heartbeat:
    """Start this process's heartbeat; a second call returns the running one (idempotent)."""
    global _heartbeat
    with _lock:
        if (
            _heartbeat is not None
            and _heartbeat.thread is not None
            and _heartbeat.thread.is_alive()
        ):
            return _heartbeat
        _heartbeat = Heartbeat(resolution, reporter, interval, sleep=sleep, record_fn=record_fn)
    return _heartbeat.start()


def stop_heartbeat() -> None:
    """Stop and forget this process's heartbeat. For tests and shutdown."""
    global _heartbeat
    with _lock:
        running, _heartbeat = _heartbeat, None
    if running is not None:
        running.stop()
