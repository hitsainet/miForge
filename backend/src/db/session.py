"""Engine and session management  (task 4.1; ADR-003).

One engine per process, built lazily from :func:`core.config.get_settings`. Lazily
because importing this module must not open a socket: the no-sibling-import guard, the
GPU ratchet and the schema-sync test all import across the tree, and a module that
connects on import turns every one of them into a test that needs a database.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from core.config import get_settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None

#: The readiness probe calls ``GET /api/health`` every 10 s (``gitops/pilot/backend.yaml``).
#: Every database call on that path goes through :func:`probe_session_scope`: connect and
#: each statement are bounded, and a one-connection pool refuses a second waiter after the
#: same time, so a hung database answers "unavailable" within seconds instead of stacking
#: one stuck probe on another (015 review, 2026-11-08).
PROBE_TIMEOUT_SECONDS = 2
_probe_engines: dict[str, Engine] = {}


def make_engine(url: str, **kwargs: object) -> Engine:
    """Build an engine for ``url``.

    Exported so tests can build one against SQLite or against a real PostgreSQL named by
    the environment, using **the same constructor production uses**. A test that builds
    its tables a different way is a fixture whose fields agree by construction (ADR-016).
    """
    if url.startswith("sqlite"):
        # SQLite does not enforce foreign keys unless asked, and a FK the tests do not
        # enforce is a FK the migration is not actually tested against.
        from sqlalchemy import event

        engine = create_engine(url, **kwargs)

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_connection: object, _record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine
    return create_engine(url, pool_pre_ping=True, **kwargs)


def get_engine() -> Engine:
    """The process-wide engine, built once from settings."""
    global _engine
    if _engine is None:
        _engine = make_engine(get_settings().database_url)
    return _engine


def get_probe_engine() -> Engine:
    """A small engine for the health path whose every wait is bounded."""
    url = get_settings().database_url
    engine = _probe_engines.get(url)
    if engine is None:
        seconds = PROBE_TIMEOUT_SECONDS
        if url.startswith("sqlite"):
            engine = make_engine(url, connect_args={"timeout": seconds}, pool_timeout=seconds)
        else:
            engine = make_engine(
                url,
                connect_args={
                    "connect_timeout": seconds,
                    "options": f"-c statement_timeout={seconds * 1000} "
                    f"-c lock_timeout={seconds * 1000}",
                    # Client-side: unacknowledged data on a dark path fails the socket.
                    "tcp_user_timeout": seconds * 1000,
                    "keepalives": 1,
                    "keepalives_idle": seconds,
                    "keepalives_interval": 1,
                    "keepalives_count": 2,
                },
                # No pool: a connection whose path went dark is never handed to a later probe.
                poolclass=NullPool,
            )
        _probe_engines[url] = engine
    return engine


@contextmanager
def probe_session_scope() -> Iterator[Session]:
    """:func:`session_scope` on :func:`get_probe_engine`, for the health path only."""
    session = Session(bind=get_probe_engine(), expire_on_commit=False)
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """A transactional scope. Commits on success, rolls back on any exception.

    ``BaseException``, not ``Exception``: cooperative cancellation (ADR-006/ADR-007) is a
    ``BaseException`` subclass, and a cancelled step that left its transaction open would
    hold locks until the worker died.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Drop the cached engine and session factory. For tests that change settings."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
    for engine in _probe_engines.values():
        engine.dispose()
    _probe_engines.clear()
