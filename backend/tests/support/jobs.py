"""Fixtures for the jobs machinery: the real schema, the real statements, real round trips.

ADR-016: *fixtures whose fields agree by construction are how a suite stays green over a
real bug.* So nothing here declares a table. The schema comes from ``db.base.Base``'s
metadata - the same declarations the Alembic migration is generated from - and the engine
is built by ``db.session.make_engine``, which is **the constructor production uses**. 4.1's
own note makes the same point: *"a test that builds its tables a different way is a fixture
whose fields agree by construction."*

**This file is the fix for a defect, and the defect is worth recording.** The jobs package
originally carried its own ``sa.Table`` declarations as a defensive seam, written while
task 4.2 was in flight, and this module built *those* tables. Every test passed. When 4.2
landed, **six of the seven column names were wrong** - ``run_id`` for ``id``, ``state`` for
``status``, a boolean ``stop_requested`` for a ``cancel_requested_at`` timestamp,
``error_code`` for ``failure_code``, milliseconds for seconds, and a ``heartbeat_writes``
column that does not exist - and the suite had no way to notice, because the fixture and the
code were two views of the same mistake. Importing the real models makes a rename red.

What SQLite does and does not buy, stated because an in-memory database is exactly the kind
of convenience that makes a suite agree with itself. It **does** exercise the real SQL and
the real constraints: the terminal-status predicate in the heartbeat's ``WHERE`` clause, the
``terminal_iff_finished`` check that forces the janitor to set ``finished_at``, the
``CASE`` that keeps the longest gap, and - crucially for the lease - the **partial unique
index** over unreleased rows, which SQLite supports. It does **not** exercise
``timestamptz`` (SQLite drops the offset, which is why
:func:`workers.run_record.as_aware` exists and is applied on every read) or real concurrent
row locking: the contention test asserts that the *index* rejects the second claimant, which
is the half of the mechanism this layer owns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

import models  # noqa: F401 - imported for its side effect on Base.metadata
from db.base import Base
from db.session import make_engine
from models.run import Run
from services.scoring.tables import metadata as score_metadata

RUNS = Run.__table__


def memory_engine() -> Engine:
    """An in-memory SQLite engine carrying the **real** schema.

    ``StaticPool`` with one shared connection so that two ``Session`` objects see one
    database - without it each connection gets its own empty ``:memory:`` and the lease
    contention test would pass by having nothing to contend with.

    The two event hooks are the documented pysqlite workaround: the driver's implicit
    ``BEGIN`` handling breaks ``SAVEPOINT``, and the lease's insert path uses a savepoint so
    that losing the unique-index race does not poison the caller's transaction. Without them
    that path raises instead of refusing, and the failure would look like a lease bug.
    """
    engine = make_engine(
        "sqlite://",
        poolclass=sa.pool.StaticPool,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _disable_implicit_begin(dbapi_connection: Any, _record: Any) -> None:
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def _emit_begin(connection: Any) -> None:
        connection.exec_driver_sql("BEGIN")

    Base.metadata.create_all(engine)

    # F4'S SCORE TABLES, WHICH `Base.metadata` DOES NOT CARRY. They live on their own
    # metadata on purpose - foundation 4.2's ratchet requires it, because *a table only one
    # feature reads belongs in that feature's own migration* - so `create_all` above misses
    # them and this fixture's "the **real** schema" was a schema short of two migrated
    # tables.
    #
    # That mattered the moment a production save started reading them: F5's checkpoint
    # manifest carries this step's `test` and `retention` ScoreRecords (R-02.16, R-02.18),
    # joined from `mf_scores` by `training.tasks.drive_run`. Every loop test driving a save
    # over this engine then hit `no such table: mf_scores` - a failure of the fixture, not
    # of the code: in a deployment the table is migrated and always present, so tolerating
    # its absence in the save path would mean inventing a rollback policy in the middle of
    # a checkpoint write, which is the mistake `services/reindex.py` records making once
    # ("a missing table raises from the database and poisons the transaction").
    score_metadata.create_all(engine)
    return engine


@contextmanager
def session_for(engine: Engine) -> Iterator[Session]:
    """One session, committed on a clean exit - the shape ``db.session.session_scope`` has."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def session_scope_factory(engine: Engine) -> Callable[[], AbstractContextManager[Session]]:
    """A ``session_scope``-shaped callable over ``engine``.

    What the cancellation token is injected with, so its fresh-transaction discipline is
    exercised for real: each check opens and closes its own session.
    """

    def scope() -> AbstractContextManager[Session]:
        return session_for(engine)

    return scope


def insert_run(
    session: Session,
    run_id: str,
    *,
    status: str = "pending",
    # ``change``, matching mf_runs.run_type's CHECK since migration 0014. It said "full"
    # until 2026-10-14 and this ONE default argument feeds every run-creating test in the
    # suite, so the rename produced 123 collection errors from this line alone - while a grep
    # for the assignment spellings (`run_type="full"`) found only four call sites and missed
    # the default that supplies all the rest. Search for the VALUE, not for the way it is
    # written.
    run_type: str = "change",
    heartbeat_at: datetime | None = None,
    longest_heartbeat_gap_seconds: int | None = None,
    cancel_requested_at: datetime | None = None,
    failure_code: str | None = None,
    finished_at: datetime | None = None,
    last_step: int | None = None,
    # ── The three columns a run acquires at SUBMIT, and they are not interchangeable ──
    # `gpu_request` is what was ASKED FOR (usually ``auto``); `gpu_uuid` is the card that
    # was actually leased. `training/run_driver.py` reads the SECOND one and hands it to
    # acquisition, so a test of that wiring has to be able to set them apart - which it
    # could not until 2026-10-16, because neither was expressible here.
    gpu_request: str | None = None,
    gpu_uuid: str | None = None,
    # R-02.53's recorded definition, as the INDEX over the run log. `read_definition` reads
    # the log first and falls back to these, so setting them is how a test gets a recorded
    # definition without writing a log file.
    definition_submitted: dict[str, Any] | None = None,
    definition_resolved: dict[str, Any] | None = None,
    definition_sha256: str | None = None,
) -> None:
    """Insert a run row **through the production table**, not a hand-built dict.

    ``finished_at`` defaults to ``heartbeat_at`` or now for a terminal status, because
    ``mf_runs``' ``terminal_iff_finished`` constraint rejects a terminal row without one -
    so a test asking for a ``completed`` run gets a *legal* completed run rather than an
    insert error that says nothing about what it was testing.
    """
    from workers.run_record import TERMINAL_STATES, utcnow

    if status in TERMINAL_STATES and finished_at is None:
        finished_at = heartbeat_at or utcnow()

    session.execute(
        sa.insert(RUNS).values(
            id=run_id,
            status=status,
            run_type=run_type,
            heartbeat_at=heartbeat_at,
            longest_heartbeat_gap_seconds=longest_heartbeat_gap_seconds,
            cancel_requested_at=cancel_requested_at,
            failure_code=failure_code,
            finished_at=finished_at,
            last_step=last_step,
            gpu_request=gpu_request,
            gpu_uuid=gpu_uuid,
            definition_submitted=definition_submitted,
            definition_resolved=definition_resolved,
            definition_sha256=definition_sha256,
        )
    )


def read_run(session: Session, run_id: str) -> dict[str, Any]:
    """Read a run row back as a mapping, through the same table object."""
    session.expire_all()
    row = session.execute(sa.select(RUNS).where(RUNS.c.id == run_id)).one()
    return dict(row._mapping)


def read_leases(session: Session, gpu_uuid: str) -> list[dict[str, Any]]:
    """Every lease row for a card, released or not - the whole history, since
    ``mf_leases`` is append-and-release rather than a row per card."""
    from models.lease import Lease

    rows = session.execute(
        sa.select(Lease.__table__).where(Lease.__table__.c.resource == gpu_uuid)
    ).all()
    return [dict(r._mapping) for r in rows]
