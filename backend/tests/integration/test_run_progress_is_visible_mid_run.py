"""A running run's progress is visible to OTHER connections while the run is still going.

Found live 2026-10-17 (F11 window 2, run r3): nothing committed during a run. The heartbeat,
the step index rows and ``last_step`` sat in the loop's one transaction until the run ended, so

* the janitor (another connection) read a stale ``heartbeat_at`` on a healthy run,
* ``GET /runs/{id}`` showed ``last_step`` 0 at step 650,
* a failure rolled every step row back, and
* the run's row stayed locked for the whole run.

So this is two engines on one WAL-mode SQLite file - the loop's connection and "the API's" -
and the loop's session stays OPEN across the assertions, as the production loop's does. An
in-memory StaticPool would share one connection and see uncommitted rows, passing over the
defect (that is the fixture this test exists to avoid).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker

from tests.support.jobs import insert_run, read_run, session_for
from training.run_config import RunConfig
from training.step import StepOutcome
from training.step_record import StepWriter
from workers.heartbeat import Heartbeat

RUN = "run-visible-mid-run"
CFG = RunConfig(learning_rate=2e-5, warmup_steps=5, batch_size=4, gradient_accumulation=2)


def _outcome() -> StepOutcome:
    return StepOutcome(
        loss=0.5, objective="sft", grad_norm=1.0, wall_time_ms=10.0, batch_sha256="d" * 64
    )


@pytest.fixture
def engines(tmp_path: Path) -> Iterator[tuple[Any, Any]]:
    import models  # noqa: F401 - registers the tables on Base.metadata
    from db.base import Base
    from db.session import make_engine

    url = f"sqlite:///{tmp_path / 'runs.db'}"

    def build() -> Any:
        built = make_engine(url, connect_args={"check_same_thread": False, "timeout": 0.5})

        @event.listens_for(built, "connect")
        def _wal(dbapi_connection: Any, _record: Any) -> None:
            dbapi_connection.isolation_level = None
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

        @event.listens_for(built, "begin")
        def _begin(connection: Any) -> None:
            connection.exec_driver_sql("BEGIN")

        return built

    other = build()
    Base.metadata.create_all(other)
    with session_for(other) as session:
        insert_run(session, RUN, status="running")
    loop_engine = build()
    yield loop_engine, other
    loop_engine.dispose()
    other.dispose()


def test_the_beat_and_the_step_rows_are_visible_while_the_run_holds_its_session(
    engines: tuple[Any, Any], tmp_path: Path
) -> None:
    loop_engine, other = engines
    writer = StepWriter(run_id=RUN, log_root=tmp_path / "runs", heartbeat=Heartbeat(RUN))
    loop_session = sessionmaker(bind=loop_engine, expire_on_commit=False)()
    try:
        writer.write(session=loop_session, step=1, outcome=_outcome(), cfg=CFG)
        # The loop's session is still open, exactly as during a production run.
        with session_for(other) as api:
            row = read_run(api, RUN)
            steps = api.execute(sa.text("SELECT step FROM mf_steps WHERE run_id = :r"), {"r": RUN})
            visible_steps = [r[0] for r in steps]
        assert row["heartbeat_at"] is not None, "the beat is not visible to the janitor mid-run"
        assert row["last_step"] == 1, "the run row's last_step (the list's column) lags"
        assert visible_steps == [1], f"step rows invisible mid-run: {visible_steps}"
        # And the run's row is not held locked: another connection can write it.
        with session_for(other) as api:
            api.execute(sa.text("UPDATE mf_runs SET status = status WHERE id = :r"), {"r": RUN})
    finally:
        loop_session.rollback()
        loop_session.close()
