"""The attachment-state publish never waits on the run's own transaction  (found live).

F11 window 2 (2026-10-17): `detach` published the attachment state through a SECOND
connection while the run's session held an uncommitted write to the same `mf_runs` row - so the
publish waited on a lock its own thread held, forever (Postgres has no lock timeout set). The
worker hung with the run `running`. The suite's shared in-memory SQLite connection cannot show a
lock wait at all, so this uses a FILE-backed database with genuinely separate connections.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from db.base import Base
from db.session import make_engine
from models.run import Run
from services.scoring.tables import metadata as score_metadata
from tests.support.fixtures_hotupdate import Harness
from tests.support.jobs import insert_run

RUN = "hot-run"


@pytest.fixture
def engine(tmp_path: Path):  # type: ignore[no-untyped-def]
    engine = make_engine(f"sqlite:///{tmp_path / 'runs.db'}", connect_args={"timeout": 1})
    Base.metadata.create_all(engine)
    score_metadata.create_all(engine)
    with Session(engine) as session:
        insert_run(session, RUN, status="running")
        session.commit()
    return engine


def _acquirer(harness: Harness) -> Any:
    from training.hotupdate.attach import AttachAcquirer

    acquirer = AttachAcquirer(transport=harness.fake.transport())
    acquirer.client = acquirer._open_client()
    response = acquirer.client.call("attach")
    acquirer.client.remember_attachment(str(response["attachment_id"]))
    return acquirer


def test_publishing_while_the_run_holds_its_row_neither_waits_nor_fails(engine, harness) -> None:  # type: ignore[no-untyped-def]
    @contextmanager
    def second_connection():  # what bind_run used to write through
        with Session(engine) as other:
            yield other
            other.commit()

    acquirer = _acquirer(harness)
    with Session(engine) as run_session:
        # The run's own uncommitted write to its row - a heartbeat, a step - holds the lock.
        run_session.execute(
            sa.update(Run.__table__).where(Run.__table__.c.id == RUN).values(last_step=1)
        )
        acquirer.bind_run(run_id=RUN, session_scope=second_connection, session=run_session)
        acquirer.publish_status({"attached": False, "weights_state": "clean"})
        run_session.commit()
    with engine.connect() as connection:
        row = connection.execute(
            sa.select(Run.__table__.c.hotupdate_status, Run.__table__.c.attachment_id).where(
                Run.__table__.c.id == RUN
            )
        ).one()
    assert row.hotupdate_status["attached"] is False
    assert row.attachment_id
