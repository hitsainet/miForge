"""019 P3 (D-2; FR-019.8, F19-D17, D24 reading 2): the database itself holds the scheduling rules,
whatever a driver pass does. A second ACTIVE run (``pushing`` or ``running``) on one key, and a
second run HOLDING one worker (``pushing``, ``running`` or ``collecting``), are refused, inserted
directly. Runs on SQLite, and on PostgreSQL when ``MIFORGE_TEST_DATABASE_URL`` names one.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from tests.integration.wsp.wsp_db import NOW, bundle_row, engine_at_head, insert, run_row


@pytest.fixture()
def engine(tmp_path):
    built = engine_at_head(tmp_path)
    with built.begin() as conn:
        bundle_row(conn)
    yield built
    built.dispose()


def _run(engine: sa.Engine, n: int, **values) -> None:
    with engine.begin() as conn:
        insert(conn, "mf_wsp_worker_boots", boot_id=f"{n:032x}", first_seen_at=NOW, run_id=None)
        run_row(conn, f"wsp_{n:012x}", boot_id=f"{n:032x}", **values)


@pytest.mark.parametrize("second", ["pushing", "running"])
def test_a_second_active_run_on_one_key_is_refused(engine: sa.Engine, second: str) -> None:
    _run(engine, 1, status="running", schedule_key="miserver:http://m:8000", worker_index=0)
    with pytest.raises(sa.exc.IntegrityError):
        _run(engine, 2, status=second, schedule_key="miserver:http://m:8000", worker_index=1)


def test_another_key_and_a_finished_run_on_the_key_are_admitted(engine: sa.Engine) -> None:
    """The controls: the index is partial over ACTIVE rows and keyed, not a global one-active."""
    _run(engine, 1, status="running", schedule_key="endpoint:a", worker_index=0)
    _run(engine, 2, status="running", schedule_key="endpoint:b", worker_index=1)
    _run(
        engine,
        3,
        status="cancelled",
        stop_reason="cancelled",
        schedule_key="endpoint:a",
        worker_index=0,
    )
    _run(engine, 4, status="queued", schedule_key=None)
    # A run that is collecting has left the key (its server is free), but still holds its worker.
    _run(engine, 5, status="collecting", schedule_key="endpoint:c", worker_index=2)
    _run(engine, 6, status="pushing", schedule_key="endpoint:c", worker_index=3)


@pytest.mark.parametrize("second", ["pushing", "running", "collecting"])
def test_a_second_run_holding_one_worker_is_refused(engine: sa.Engine, second: str) -> None:
    _run(engine, 1, status="collecting", schedule_key="endpoint:a", worker_index=0)
    with pytest.raises(sa.exc.IntegrityError):
        _run(engine, 2, status=second, schedule_key="endpoint:b", worker_index=0)


def test_keyless_runs_are_bounded_only_by_their_worker(engine: sa.Engine) -> None:
    """A replay or a rerun has no key (F19-D7): two may run, on two workers, never on one."""
    _run(engine, 1, status="running", schedule_key=None, worker_index=0)
    _run(engine, 2, status="running", schedule_key=None, worker_index=1)
    with pytest.raises(sa.exc.IntegrityError):
        _run(engine, 3, status="running", schedule_key=None, worker_index=1)
