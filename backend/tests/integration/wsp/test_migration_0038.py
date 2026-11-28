"""Migration ``0038_wsp_pool`` (feature 019 P3; D-22, FTASKS 14.3): pre-P3 rows read null in every
new column, each CHECK refuses its bad value, the downgrade returns the schema to ``0037`` and
refuses, naming them, while two runs are active (the one-active index cannot be rebuilt over
them). Runs on SQLite, and on PostgreSQL when ``MIFORGE_TEST_DATABASE_URL`` names one.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from alembic import command

from tests.integration.wsp.wsp_db import NOW, bundle_row, config, engine_at_head, insert, run_row

POOL = "0038_wsp_pool"
BEFORE = "0037_wsp_protocol_pass"
RUN_COLUMNS = ("schedule_key", "worker_index", "push_heartbeat_at")


def _boot(conn: sa.Connection, boot: str) -> str:
    insert(conn, "mf_wsp_worker_boots", boot_id=boot, first_seen_at=NOW, run_id=None)
    return boot


def _columns(engine: sa.Engine, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(engine).get_columns(table)}


def _indexes(engine: sa.Engine) -> set[str]:
    """From the catalog: SQLite's inspector skips an expression index such as 0037's ``(1)``."""
    query = (
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'mf_wsp_runs'"
        if engine.dialect.name == "sqlite"
        else "SELECT indexname FROM pg_indexes WHERE tablename = 'mf_wsp_runs'"
    )
    with engine.connect() as conn:
        return set(conn.execute(sa.text(query)).scalars())


@pytest.fixture()
def at_0037(tmp_path):
    engine = engine_at_head(tmp_path)
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    command.downgrade(config(url), BEFORE)
    old = sa.create_engine(url)
    assert not set(RUN_COLUMNS) & _columns(old, "mf_wsp_runs")
    assert "mf_wsp_workers" not in sa.inspect(old).get_table_names()
    assert "uq_mf_wsp_runs_one_active" in _indexes(old)
    with old.begin() as conn:
        bundle_row(conn)
        run_row(conn, status="running", boot_id=_boot(conn, "b" * 32))
        run_row(conn, "wsp_000000000002")
    old.dispose()
    yield url
    command.upgrade(config(url), "head")


def test_existing_rows_read_null_in_every_new_column(at_0037: str) -> None:
    command.upgrade(config(at_0037), POOL)
    engine = sa.create_engine(at_0037)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                sa.text(f"SELECT {', '.join(RUN_COLUMNS)} FROM mf_wsp_runs ORDER BY run_id")
            ).all()
            workers = conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_workers")).scalar_one()
        assert [tuple(r) for r in rows] == [(None, None, None), (None, None, None)]
        assert workers == 0
        indexes = _indexes(engine)
        assert {"uq_mf_wsp_runs_one_active_key", "uq_mf_wsp_runs_one_per_worker"} <= indexes
        assert "uq_mf_wsp_runs_one_active" not in indexes
    finally:
        engine.dispose()


def test_a_negative_worker_index_is_refused(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        bundle_row(conn)
        run_row(conn, worker_index=0)  # the control: index 0 is stored
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        run_row(conn, "wsp_000000000002", worker_index=-1)
    engine.dispose()


@pytest.mark.parametrize(("column", "bad"), [("state", "on_fire"), ("worker_index", -1)])
def test_each_worker_row_check_refuses_its_bad_value(tmp_path, column: str, bad: object) -> None:
    engine = engine_at_head(tmp_path)
    good = {"worker_index": 0, "address": "http://w-0:9000", "state": "idle", "read_at": NOW}
    with engine.begin() as conn:
        insert(conn, "mf_wsp_workers", **good)  # the control
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        insert(conn, "mf_wsp_workers", **{**good, "worker_index": 1, column: bad})
    engine.dispose()


def test_the_downgrade_returns_the_schema_to_0037(at_0037: str) -> None:
    command.upgrade(config(at_0037), POOL)
    command.downgrade(config(at_0037), BEFORE)
    engine = sa.create_engine(at_0037)
    try:
        assert not set(RUN_COLUMNS) & _columns(engine, "mf_wsp_runs")
        assert "mf_wsp_workers" not in sa.inspect(engine).get_table_names()
        assert "uq_mf_wsp_runs_one_active" in _indexes(engine)
        with engine.connect() as conn:
            assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_runs")).scalar_one() == 2
    finally:
        engine.dispose()


def test_the_downgrade_refuses_with_two_active_runs_naming_both(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    url = engine.url.render_as_string(hide_password=False)
    with engine.begin() as conn:
        bundle_row(conn)
        run_row(
            conn,
            "wsp_00000000000a",
            status="running",
            boot_id=_boot(conn, "a" * 32),
            worker_index=0,
            schedule_key="endpoint:a",
        )
        run_row(
            conn,
            "wsp_00000000000b",
            status="pushing",
            boot_id=_boot(conn, "b" * 32),
            worker_index=1,
            schedule_key="endpoint:b",
        )
    engine.dispose()
    with pytest.raises(RuntimeError) as refused:
        command.downgrade(config(url), BEFORE)
    assert "wsp_00000000000a" in str(refused.value) and "wsp_00000000000b" in str(refused.value)
    check = sa.create_engine(url)
    try:
        assert "schedule_key" in _columns(check, "mf_wsp_runs")  # nothing was dropped
    finally:
        check.dispose()
