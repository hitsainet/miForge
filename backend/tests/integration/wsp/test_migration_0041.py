"""Migration ``0041_wsp_worker_disk`` (2026-11-26, the 019 P4 disk gap): a worker row from before
it reads no disk figures (null, never zero), a row written after stores both, and the round trip
keeps the row. Runs on SQLite, and on PostgreSQL when ``MIFORGE_TEST_DATABASE_URL`` names one.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from alembic import command

from tests.integration.wsp.wsp_db import NOW, config, engine_at_head, insert

DISK = "0041_wsp_worker_disk"
BEFORE = "0040_wsp_submitted_check"
COLUMNS = {"work_free_bytes", "work_total_bytes"}


def _columns(engine: sa.Engine) -> set[str]:
    return {c["name"] for c in sa.inspect(engine).get_columns("mf_wsp_workers")}


def _worker(conn: sa.Connection) -> None:
    insert(
        conn,
        "mf_wsp_workers",
        worker_index=0,
        address="miforge-wsp-worker-0.miforge-wsp-workers:9000",
        state="idle",
        boot_id="b" * 32,
        read_at=NOW,
        answered_at=NOW,
    )


@pytest.fixture()
def at_0040(tmp_path):
    """A database at ``0040`` holding one worker row, read before the disk figures existed."""
    engine = engine_at_head(tmp_path)
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    command.downgrade(config(url), BEFORE)
    old = sa.create_engine(url)
    assert not COLUMNS & _columns(old)
    with old.begin() as conn:
        conn.execute(sa.text("DELETE FROM mf_wsp_workers"))
        _worker(conn)
    old.dispose()
    yield url
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        conn.execute(sa.text("DELETE FROM mf_wsp_workers"))
    engine.dispose()


def test_an_existing_worker_row_reads_no_figures(at_0040: str) -> None:
    command.upgrade(config(at_0040), DISK)
    engine = sa.create_engine(at_0040)
    try:
        assert COLUMNS <= _columns(engine)
        with engine.connect() as conn:
            row = conn.execute(
                sa.text("SELECT work_free_bytes, work_total_bytes FROM mf_wsp_workers")
            ).one()
        assert tuple(row) == (None, None)
    finally:
        engine.dispose()


def test_a_row_stores_both_figures_beyond_32_bits(at_0040: str) -> None:
    command.upgrade(config(at_0040), DISK)
    engine = sa.create_engine(at_0040)
    try:
        with engine.begin() as conn:
            conn.execute(
                sa.text("UPDATE mf_wsp_workers SET work_free_bytes = :f, work_total_bytes = :t"),
                {"f": 157_212_762_112, "t": 844_362_645_504},
            )
        with engine.connect() as conn:
            row = conn.execute(
                sa.text("SELECT work_free_bytes, work_total_bytes FROM mf_wsp_workers")
            ).one()
        assert tuple(row) == (157_212_762_112, 844_362_645_504)
    finally:
        engine.dispose()


def test_the_round_trip(at_0040: str) -> None:
    command.upgrade(config(at_0040), DISK)
    command.downgrade(config(at_0040), BEFORE)
    engine = sa.create_engine(at_0040)
    try:
        assert not COLUMNS & _columns(engine)
        with engine.connect() as conn:  # the worker row survives both directions
            assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_workers")).scalar_one() == 1
    finally:
        engine.dispose()
    command.upgrade(config(at_0040), "head")
