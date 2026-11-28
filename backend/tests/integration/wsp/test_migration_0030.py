"""Migration ``0030_wsp_live_progress`` (live progress, 2026-11-20): a pre-existing run and usage
row read null in every new column, every CHECK refuses its bad value, ``mf_wsp_usage`` stays
append-only, and the round trip works. Runs on SQLite, and on PostgreSQL when
``MIFORGE_TEST_DATABASE_URL`` names one (the serial PostgreSQL job).
"""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa
from alembic import command

from tests.integration.wsp.wsp_db import (
    NOW,
    bundle_row,
    config,
    engine_at_head,
    insert,
    run_row,
    usage_row,
)

LIVE = "0030_wsp_live_progress"
BEFORE = "0029_wsp_engine"
RUN_COLUMNS = ("worker_state", "worker_elapsed_s")
USAGE_COLUMNS = ("outcome", "http_status", "latency_ms", "finish_reason", "refusal_code")


def _columns(engine: sa.Engine, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(engine).get_columns(table)}


@pytest.fixture()
def at_0029(tmp_path):
    """A database at ``0029`` holding one run with one metered call, written before 0030."""
    engine = engine_at_head(tmp_path)
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    command.downgrade(config(url), BEFORE)
    old = sa.create_engine(url)
    assert not set(USAGE_COLUMNS) & _columns(old, "mf_wsp_usage")
    with old.begin() as conn:
        bundle_row(conn)
        run_row(conn)
        usage_row(conn, "wsp_000000000001")
    old.dispose()
    yield url


def test_existing_rows_read_null(at_0029: str) -> None:
    command.upgrade(config(at_0029), LIVE)
    engine = sa.create_engine(at_0029)
    try:
        with engine.connect() as conn:
            run = conn.execute(sa.text(f"SELECT {', '.join(RUN_COLUMNS)} FROM mf_wsp_runs"))
            usage = conn.execute(sa.text(f"SELECT {', '.join(USAGE_COLUMNS)} FROM mf_wsp_usage"))
            assert dict(run.mappings().one()) == dict.fromkeys(RUN_COLUMNS)
            assert dict(usage.mappings().one()) == dict.fromkeys(USAGE_COLUMNS)
    finally:
        engine.dispose()


def _usage(conn: sa.Connection, seq: int, **extra: Any) -> None:
    insert(
        conn,
        "mf_wsp_usage",
        run_id="wsp_000000000001",
        seq=seq,
        endpoint_id="ep",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0,
        at=NOW,
        **extra,
    )


@pytest.mark.parametrize(
    ("table", "column", "bad", "good"),
    [
        ("mf_wsp_runs", "worker_state", "free_text", "running"),
        ("mf_wsp_runs", "worker_elapsed_s", -1, 12),
        ("mf_wsp_usage", "outcome", "free_text", "served"),
        ("mf_wsp_usage", "http_status", 42, 200),
        ("mf_wsp_usage", "latency_ms", -1, 830),
    ],
)
def test_each_check_refuses_its_bad_value(
    tmp_path, table: str, column: str, bad: Any, good: Any
) -> None:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        bundle_row(conn)
        if table == "mf_wsp_runs":
            run_row(conn, "wsp_000000000001", **{column: good})  # the control
        else:
            run_row(conn, "wsp_000000000001")
            _usage(conn, 0, **{column: good})
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        if table == "mf_wsp_runs":
            run_row(conn, "wsp_000000000002", **{column: bad})
        else:
            _usage(conn, 1, **{column: bad})
    engine.dispose()


def test_the_usage_table_stays_append_only(at_0029: str) -> None:
    """``ADD COLUMN`` rebuilds no table, so the UPDATE and DELETE triggers survive it."""
    command.upgrade(config(at_0029), LIVE)
    engine = sa.create_engine(at_0029)
    try:
        for statement in (
            "UPDATE mf_wsp_usage SET outcome = 'served'",
            "DELETE FROM mf_wsp_usage",
        ):
            with pytest.raises(sa.exc.DBAPIError), engine.begin() as conn:
                conn.execute(sa.text(statement))
    finally:
        engine.dispose()


def test_the_round_trip(at_0029: str) -> None:
    command.upgrade(config(at_0029), LIVE)
    command.downgrade(config(at_0029), BEFORE)
    engine = sa.create_engine(at_0029)
    assert not set(RUN_COLUMNS) & _columns(engine, "mf_wsp_runs")
    assert not set(USAGE_COLUMNS) & _columns(engine, "mf_wsp_usage")
    with engine.connect() as conn:  # the pre-existing rows survive both directions
        assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_usage")).scalar_one() == 1
    with pytest.raises(sa.exc.DBAPIError), engine.begin() as conn:  # still append-only
        conn.execute(sa.text("DELETE FROM mf_wsp_usage"))
    engine.dispose()
    command.upgrade(config(at_0029), "head")
    engine = sa.create_engine(at_0029)
    assert set(USAGE_COLUMNS) <= _columns(engine, "mf_wsp_usage")
    assert set(RUN_COLUMNS) <= _columns(engine, "mf_wsp_runs")
    engine.dispose()
