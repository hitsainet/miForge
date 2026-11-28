"""Migration ``0037_wsp_protocol_pass`` (feature 019 P2; S-7, FTASKS 9.1): pre-019-P2 rows read
null in every new column, each CHECK refuses its bad value, and the downgrade returns the schema to
``0036``. Runs on SQLite, and on PostgreSQL when ``MIFORGE_TEST_DATABASE_URL`` names one.
"""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa
from alembic import command

from tests.integration.wsp.wsp_db import NOW, bundle_row, config, engine_at_head, insert, run_row

PASS = "0037_wsp_protocol_pass"
BEFORE = "0036_wsp_window_ceiling"
ENDPOINT_COLUMNS = (
    "protocol_test_result",
    "protocol_test_run_id",
    "protocol_test_at",
    "protocol_tested_base_url",
    "protocol_tested_model",
    "protocol_tested_action_protocol",
)
RUN_COLUMNS = ("protocol_test_endpoint_id",)


def _endpoint(conn: sa.Connection, endpoint_id: str = "ep-old", **extra: Any) -> None:
    insert(
        conn,
        "mf_wsp_endpoints",
        id=endpoint_id,
        kind="openai_compatible",
        base_url="http://vllm.inference.svc:8000/v1",
        model="m",
        locality="local",
        output_cap_field="max_completion_tokens",
        serving="generic",
        created_at=NOW,
        created_by="operator",
        **extra,
    )


def _columns(engine: sa.Engine, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(engine).get_columns(table)}


@pytest.fixture()
def at_0036(tmp_path):
    engine = engine_at_head(tmp_path)
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    command.downgrade(config(url), BEFORE)
    old = sa.create_engine(url)
    assert not set(ENDPOINT_COLUMNS) & _columns(old, "mf_wsp_endpoints")
    assert not set(RUN_COLUMNS) & _columns(old, "mf_wsp_runs")
    with old.begin() as conn:
        _endpoint(conn)
        bundle_row(conn)
        run_row(conn, endpoint_id="ep-old")
    old.dispose()
    yield url


def test_existing_rows_read_null_in_every_new_column(at_0036: str) -> None:
    command.upgrade(config(at_0036), PASS)
    engine = sa.create_engine(at_0036)
    try:
        with engine.connect() as conn:
            endpoint = (
                conn.execute(sa.text(f"SELECT {', '.join(ENDPOINT_COLUMNS)} FROM mf_wsp_endpoints"))
                .mappings()
                .one()
            )
            run = conn.execute(sa.text("SELECT protocol_test_endpoint_id FROM mf_wsp_runs")).one()
        assert dict(endpoint) == dict.fromkeys(ENDPOINT_COLUMNS)
        assert tuple(run) == (None,)
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ("column", "bad", "good"),
    [
        ("protocol_test_result", "maybe", "passed"),
        ("protocol_tested_action_protocol", "free_text", "structured"),
    ],
)
def test_each_check_refuses_its_bad_value(tmp_path, column: str, bad: Any, good: Any) -> None:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        _endpoint(conn, "ep-1", **{column: good})  # the control: a good value is stored
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        _endpoint(conn, "ep-2", **{column: bad})
    engine.dispose()


def test_a_new_row_stores_every_column(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        _endpoint(
            conn,
            "ep-new",
            protocol_test_result="passed",
            protocol_test_run_id="wsp_000000000001",
            protocol_test_at=NOW,
            protocol_tested_base_url="http://vllm.inference.svc:8000/v1",
            protocol_tested_model="m",
            protocol_tested_action_protocol="structured",
        )
        bundle_row(conn)
        run_row(conn, endpoint_id="ep-new", protocol_test_endpoint_id="ep-new")
    with engine.connect() as conn:
        assert conn.execute(
            sa.text(
                "SELECT protocol_test_result, protocol_tested_action_protocol FROM mf_wsp_endpoints"
            )
        ).one() == ("passed", "structured")
        assert conn.execute(sa.text("SELECT protocol_test_endpoint_id FROM mf_wsp_runs")).one() == (
            "ep-new",
        )
    engine.dispose()


def test_the_downgrade_returns_the_schema_to_0036(at_0036: str) -> None:
    command.upgrade(config(at_0036), PASS)
    command.downgrade(config(at_0036), BEFORE)
    engine = sa.create_engine(at_0036)
    assert not set(ENDPOINT_COLUMNS) & _columns(engine, "mf_wsp_endpoints")
    assert not set(RUN_COLUMNS) & _columns(engine, "mf_wsp_runs")
    with engine.connect() as conn:
        assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_endpoints")).scalar_one() == 1
        assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_runs")).scalar_one() == 1
    engine.dispose()
    command.upgrade(config(at_0036), "head")
