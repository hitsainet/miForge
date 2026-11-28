"""Migration ``0036_wsp_window_ceiling`` (feature 019 P1; W-10, FTASKS 3.3): pre-019 rows read null
in every new column, every CHECK refuses its bad value, and the downgrade returns the schema to
``0035``. Runs on SQLite, and on PostgreSQL when ``MIFORGE_TEST_DATABASE_URL`` names one (the
serial PostgreSQL job).
"""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa
from alembic import command

from tests.integration.wsp.wsp_db import NOW, bundle_row, config, engine_at_head, insert, run_row

WINDOW = "0036_wsp_window_ceiling"
BEFORE = "0035_wsp_outcome_reason"
ENDPOINT_COLUMNS = (
    "output_ceiling_tokens",
    "registration_window_tokens",
    "registration_window_read_at",
)
RUN_COLUMNS = ("window_tokens", "window_source", "output_ceiling_tokens", "output_ceiling_lowered")


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
def at_0035(tmp_path):
    """A database at ``0035`` holding one endpoint and one run, written before 019."""
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
        run_row(conn, served_context_tokens=32_768)
    old.dispose()
    yield url


def test_pre_019_rows_read_null_in_every_new_column(at_0035: str) -> None:
    command.upgrade(config(at_0035), WINDOW)
    engine = sa.create_engine(at_0035)
    try:
        with engine.connect() as conn:
            endpoint = (
                conn.execute(sa.text(f"SELECT {', '.join(ENDPOINT_COLUMNS)} FROM mf_wsp_endpoints"))
                .mappings()
                .one()
            )
            run = (
                conn.execute(
                    sa.text(
                        f"SELECT {', '.join(RUN_COLUMNS)}, served_context_tokens FROM mf_wsp_runs"
                    )
                )
                .mappings()
                .one()
            )
        assert dict(endpoint) == dict.fromkeys(ENDPOINT_COLUMNS)
        assert dict(run) == {**dict.fromkeys(RUN_COLUMNS), "served_context_tokens": 32_768}
    finally:
        engine.dispose()


def test_a_new_row_stores_every_column(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        _endpoint(
            conn,
            "ep-new",
            output_ceiling_tokens=4_096,
            registration_window_tokens=32_768,
            registration_window_read_at=NOW,
        )
        bundle_row(conn)
        run_row(
            conn,
            window_tokens=32_768,
            window_source="reported",
            output_ceiling_tokens=4_096,
            output_ceiling_lowered=False,
        )
    with engine.connect() as conn:
        endpoint = conn.execute(
            sa.text(
                "SELECT output_ceiling_tokens, registration_window_tokens FROM mf_wsp_endpoints"
            )
        ).one()
        run = conn.execute(
            sa.text("SELECT window_tokens, window_source, output_ceiling_tokens FROM mf_wsp_runs")
        ).one()
    assert tuple(endpoint) == (4_096, 32_768)
    assert tuple(run) == (32_768, "reported", 4_096)
    engine.dispose()


@pytest.mark.parametrize(
    ("table", "column", "bad", "good"),
    [
        ("mf_wsp_endpoints", "output_ceiling_tokens", 255, 256),
        ("mf_wsp_endpoints", "registration_window_tokens", 0, 8_192),
        ("mf_wsp_runs", "window_tokens", 0, 8_192),
        ("mf_wsp_runs", "window_source", "guessed", "assumed"),
        ("mf_wsp_runs", "output_ceiling_tokens", 0, 32_000),
    ],
)
def test_each_check_refuses_its_bad_value(
    tmp_path, table: str, column: str, bad: Any, good: Any
) -> None:
    engine = engine_at_head(tmp_path)
    endpoint = table == "mf_wsp_endpoints"

    def row(conn: sa.Connection, value: Any, n: int) -> None:
        if endpoint:
            _endpoint(conn, f"ep-{n}", **{column: value})
        else:
            run_row(conn, f"wsp_00000000000{n}", **{column: value})

    with engine.begin() as conn:
        if not endpoint:
            bundle_row(conn)
        row(conn, good, 1)  # the control: a good value is stored
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        row(conn, bad, 2)
    engine.dispose()


def test_the_downgrade_returns_the_schema_to_0035(at_0035: str) -> None:
    command.upgrade(config(at_0035), WINDOW)
    command.downgrade(config(at_0035), BEFORE)
    engine = sa.create_engine(at_0035)
    assert not set(ENDPOINT_COLUMNS) & _columns(engine, "mf_wsp_endpoints")
    assert not set(RUN_COLUMNS) & _columns(engine, "mf_wsp_runs")
    assert "served_context_tokens" in _columns(engine, "mf_wsp_runs")
    with engine.connect() as conn:  # the pre-existing rows survive both directions
        assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_endpoints")).scalar_one() == 1
        assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_runs")).scalar_one() == 1
    engine.dispose()
    command.upgrade(config(at_0035), "head")
