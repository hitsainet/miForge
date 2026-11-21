"""Migration ``0029_wsp_engine`` (018 FTASKS 8.1): a pre-existing endpoint gets ``tool_calls``
and no window, a pre-existing run gets null counters, every CHECK refuses its bad value, and the
round trip works. Runs on SQLite, and on PostgreSQL when ``MIFORGE_TEST_DATABASE_URL`` names one
(the serial PostgreSQL job).
"""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa
from alembic import command

from tests.integration.wsp.wsp_db import NOW, bundle_row, config, engine_at_head, insert, run_row

ENGINE = "0029_wsp_engine"
BEFORE = "0028_wsp_provider"
RUN_COLUMNS = ("action_protocol", "handoffs", "refused_turns", "kernel_restarts")


def _endpoint(conn: sa.Connection, endpoint_id: str = "ep-old", **extra: Any) -> None:
    insert(
        conn,
        "mf_wsp_endpoints",
        id=endpoint_id,
        kind="openai_compatible",
        base_url="http://miserver-backend.miserver.svc:8000/v1",
        model="m",
        locality="local",
        output_cap_field="max_tokens",
        serving="miserver",
        created_at=NOW,
        created_by="operator",
        **extra,
    )


def _columns(engine: sa.Engine, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(engine).get_columns(table)}


@pytest.fixture()
def at_0028(tmp_path):
    """A database at ``0028`` holding one endpoint and one run, written before 018 existed."""
    engine = engine_at_head(tmp_path)
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    command.downgrade(config(url), BEFORE)
    old = sa.create_engine(url)
    assert "action_protocol" not in _columns(old, "mf_wsp_endpoints")
    with old.begin() as conn:
        _endpoint(conn)
        bundle_row(conn)
        run_row(conn)
    old.dispose()
    yield url


def test_existing_rows_take_the_defaults(at_0028: str) -> None:
    command.upgrade(config(at_0028), ENGINE)
    engine = sa.create_engine(at_0028)
    try:
        with engine.connect() as conn:
            endpoint = (
                conn.execute(
                    sa.text(
                        "SELECT action_protocol, context_window_tokens FROM mf_wsp_endpoints "
                        "WHERE id = 'ep-old'"
                    )
                )
                .mappings()
                .one()
            )
            run = (
                conn.execute(sa.text(f"SELECT {', '.join(RUN_COLUMNS)} FROM mf_wsp_runs"))
                .mappings()
                .one()
            )
        assert dict(endpoint) == {"action_protocol": "tool_calls", "context_window_tokens": None}
        assert dict(run) == dict.fromkeys(RUN_COLUMNS)
    finally:
        engine.dispose()


def test_a_new_endpoint_row_defaults_to_tool_calls(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        _endpoint(conn, "ep-new")
    with engine.connect() as conn:
        value = conn.execute(
            sa.text("SELECT action_protocol FROM mf_wsp_endpoints WHERE id = 'ep-new'")
        ).scalar_one()
    assert value == "tool_calls"
    engine.dispose()


@pytest.mark.parametrize(
    ("table", "column", "bad", "good"),
    [
        ("mf_wsp_endpoints", "action_protocol", "free_text", "structured"),
        ("mf_wsp_endpoints", "context_window_tokens", 0, 32768),
        ("mf_wsp_runs", "action_protocol", "free_text", "structured"),
        ("mf_wsp_runs", "handoffs", -1, 0),
        ("mf_wsp_runs", "refused_turns", -1, 3),
        ("mf_wsp_runs", "kernel_restarts", -1, 1),
    ],
)
def test_each_check_refuses_its_bad_value(
    tmp_path, table: str, column: str, bad: Any, good: Any
) -> None:
    engine = engine_at_head(tmp_path)
    write = _endpoint if table == "mf_wsp_endpoints" else None

    def row(conn: sa.Connection, value: Any, n: int) -> None:
        if write is not None:
            write(conn, f"ep-{n}", **{column: value})
        else:
            run_row(conn, f"wsp_00000000000{n}", **{column: value})

    with engine.begin() as conn:
        if write is None:
            bundle_row(conn)
        row(conn, good, 1)  # the control: a good value is stored
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        row(conn, bad, 2)
    engine.dispose()


def test_the_round_trip(at_0028: str) -> None:
    command.upgrade(config(at_0028), ENGINE)
    command.downgrade(config(at_0028), BEFORE)
    engine = sa.create_engine(at_0028)
    assert "action_protocol" not in _columns(engine, "mf_wsp_endpoints")
    assert not set(RUN_COLUMNS) & _columns(engine, "mf_wsp_runs")
    with engine.connect() as conn:  # the pre-existing rows survive both directions
        assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_endpoints")).scalar_one() == 1
        assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_runs")).scalar_one() == 1
    engine.dispose()
    command.upgrade(config(at_0028), "head")
    engine = sa.create_engine(at_0028)
    assert {"action_protocol", "context_window_tokens"} <= _columns(engine, "mf_wsp_endpoints")
    assert set(RUN_COLUMNS) <= _columns(engine, "mf_wsp_runs")
    engine.dispose()
