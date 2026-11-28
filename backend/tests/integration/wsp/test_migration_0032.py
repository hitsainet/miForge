"""Migration ``0032_wsp_thinking`` (2026-11-21): a pre-existing endpoint gets ``thinking: off``
and no allowance, a pre-existing run a null ``reasoning_turns``, every CHECK refuses its bad value,
and the round trip works. Runs on SQLite, and on PostgreSQL when ``MIFORGE_TEST_DATABASE_URL``
names one (the serial PostgreSQL job).
"""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa
from alembic import command

from tests.integration.wsp.wsp_db import NOW, bundle_row, config, engine_at_head, insert, run_row

THINKING = "0032_wsp_thinking"
BEFORE = "0031_wsp_endpoint_register"
ENDPOINT_COLUMNS = ("thinking", "thinking_tokens")


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
def at_0031(tmp_path):
    """A database at ``0031`` holding one endpoint and one run, written before thinking existed."""
    engine = engine_at_head(tmp_path)
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    command.downgrade(config(url), BEFORE)
    old = sa.create_engine(url)
    assert "thinking" not in _columns(old, "mf_wsp_endpoints")
    with old.begin() as conn:
        _endpoint(conn)
        bundle_row(conn)
        run_row(conn)
    old.dispose()
    yield url


def test_existing_rows_take_the_defaults(at_0031: str) -> None:
    command.upgrade(config(at_0031), THINKING)
    engine = sa.create_engine(at_0031)
    try:
        with engine.connect() as conn:
            endpoint = (
                conn.execute(
                    sa.text(
                        "SELECT thinking, thinking_tokens FROM mf_wsp_endpoints "
                        "WHERE id = 'ep-old'"
                    )
                )
                .mappings()
                .one()
            )
            turns = (
                conn.execute(sa.text("SELECT reasoning_turns, repeated_turns FROM mf_wsp_runs"))
                .mappings()
                .one()
            )
        assert dict(endpoint) == {"thinking": "off", "thinking_tokens": None}
        assert dict(turns) == {"reasoning_turns": None, "repeated_turns": None}
    finally:
        engine.dispose()


def test_a_new_endpoint_row_defaults_to_off(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        _endpoint(conn, "ep-new")
    with engine.connect() as conn:
        value = conn.execute(
            sa.text("SELECT thinking FROM mf_wsp_endpoints WHERE id = 'ep-new'")
        ).scalar_one()
    assert value == "off"
    engine.dispose()


@pytest.mark.parametrize(
    ("table", "column", "bad", "good"),
    [
        ("mf_wsp_endpoints", "thinking", "maybe", "on"),
        ("mf_wsp_endpoints", "thinking", "xhigh", "default"),
        ("mf_wsp_endpoints", "thinking_tokens", 0, 4096),
        ("mf_wsp_runs", "reasoning_turns", -1, 2),
        ("mf_wsp_runs", "repeated_turns", -1, 3),
        ("mf_wsp_runs", "served_context_tokens", 0, 32768),
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


def test_the_round_trip(at_0031: str) -> None:
    command.upgrade(config(at_0031), THINKING)
    command.downgrade(config(at_0031), BEFORE)
    engine = sa.create_engine(at_0031)
    assert not set(ENDPOINT_COLUMNS) & _columns(engine, "mf_wsp_endpoints")
    assert not {"reasoning_turns", "repeated_turns", "served_context_tokens"} & _columns(
        engine, "mf_wsp_runs"
    )
    with engine.connect() as conn:  # the pre-existing rows survive both directions
        assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_endpoints")).scalar_one() == 1
        assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_runs")).scalar_one() == 1
    engine.dispose()
    command.upgrade(config(at_0031), "head")
