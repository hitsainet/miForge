"""Migration ``0040_wsp_submitted_check`` (018 FU-2, 2026-11-25): a pre-existing run gets no
check (nothing is backfilled, so its document and ``run_sha256`` are unchanged), a new run stores
one, the downgrade refuses while any run holds one, and the round trip works without. Runs on
SQLite, and on PostgreSQL when ``MIFORGE_TEST_DATABASE_URL`` names one (the serial PostgreSQL job).
"""

from __future__ import annotations

import json

import pytest
import sqlalchemy as sa
from alembic import command

from tests.integration.wsp.wsp_db import bundle_row, config, engine_at_head, run_row

CHECK = "0040_wsp_submitted_check"
BEFORE = "0039_wsp_hosted"
VALUE = {
    "stated": False,
    "stated_split": None,
    "compared": False,
    "differs": None,
    "checks": [],
    "measured": False,
    "split": None,
}


def _columns(engine: sa.Engine, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(engine).get_columns(table)}


@pytest.fixture()
def at_0039(tmp_path):
    """A database at ``0039`` holding one run, taken in before the check existed."""
    engine = engine_at_head(tmp_path)
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    command.downgrade(config(url), BEFORE)
    old = sa.create_engine(url)
    assert "submitted_check" not in _columns(old, "mf_wsp_runs")
    with old.begin() as conn:
        bundle_row(conn)
        run_row(conn)
    old.dispose()
    yield url


def test_an_existing_run_has_no_check(at_0039: str) -> None:
    command.upgrade(config(at_0039), CHECK)
    engine = sa.create_engine(at_0039)
    try:
        with engine.connect() as conn:
            value = conn.execute(sa.text("SELECT submitted_check FROM mf_wsp_runs")).scalar_one()
        assert value is None
    finally:
        engine.dispose()


def test_a_new_run_stores_its_check(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        bundle_row(conn)
        run_row(conn, submitted_check=json.dumps(VALUE))
    with engine.connect() as conn:
        stored = conn.execute(sa.text("SELECT submitted_check FROM mf_wsp_runs")).scalar_one()
    assert (json.loads(stored) if isinstance(stored, str) else stored) == VALUE
    engine.dispose()


def test_the_downgrade_refuses_over_a_stored_check(at_0039: str) -> None:
    command.upgrade(config(at_0039), CHECK)
    engine = sa.create_engine(at_0039)
    with engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE mf_wsp_runs SET submitted_check = :v"), {"v": json.dumps(VALUE)}
        )
    engine.dispose()
    with pytest.raises(RuntimeError, match=r"1 run\(s\) hold a submitted check"):
        command.downgrade(config(at_0039), BEFORE)
    engine = sa.create_engine(at_0039)
    assert "submitted_check" in _columns(engine, "mf_wsp_runs")
    with engine.begin() as conn:
        conn.execute(sa.text("UPDATE mf_wsp_runs SET submitted_check = NULL"))
    engine.dispose()
    command.upgrade(config(at_0039), "head")


def test_the_round_trip(at_0039: str) -> None:
    command.upgrade(config(at_0039), CHECK)
    command.downgrade(config(at_0039), BEFORE)
    engine = sa.create_engine(at_0039)
    assert "submitted_check" not in _columns(engine, "mf_wsp_runs")
    with engine.connect() as conn:  # the pre-existing run survives both directions
        assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_runs")).scalar_one() == 1
    engine.dispose()
    command.upgrade(config(at_0039), "head")
