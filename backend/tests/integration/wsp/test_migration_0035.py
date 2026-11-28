"""Migration ``0035_wsp_outcome_reason`` (2026-11-23): a pre-existing run gets a null reason, a
new run stores one, and the round trip works. Runs on SQLite, and on PostgreSQL when
``MIFORGE_TEST_DATABASE_URL`` names one (the serial PostgreSQL job).
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from alembic import command

from tests.integration.wsp.wsp_db import bundle_row, config, engine_at_head, run_row

REASON = "0035_wsp_outcome_reason"
BEFORE = "0034_wsp_paging"


def _columns(engine: sa.Engine, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(engine).get_columns(table)}


@pytest.fixture()
def at_0034(tmp_path):
    """A database at ``0034`` holding one run, finished before the reason was stored."""
    engine = engine_at_head(tmp_path)
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    command.downgrade(config(url), BEFORE)
    old = sa.create_engine(url)
    assert "outcome_reason" not in _columns(old, "mf_wsp_runs")
    with old.begin() as conn:
        bundle_row(conn)
        run_row(conn)
    old.dispose()
    yield url


def test_an_existing_run_has_no_reason(at_0034: str) -> None:
    command.upgrade(config(at_0034), REASON)
    engine = sa.create_engine(at_0034)
    try:
        with engine.connect() as conn:
            value = conn.execute(sa.text("SELECT outcome_reason FROM mf_wsp_runs")).scalar_one()
        assert value is None
    finally:
        engine.dispose()


def test_a_new_run_stores_its_reason(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    text = "practice-validation AUROC 0.990293 is below the parent's 0.990332"
    with engine.begin() as conn:
        bundle_row(conn)
        run_row(conn, outcome_reason=text)
    with engine.connect() as conn:
        assert conn.execute(sa.text("SELECT outcome_reason FROM mf_wsp_runs")).scalar_one() == text
    engine.dispose()


def test_the_round_trip(at_0034: str) -> None:
    command.upgrade(config(at_0034), REASON)
    command.downgrade(config(at_0034), BEFORE)
    engine = sa.create_engine(at_0034)
    assert "outcome_reason" not in _columns(engine, "mf_wsp_runs")
    with engine.connect() as conn:  # the pre-existing run survives both directions
        assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_runs")).scalar_one() == 1
    engine.dispose()
    command.upgrade(config(at_0034), "head")
