"""Migration ``0042_wsp_editor_wait`` (2026-11-27, run ``wsp_4fbf5a18758e``): the column is a
nullable timestamp (not the NUMERIC a raw DDL type reflects as on SQLite), the model declares the
same, and the round trip drops and restores it. Runs on SQLite, and on PostgreSQL when
``MIFORGE_TEST_DATABASE_URL`` names one.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from alembic import command

from tests.integration.wsp.wsp_db import config, engine_at_head

WAIT = "0042_wsp_editor_wait"
BEFORE = "0041_wsp_worker_disk"
COLUMN = "editor_wait_since"


def _columns(engine: sa.Engine) -> dict[str, dict]:
    return {c["name"]: c for c in sa.inspect(engine).get_columns("mf_wsp_runs")}


@pytest.fixture()
def at_0041(tmp_path):
    engine = engine_at_head(tmp_path)
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    command.downgrade(config(url), BEFORE)
    old = sa.create_engine(url)
    assert COLUMN not in _columns(old)
    old.dispose()
    yield url


def test_the_column_is_a_nullable_aware_timestamp(at_0041: str) -> None:
    command.upgrade(config(at_0041), WAIT)
    engine = sa.create_engine(at_0041)
    try:
        column = _columns(engine)[COLUMN]
        assert column["nullable"] is True
        # Not the NUMERIC a raw TIMESTAMP WITH TIME ZONE reflects as on SQLite.
        assert isinstance(column["type"], sa.DateTime), column["type"]
    finally:
        engine.dispose()


def test_the_round_trip(at_0041: str) -> None:
    command.upgrade(config(at_0041), WAIT)
    command.downgrade(config(at_0041), BEFORE)
    engine = sa.create_engine(at_0041)
    try:
        assert COLUMN not in _columns(engine)
    finally:
        engine.dispose()
    command.upgrade(config(at_0041), "head")
    engine = sa.create_engine(at_0041)
    try:
        assert COLUMN in _columns(engine)
    finally:
        engine.dispose()


def test_the_model_and_the_migration_agree() -> None:
    from wsp.models import WspRun

    column = WspRun.__table__.c[COLUMN]
    assert column.nullable and isinstance(column.type, sa.DateTime) and column.type.timezone
