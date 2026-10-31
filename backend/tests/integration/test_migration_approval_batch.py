"""Migration ``0025_approval_batch``, round-tripped **with rows present**, on either engine.

``mf_approvals.batch_id`` is additive (ENH-006 D79). The round trip seeds two approvals at
``0024`` - one pending, one executed (the executed one is what the two CHECK constraints
``executed_iff_result`` and ``failed_iff_error`` hold rows to) - upgrades, writes a batch id,
downgrades and upgrades again, and asserts:

1. every row and its outcome survives both directions;
2. the column appears, carries a value, is dropped, and comes back empty (the batch record is
   the one thing a downgrade loses, asserted as a loss rather than left unsaid);
3. the table's CHECK constraints still bite after the downgrade - checked by **inserting a row
   that breaks one**, not by reflection, which does not see SQLite CHECKs at all.

Runs on SQLite by default and on PostgreSQL when ``MIFORGE_TEST_DATABASE_URL`` names one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from db.base import JsonColumn
from models.approval import NullableJson
from tests.support import db as support

BEFORE = "0024_setting_changes"
WITH_BATCH = "0025_approval_batch"

_APPROVALS = sa.table(
    "mf_approvals",
    sa.column("id", sa.String),
    sa.column("action", sa.String),
    sa.column("executor_key", sa.String),
    sa.column("summary", JsonColumn),
    sa.column("payload", JsonColumn),
    sa.column("request_digest", sa.String),
    sa.column("requested_by", sa.String),
    sa.column("status", sa.String),
    sa.column("result", NullableJson),
    sa.column("decided_by", sa.String),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("expires_at", sa.DateTime(timezone=True)),
)

_SEED = (
    {"id": "apr_waiting", "status": "pending", "result": None, "decided_by": None},
    {"id": "apr_done", "status": "executed", "result": {"key": "x"}, "decided_by": "operator"},
)


def _config(url: str) -> Config:
    cfg = Config(str(support.REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(support.REPO_ROOT / "backend" / "alembic"))
    cfg.attributes["configure_logger"] = False
    cfg.cmd_opts = support._XArgs(url)  # type: ignore[assignment]
    return cfg


def _columns(engine: sa.Engine) -> set[str]:
    return {column["name"] for column in sa.inspect(engine).get_columns("mf_approvals")}


def _rows(engine: sa.Engine) -> list[tuple[str, str, str | None]]:
    with engine.connect() as conn:
        rows = conn.execute(
            sa.text("SELECT id, status, decided_by FROM mf_approvals ORDER BY id")
        ).fetchall()
    return [(row[0], row[1], row[2]) for row in rows]


@pytest.fixture()
def seeded(tmp_path: Path):
    url = support.database_url(tmp_path)
    cfg = _config(url)
    if support._already_stamped(url):
        command.downgrade(cfg, "base")
    command.upgrade(cfg, BEFORE)
    engine = support.make_engine(url)
    with engine.begin() as conn:
        for seed in _SEED:
            conn.execute(
                _APPROVALS.insert().values(
                    action="config_write",
                    executor_key="api.v1.endpoints.settings.write_setting",
                    summary={"key": "log_level"},
                    payload={"key": "log_level"},
                    request_digest="0" * 64,
                    requested_by="agent:forge-mcp",
                    created_at=sa.func.now(),
                    expires_at=sa.func.now(),
                    **seed,
                )
            )
    try:
        yield cfg, engine
    finally:
        engine.dispose()


def test_the_round_trip_keeps_every_row_and_the_constraints(seeded) -> None:
    cfg, engine = seeded
    expected = [("apr_done", "executed", "operator"), ("apr_waiting", "pending", None)]
    assert "batch_id" not in _columns(engine)

    command.upgrade(cfg, WITH_BATCH)
    assert "batch_id" in _columns(engine)
    assert _rows(engine) == expected
    with engine.begin() as conn:
        assert conn.execute(sa.text("SELECT batch_id FROM mf_approvals")).scalars().all() == [
            None,
            None,
        ], "existing rows were decided on their own; nothing may be backfilled"
        conn.execute(sa.text("UPDATE mf_approvals SET batch_id = 'bat_1' WHERE id = 'apr_done'"))

    command.downgrade(cfg, BEFORE)
    engine.dispose()
    assert "batch_id" not in _columns(engine)
    assert _rows(engine) == expected
    # The CHECK constraints survived: an executed row with no result is still refused.
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE mf_approvals SET status = 'executed' WHERE id = 'apr_waiting'")
        )

    command.upgrade(cfg, WITH_BATCH)
    engine.dispose()
    assert _rows(engine) == expected
    with engine.connect() as conn:
        # The one loss a downgrade causes: which decisions were made together.
        assert conn.execute(sa.text("SELECT batch_id FROM mf_approvals")).scalars().all() == [
            None,
            None,
        ]
