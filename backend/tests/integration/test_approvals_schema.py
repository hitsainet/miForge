"""``mf_approvals`` and ``mf_settings`` as migration 0021 builds them  (F12 task 2.3).

The two CHECKs hold in both directions (``mf_merges``' rule), the vocabularies are the
models', and neither table is something a reindex may drop.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from tests.support.db import migrated_engine


@pytest.fixture
def engine(tmp_path: Path) -> Any:
    built = migrated_engine(tmp_path / "db")
    yield built
    built.dispose()


def _row(**overrides: Any) -> dict[str, Any]:
    now = datetime.now(UTC)
    row: dict[str, Any] = {
        "id": "apr_1",
        "action": "gpu_start",
        "executor_key": "m.f",
        "summary": {},
        "payload": {},
        "request_digest": "0" * 64,
        "requested_by": "agent:t",
        "status": "pending",
        "created_at": now,
        "expires_at": now + timedelta(hours=1),
    }
    row.update(overrides)
    return row


def _insert(engine: Any, row: dict[str, Any]) -> None:
    from models.approval import Approval

    with engine.begin() as conn:
        conn.execute(sa.insert(Approval.__table__).values(**row))


def test_a_pending_row_is_accepted(engine: Any) -> None:
    _insert(engine, _row())


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "executed"},  # executed with no result
        {"status": "failed"},  # failed with no error
        {"status": "pending", "result": {"run_id": "r"}},  # a result on an unexecuted row
        {"status": "pending", "error": {"code": "x"}},  # an error on a row that did not fail
        {"status": "done"},
        {"action": "launch"},
    ],
)
def test_the_checks_refuse_an_inconsistent_row(engine: Any, overrides: dict[str, Any]) -> None:
    with pytest.raises(IntegrityError):
        _insert(engine, _row(**overrides))


def test_the_vocabularies_are_the_models(engine: Any) -> None:
    from api.agent_origin import APPROVAL_ACTIONS
    from models.approval import APPROVAL_ACTION_NAMES

    assert set(APPROVAL_ACTIONS) == set(APPROVAL_ACTION_NAMES)
    for action in APPROVAL_ACTION_NAMES:
        _insert(engine, _row(id=f"apr_{action}", action=action))


def test_neither_table_is_a_reindex_index_table() -> None:
    from services.reindex import INDEX_TABLES

    assert "mf_approvals" not in INDEX_TABLES and "mf_settings" not in INDEX_TABLES


def test_the_migration_round_trips(tmp_path: Path) -> None:
    from alembic import command
    from alembic.config import Config

    from tests.support.db import REPO_ROOT, _XArgs, database_url

    engine = migrated_engine(tmp_path / "rt")
    url = database_url(tmp_path / "rt")
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "backend" / "alembic"))
    cfg.attributes["configure_logger"] = False
    cfg.cmd_opts = _XArgs(url)  # type: ignore[assignment]
    command.downgrade(cfg, "0020_hotupdate_status")
    assert not sa.inspect(engine).has_table("mf_approvals")
    assert not sa.inspect(engine).has_table("mf_settings")
    command.upgrade(cfg, "head")
    assert sa.inspect(engine).has_table("mf_approvals")
    engine.dispose()
