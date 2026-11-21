"""Migration ``0031_wsp_endpoint_register`` (2026-11-20): the action CHECK admits
``wsp_endpoint_register`` and still refuses others, a SQLite rebuild keeps the other three CHECKs
and the index, the downgrade narrows it back, and refuses while a row uses the new action. Runs on
SQLite, and on PostgreSQL when ``MIFORGE_TEST_DATABASE_URL`` names one (the serial PostgreSQL job).
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from alembic import command

from tests.integration.wsp.wsp_db import HEX, NOW, config, engine_at_head, insert

REGISTER = "0031_wsp_endpoint_register"
BEFORE = "0030_wsp_live_progress"


def _approval(conn: sa.Connection, approval_id: str, action: str, **extra: object) -> None:
    insert(
        conn,
        "mf_approvals",
        **{
            "id": approval_id,
            "action": action,
            "executor_key": "x",
            "summary": "{}",
            "payload": "{}",
            "request_digest": HEX,
            "requested_by": "agent:t",
            "status": "pending",
            "expires_at": NOW,
            "created_at": NOW,
            **extra,
        },
    )


def test_the_action_check_admits_wsp_endpoint_register_and_still_refuses_others(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        _approval(conn, "a1", "wsp_endpoint_register")
        _approval(conn, "a2", "wsp_session_external")
        _approval(conn, "a3", "gpu_start")
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        _approval(conn, "a4", "not_an_action")
    engine.dispose()


def test_the_rebuild_keeps_the_other_approval_checks_and_the_index(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        _approval(conn, "b1", "wsp_endpoint_register", status="nonsense")
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        _approval(conn, "b2", "wsp_endpoint_register", status="executed", result=None)
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        _approval(conn, "b3", "wsp_endpoint_register", status="failed", error=None)
    names = {ix["name"] for ix in sa.inspect(engine).get_indexes("mf_approvals")}
    assert "ix_mf_approvals_status_created_at" in names
    assert "batch_id" in {c["name"] for c in sa.inspect(engine).get_columns("mf_approvals")}
    engine.dispose()


def test_the_downgrade_narrows_the_check_and_the_upgrade_widens_it_again(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    command.downgrade(config(url), BEFORE)
    down = sa.create_engine(url)
    with pytest.raises(sa.exc.IntegrityError), down.begin() as conn:
        _approval(conn, "c1", "wsp_endpoint_register")
    with down.begin() as conn:  # the control: 0028's actions still go in
        _approval(conn, "c2", "wsp_session")
    down.dispose()
    command.upgrade(config(url), REGISTER)
    up = sa.create_engine(url)
    with up.begin() as conn:
        _approval(conn, "c3", "wsp_endpoint_register")
    up.dispose()


def test_the_downgrade_refuses_while_a_wsp_endpoint_register_approval_exists(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        _approval(conn, "d1", "wsp_endpoint_register")
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    with pytest.raises(RuntimeError, match="1 approval row"):
        command.downgrade(config(url), BEFORE)
    check = sa.create_engine(url)
    with check.connect() as conn:  # the row is still there
        count = conn.execute(sa.text("SELECT COUNT(*) FROM mf_approvals")).scalar_one()
    assert count == 1
    with check.begin() as conn:
        conn.execute(sa.text("DELETE FROM mf_approvals"))
    check.dispose()
    command.downgrade(config(url), BEFORE)  # the control: with the row gone, it downgrades
