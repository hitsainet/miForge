"""Migration ``0028_wsp_provider`` (FTASKS 3.1): round trip, the CHECKs, the action widening, and a
downgrade that refuses while a WSP approval exists. Runs on SQLite, and on PostgreSQL when
``MIFORGE_TEST_DATABASE_URL`` names one (the serial PostgreSQL job).
"""

from __future__ import annotations

import json

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.script import ScriptDirectory

from tests.integration.wsp.wsp_db import (
    HEX,
    NOW,
    bundle_row,
    candidate_row,
    config,
    engine_at_head,
    insert,
    run_row,
)
from tests.support import db as support

#: The revision before 016's, read from the chain (it becomes 017's ``0027_run_hub_access`` at the
#: re-parent; ``test_migration_chain.py`` pins which).
BEFORE = str(
    ScriptDirectory.from_config(config("sqlite://")).get_revision("0028_wsp_provider").down_revision
)
WSP_TABLES = {
    "mf_wsp_bundles",
    "mf_wsp_endpoints",
    "mf_wsp_runs",
    "mf_wsp_worker_boots",
    "mf_wsp_reservations",
    "mf_wsp_spend_days",
    "mf_wsp_candidates",
    "mf_wsp_programs",
    "mf_wsp_verdicts",
    "mf_wsp_heldout_uses",
    "mf_wsp_usage",
}


def _tables(engine: sa.Engine) -> set[str]:
    return set(sa.inspect(engine).get_table_names())


def _approval(conn: sa.Connection, approval_id: str, action: str) -> None:
    insert(
        conn,
        "mf_approvals",
        id=approval_id,
        action=action,
        executor_key="x",
        summary="{}",
        payload="{}",
        request_digest=HEX,
        requested_by="agent:t",
        status="pending",
        expires_at=NOW,
        created_at=NOW,
    )


def test_eleven_tables_and_a_round_trip(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    assert WSP_TABLES <= _tables(engine)
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    command.downgrade(config(url), BEFORE)
    down = support.make_engine(url) if hasattr(support, "make_engine") else sa.create_engine(url)
    assert not (WSP_TABLES & _tables(down))
    down.dispose()
    command.upgrade(config(url), "head")
    up = sa.create_engine(url)
    assert WSP_TABLES <= _tables(up)
    up.dispose()


def test_a_candidate_outcome_without_a_patch_is_refused(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        bundle_row(conn)
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        run_row(conn, status="completed", outcome="candidate", patch_sha256=None)
    with engine.begin() as conn:  # the control: the same row with its patch is accepted
        run_row(conn, status="completed", outcome="candidate", patch_sha256=HEX)


def test_a_patch_without_a_candidate_outcome_is_refused(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        bundle_row(conn)
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        run_row(conn, status="completed", outcome="no_change", patch_sha256=HEX)


def test_an_external_endpoint_without_prices_is_refused(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    row = {
        "id": "ep1",
        "kind": "anthropic",
        "base_url": "https://api.anthropic.com",
        "model": "m",
        "locality": "external",
        "price_in_per_mtok": None,
        "price_out_per_mtok": 15,
        "output_cap_field": "max_tokens",
        "serving": "generic",
        "created_at": NOW,
        "created_by": "operator",
    }
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        insert(conn, "mf_wsp_endpoints", **row)
    with (
        engine.begin() as conn
    ):  # a local endpoint needs no prices; an external one with both is fine
        insert(
            conn,
            "mf_wsp_endpoints",
            **dict(row, id="ep2", locality="local", price_out_per_mtok=None),
        )
        insert(conn, "mf_wsp_endpoints", **dict(row, id="ep3", price_in_per_mtok=3))


def test_a_rerun_or_replay_with_a_token_is_refused(tmp_path) -> None:
    """R3A-13 / 9.2a: no token is ever minted for a program_rerun or a replay."""
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        bundle_row(conn)
    for i, style in enumerate(("program_rerun", "replay")):
        with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
            run_row(conn, run_id=f"wsp_00000000000{i}", style=style, gateway_token_sha256=HEX)
    with engine.begin() as conn:
        run_row(conn, run_id="wsp_000000000009", style="session", gateway_token_sha256=HEX)


def test_at_most_one_run_pushing_or_running(tmp_path) -> None:
    """0028's rule, read at ``0037``, the last revision that carries it: 019 P3's ``0038`` replaced
    the one-active index with one per scheduling key and one per worker (that rule's test is
    ``test_schedule_key_index.py``)."""
    head = engine_at_head(tmp_path)
    url = head.url.render_as_string(hide_password=False)
    head.dispose()
    command.downgrade(config(url), "0037_wsp_protocol_pass")
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        bundle_row(conn)
        run_row(conn, run_id="wsp_000000000001", status="running")
        run_row(conn, run_id="wsp_000000000002", status="queued")
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        run_row(conn, run_id="wsp_000000000003", status="pushing")
    engine.dispose()
    command.upgrade(config(url), "head")


def test_the_action_check_admits_the_two_new_actions_and_still_refuses_others(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        _approval(conn, "a1", "wsp_session")
        _approval(conn, "a2", "wsp_session_external")
        _approval(conn, "a3", "gpu_start")
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        _approval(conn, "a4", "not_an_action")


def test_the_rebuild_keeps_the_other_approval_checks_and_the_index(tmp_path) -> None:
    """0014's lesson: a reflecting SQLite rebuild drops CHECKs it cannot see. These must survive."""
    engine = engine_at_head(tmp_path)
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        insert(
            conn,
            "mf_approvals",
            id="b1",
            action="delete",
            executor_key="x",
            summary="{}",
            payload="{}",
            request_digest=HEX,
            requested_by="agent:t",
            status="nonsense",
            expires_at=NOW,
            created_at=NOW,
        )
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        insert(
            conn,
            "mf_approvals",
            id="b2",
            action="delete",
            executor_key="x",
            summary="{}",
            payload="{}",
            request_digest=HEX,
            requested_by="agent:t",
            status="executed",
            result=None,
            expires_at=NOW,
            created_at=NOW,
        )
    names = {ix["name"] for ix in sa.inspect(engine).get_indexes("mf_approvals")}
    assert "ix_mf_approvals_status_created_at" in names
    assert "batch_id" in {c["name"] for c in sa.inspect(engine).get_columns("mf_approvals")}


def test_the_downgrade_refuses_while_a_wsp_session_approval_exists(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        _approval(conn, "a1", "wsp_session")
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    with pytest.raises(RuntimeError, match="1 approval row"):
        command.downgrade(config(url), BEFORE)
    check = sa.create_engine(url)
    assert WSP_TABLES <= _tables(check)  # nothing was dropped
    with check.begin() as conn:
        conn.execute(sa.text("DELETE FROM mf_approvals"))
    check.dispose()
    command.downgrade(config(url), BEFORE)  # the control: with the row gone, it downgrades


def test_the_bundle_manifest_round_trips_as_json(tmp_path) -> None:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        insert(
            conn,
            "mf_wsp_bundles",
            bundle_sha256=HEX,
            owner="miworkbench",
            bundle_id="wsb_abc123",
            bundle_kind="linear_probe",
            artifact_kind="probe",
            artifact_id="p1",
            detector_digest=None,
            model="m",
            layer=1,
            sensitivity_value="not_sensitive",
            manifest=json.dumps({"k": 1}),
            bytes=1,
            import_status="imported",
            received_at=NOW,
            received_by="operator",
        )
        run_id = run_row(conn)
        candidate_row(conn, run_id)
