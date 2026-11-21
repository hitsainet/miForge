"""A run remembers whose Hub credential its worker may use (017 FR-017.27; 017-D23, D25).

``mf_runs.hub_access`` is ``anonymous`` when an agent asked for the run - directly, or as an
approved request the executor ran for it - and null for the operator's. A continue reopens the
same row and never clears ``anonymous``; a branch ORs in its parent's value; a chain stage after
stage 0 takes the previous stage's. Driven through the production gate, approval executor and
``run_launch.launch`` (``tests/support/approval_world.py`` stands in only for the card, the
enqueue and the validator's non-Hub steps).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
import sqlalchemy as sa

from tests.support.approval_world import AGENT, World, definition


def _access(world: World, run_id: str) -> str | None:
    from workers.run_record import RUNS

    with world.engine.connect() as conn:
        return conn.execute(sa.select(RUNS.c.hub_access).where(RUNS.c.id == run_id)).scalar_one()


def _agent_define_approved(world: World, run_id: str) -> None:
    filed = world.client.post("/api/v1/runs/define", json=definition(run_id), headers=AGENT)
    assert filed.status_code == 202, filed.text
    approved = world.client.post(f"/api/v1/approvals/{filed.json()['approval_id']}/approve")
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "executed", approved.json()


def _operator_define(world: World, body: dict[str, Any]) -> None:
    response = world.client.post("/api/v1/runs/define", json=body)
    assert response.status_code == 202, response.text


def _launch_as(world: World, body: dict[str, Any], *, agent: bool) -> None:
    """``run_launch.launch`` as the continue route's body runs it: as its requester."""
    from api.agent_origin import Actor, executing_as
    from schemas.run_definition import RunDefinition
    from services import hub_access, run_launch

    actor = Actor(origin="agent", agent="agent:forge-mcp") if agent else Actor(origin="operator")
    with world.session() as session:
        scope = executing_as(actor) if agent else hub_access.requesting(actor)
        with scope:
            run_launch.launch(RunDefinition(**body), session=session, enqueue=lambda run_id: None)
        session.commit()


def _branch_of(parent: str, run_id: str) -> dict[str, Any]:
    body = definition(run_id)
    body["model_source"] = {**body["model_source"], "resume_from": {"run_id": parent, "step": 4}}
    return body


def test_an_agents_define_approved_by_the_operator_is_anonymous(world: World) -> None:
    _agent_define_approved(world, "run-agent")
    assert _access(world, "run-agent") == "anonymous"


def test_the_operators_own_define_is_null(world: World) -> None:
    _operator_define(world, definition("run-op"))
    assert _access(world, "run-op") is None


def test_an_agent_continuing_an_operators_run_makes_it_anonymous(world: World) -> None:
    _operator_define(world, definition("run-op"))
    _launch_as(world, definition("run-op"), agent=True)
    assert _access(world, "run-op") == "anonymous"


def test_an_operator_continuing_an_anonymous_run_keeps_it_anonymous(world: World) -> None:
    _agent_define_approved(world, "run-agent")
    _launch_as(world, definition("run-agent"), agent=False)
    assert _access(world, "run-agent") == "anonymous"


def test_an_operator_branching_an_anonymous_run_is_anonymous(world: World) -> None:
    _agent_define_approved(world, "run-agent")
    _launch_as(world, _branch_of("run-agent", "run-branch"), agent=False)
    assert _access(world, "run-branch") == "anonymous"


def test_an_operator_branching_an_operators_run_is_null(world: World) -> None:
    _operator_define(world, definition("run-op"))
    _launch_as(world, _branch_of("run-op", "run-branch"), agent=False)
    assert _access(world, "run-branch") is None


def test_a_chain_stage_after_stage_0_takes_the_previous_stages_value(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sweep starts stage 1 with no requester; it inherits rather than failing closed.

    ``_start_stage`` is driven with a recording ``launch_fn`` that asks ``access_for_launch``
    what the launched run would record, so the scope around the launch is what is measured.
    """
    from services import hub_access
    from services.chains import service as chains
    from workers.run_record import RUNS

    seen: list[str | None] = []

    def launch(definition: Any, **_: Any) -> None:
        seen.append(hub_access.access_for_launch(parent_access=None, existing_access=None))

    previous: dict[int, Any] = {}
    monkeypatch.setattr(chains, "_stage_row", lambda session, chain_id, index: previous.get(index))
    monkeypatch.setattr(chains, "_claim_stage", lambda *args: True)
    for access in (None, "anonymous"):
        stage0 = definition(f"stage0-{access}")
        _operator_define(world, stage0)
        with world.engine.begin() as conn:
            conn.execute(
                sa.update(RUNS).where(RUNS.c.id == stage0["run_id"]).values(hub_access=access)
            )
        previous[0] = SimpleNamespace(run_id=stage0["run_id"])
        stage1 = SimpleNamespace(id=1, stage_index=1, definition=definition(None))
        with world.session() as session:
            # The chain-membership UPDATE after the recording launch matches no row (it
            # launched nothing), which is harmless and not under test here.
            chains._start_stage(
                session,
                chain_id="chain-1",
                stage=stage1,
                model_source=None,
                launch_fn=launch,
                log_root=None,
                now=datetime.now(tz=UTC),
            )
            session.rollback()
    assert seen == [None, "anonymous"]


# ==================================================== the database narrows it too (OR2-017-4)


@pytest.mark.parametrize("value", ["operator", "ANONYMOUS", ""])
def test_the_migrated_column_refuses_any_value_but_anonymous_or_null(
    tmp_path: Any, value: str
) -> None:
    """Migration 0027's CHECK (R2 review fix OR2-017-4, 2026-11-16): the column takes
    ``anonymous`` or null, on whichever engine the suite runs (SQLite or PostgreSQL)."""
    from sqlalchemy.exc import IntegrityError

    from tests.support.db import migrated_engine, session_for
    from tests.support.jobs import insert_run
    from workers.run_record import RUNS

    engine = migrated_engine(tmp_path / "db")
    try:
        with session_for(engine) as session:
            insert_run(session, "run-ok", status="pending")
            for allowed in ("anonymous", None):
                session.execute(
                    sa.update(RUNS).where(RUNS.c.id == "run-ok").values(hub_access=allowed)
                )
            session.commit()
            with pytest.raises(IntegrityError):
                session.execute(
                    sa.update(RUNS).where(RUNS.c.id == "run-ok").values(hub_access=value)
                )
                session.flush()
            session.rollback()
    finally:
        engine.dispose()


def test_0027_reverses_one_step_and_upgrades_again(tmp_path: Any) -> None:
    """``0027``'s downgrade drops the column, and a re-upgrade brings it back with its CHECK.

    ``test_db_migration.test_the_migration_reverses`` goes only to ``base``, where ``mf_runs``
    itself is dropped, so a downgrade that dropped nothing was green (R3 review fix OR3-017-4,
    2026-11-16; the reviewer's case, control N-F10)."""
    from alembic import command
    from alembic.config import Config
    from sqlalchemy.exc import IntegrityError

    from tests.support import db as support
    from tests.support.jobs import insert_run

    engine = support.migrated_engine(tmp_path / "db")
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    cfg = Config(str(support.REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(support.REPO_ROOT / "backend" / "alembic"))
    cfg.attributes["configure_logger"] = False
    cfg.cmd_opts = support._XArgs(url)  # type: ignore[assignment]
    command.downgrade(cfg, "0026_legacy_name_uses")
    down = support.make_engine(url)
    try:
        assert "hub_access" not in {c["name"] for c in sa.inspect(down).get_columns("mf_runs")}
    finally:
        down.dispose()
    command.upgrade(cfg, "head")
    up = support.make_engine(url)
    try:
        assert "hub_access" in {c["name"] for c in sa.inspect(up).get_columns("mf_runs")}
        with support.session_for(up) as session:
            insert_run(session, "run-x", status="pending")
            session.commit()
            with pytest.raises(IntegrityError):
                session.execute(
                    sa.text("UPDATE mf_runs SET hub_access = 'operator' WHERE id = 'run-x'")
                )
                session.flush()
            session.rollback()
    finally:
        up.dispose()
