"""The settings model's four gaps, closed  (013; ENH-003 section 3.4, D12).

* **G1** - a ``log_level`` override is in force in the API after a restart, not only after the
  next write: the served app's startup re-applies it.
* **G2** - an agent's approved write records the agent and the approval, not ``operator``.
* **G3** - every write leaves one append-only history row, served newest first.
* **G4** - each setting is served with what it does, when a change takes effect, its
  environment value and its last change.

Everything runs through the production routes and the production approval executor on a
MIGRATED database (``tests.support.approval_world``), so the append-only triggers are the
migration's, not a test's.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import pytest
import sqlalchemy as sa

from tests.support.approval_world import AGENT, World


def _history(world: World, **params: Any) -> list[dict[str, Any]]:
    response = world.client.get("/api/v1/settings/history", params=params)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == len(body["changes"])
    return list(body["changes"])


def _override_row(world: World, key: str) -> Any:
    with world.engine.connect() as conn:
        return conn.execute(
            sa.text("SELECT value, updated_by FROM mf_settings WHERE key = :k"), {"k": key}
        ).one_or_none()


# ======================================================================== G3: the history


class TestEveryWriteIsRecorded:
    def test_a_set_and_a_reset_each_leave_one_row_with_both_values(self, world: World) -> None:
        put = world.client.put(
            "/api/v1/settings/checkpoint_lifecycle_keep_last_n",
            json={"value": 3, "reason": "the volume is filling"},
        )
        assert put.status_code == 200, put.text
        again = world.client.put(
            "/api/v1/settings/checkpoint_lifecycle_keep_last_n", json={"value": 4}
        )
        assert again.status_code == 200, again.text
        reset = world.client.delete(
            "/api/v1/settings/checkpoint_lifecycle_keep_last_n", params={"reason": "back to 10"}
        )
        assert reset.status_code == 200, reset.text

        changes = _history(world)
        assert world.count("mf_setting_changes") == 3
        assert [c["action"] for c in changes] == ["reset", "set", "set"]  # newest first
        newest, middle, oldest = changes
        for change in changes:
            assert change["key"] == "checkpoint_lifecycle_keep_last_n"
            assert change["actor"] == "operator"
            assert change["approval_id"] is None
            assert change["changed_at"].endswith("+00:00")
        assert (oldest["previous"], oldest["previous_source"]) == (10, "env")
        assert (oldest["value"], oldest["source"]) == (3, "override")
        assert oldest["reason"] == "the volume is filling"
        assert (middle["previous"], middle["previous_source"]) == (3, "override")
        assert (middle["value"], middle["source"], middle["reason"]) == (4, "override", None)
        assert (newest["previous"], newest["previous_source"]) == (4, "override")
        assert (newest["value"], newest["source"]) == (10, "env")
        assert newest["reason"] == "back to 10"
        assert newest["id"] > middle["id"] > oldest["id"]

    def test_a_refused_write_leaves_no_row(self, world: World) -> None:
        for key, value in (("log_level", "LOUD"), ("millm_trainer_token", "s3cret")):
            response = world.client.put(f"/api/v1/settings/{key}", json={"value": value})
            assert response.status_code == 422, response.text
        missing = world.client.delete("/api/v1/settings/log_level")
        assert missing.status_code == 404
        assert world.count("mf_setting_changes") == 0

    def test_a_reason_longer_than_the_column_is_refused(self, world: World) -> None:
        response = world.client.put(
            "/api/v1/settings/log_level", json={"value": "DEBUG", "reason": "x" * 2001}
        )
        assert response.status_code == 422, response.text
        assert world.count("mf_setting_changes") == 0
        assert world.count("mf_settings") == 0

    @pytest.mark.parametrize(
        "statement",
        [
            "UPDATE mf_setting_changes SET reason = 'edited'",
            "DELETE FROM mf_setting_changes",
        ],
    )
    def test_the_history_refuses_update_and_delete(self, world: World, statement: str) -> None:
        put = world.client.put("/api/v1/settings/log_level", json={"value": "WARNING"})
        assert put.status_code == 200, put.text
        assert world.count("mf_setting_changes") == 1
        with pytest.raises(sa.exc.DBAPIError) as raised, world.engine.begin() as conn:
            conn.execute(sa.text(statement))
        assert "append-only" in str(raised.value)
        assert world.count("mf_setting_changes") == 1
        assert _history(world)[0]["reason"] is None


class TestTheHistoryRoute:
    def test_it_filters_by_key_and_limits(self, world: World) -> None:
        for value in ("DEBUG", "INFO", "WARNING"):
            assert world.client.put("/api/v1/settings/log_level", json={"value": value}).is_success
        assert world.client.put(
            "/api/v1/settings/checkpoint_lifecycle_enabled", json={"value": True}
        ).is_success
        assert len(_history(world)) == 4
        assert [c["value"] for c in _history(world, key="log_level")] == [
            "WARNING",
            "INFO",
            "DEBUG",
        ]
        assert [c["value"] for c in _history(world, key="log_level", limit=2)] == [
            "WARNING",
            "INFO",
        ]
        assert _history(world, key="checkpoint_lifecycle_enabled")[0]["value"] is True

    def test_an_unknown_key_and_an_out_of_range_limit_are_refused(self, world: World) -> None:
        unknown = world.client.get("/api/v1/settings/history", params={"key": "no_such"})
        assert unknown.status_code == 422
        assert unknown.json()["error"]["code"] == "setting_unknown"
        for limit in (0, 501):
            bad = world.client.get("/api/v1/settings/history", params={"limit": limit})
            assert bad.status_code == 422, (limit, bad.text)

    def test_an_empty_history_is_an_empty_list(self, world: World) -> None:
        assert _history(world) == []


# ======================================================================== G2: the real actor


class TestTheRealActorIsRecorded:
    def test_an_approved_agent_write_names_the_agent_and_the_approval(self, world: World) -> None:
        asked = world.client.put(
            "/api/v1/settings/checkpoint_lifecycle_enabled",
            json={"value": True, "reason": "the volume is filling"},
            headers=AGENT,
        )
        assert asked.status_code == 202, asked.text
        approval_id = asked.json()["approval_id"]
        assert asked.json()["summary"]["reason"] == "the volume is filling"
        # Waiting: nothing is recorded until the operator decides.
        assert world.count("mf_setting_changes") == 0

        decided = world.client.post(f"/api/v1/approvals/{approval_id}/approve")
        assert decided.json()["status"] == "executed", decided.text

        row = _override_row(world, "checkpoint_lifecycle_enabled")
        assert row is not None and row.updated_by == "agent:forge-mcp"
        [change] = _history(world)
        assert change["actor"] == "agent:forge-mcp"
        assert change["approval_id"] == approval_id
        assert change["reason"] == "the volume is filling"
        assert (change["previous"], change["value"]) == (False, True)
        listed = world.client.get("/api/v1/settings").json()["settings"]
        assert listed["checkpoint_lifecycle_enabled"]["updated_by"] == "agent:forge-mcp"
        assert listed["checkpoint_lifecycle_enabled"]["last_change"] == change

        # The identity does not leak into the operator's next call in the same thread.
        mine = world.client.put(
            "/api/v1/settings/checkpoint_lifecycle_enabled", json={"value": False}
        )
        assert mine.status_code == 200, mine.text
        assert _override_row(world, "checkpoint_lifecycle_enabled").updated_by == "operator"
        latest = _history(world)[0]
        assert (latest["actor"], latest["approval_id"]) == ("operator", None)
        assert world.count("mf_setting_changes") == 2

    def test_an_approved_agent_reset_names_the_agent_and_the_approval(self, world: World) -> None:
        assert world.client.put("/api/v1/settings/log_level", json={"value": "DEBUG"}).is_success
        asked = world.client.delete(
            "/api/v1/settings/log_level", params={"reason": "done chasing"}, headers=AGENT
        )
        assert asked.status_code == 202, asked.text
        assert asked.json()["summary"]["reason"] == "done chasing"
        decided = world.client.post(f"/api/v1/approvals/{asked.json()['approval_id']}/approve")
        assert decided.json()["status"] == "executed", decided.text
        newest = _history(world)[0]
        assert newest["action"] == "reset"
        assert newest["actor"] == "agent:forge-mcp"
        assert newest["approval_id"] == asked.json()["approval_id"]
        assert newest["reason"] == "done chasing"
        assert world.count("mf_settings") == 0

    def test_a_rejected_agent_write_leaves_no_history(self, world: World) -> None:
        asked = world.client.put(
            "/api/v1/settings/log_level", json={"value": "DEBUG"}, headers=AGENT
        )
        assert asked.status_code == 202
        rejected = world.client.post(f"/api/v1/approvals/{asked.json()['approval_id']}/reject")
        assert rejected.status_code == 200, rejected.text
        assert world.count("mf_setting_changes") == 0
        assert world.count("mf_settings") == 0

    def test_outside_an_approval_there_is_no_executing_actor(self) -> None:
        from api.agent_origin import Actor, executing_actor, executing_as

        assert executing_actor() is None
        actor = Actor(origin="agent", agent="agent:x", approval_id="apr_1")
        with executing_as(actor):
            assert executing_actor() == actor
        assert executing_actor() is None


# ==================================================================== G4: what each one does


class TestEachSettingSaysWhatItDoes:
    def test_the_listing_serves_the_catalog_and_the_environment_value(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from core import settings_catalog
        from core.config import Settings, reset_settings

        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        reset_settings()
        assert world.client.put("/api/v1/settings/log_level", json={"value": "ERROR"}).is_success
        settings = world.client.get("/api/v1/settings").json()["settings"]
        assert set(settings) == set(Settings.model_fields)
        for key, row in settings.items():
            info = settings_catalog.info_for(key)
            assert row["label"] == info.label, key
            assert row["description"] == info.description, key
            assert row["effect"] == settings_catalog.effect_payload(key), key
            assert row["group"] == settings_catalog.GROUP_OF[key], key
        body = world.client.get("/api/v1/settings").json()
        assert body["groups"] == settings_catalog.groups_payload()
        assert body["settings"]["hotupdate_verify_prompt"]["group"] == "hot_update"
        log_level = settings["log_level"]
        assert (log_level["value"], log_level["env_value"]) == ("ERROR", "WARNING")
        assert log_level["effect"]["when"] == "now"
        assert log_level["updated_by"] == "operator"
        assert log_level["updated_at"] is not None
        assert log_level["last_change"]["value"] == "ERROR"
        untouched = settings["checkpoint_lifecycle_dry_run"]
        assert (untouched["updated_at"], untouched["updated_by"], untouched["last_change"]) == (
            None,
            None,
            None,
        )
        fixed = settings["database_url"]
        assert fixed["effect"] == {
            "when": "restart",
            "text": settings_catalog.EFFECT_TEXT["restart"],
            "pod": "miforge-backend",
        }

    def test_the_environment_value_of_a_secret_is_redacted_too(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from core.config import reset_settings

        sentinel = "s3ntinel-013-history"
        monkeypatch.setenv("MILLM_TRAINER_TOKEN", sentinel)
        monkeypatch.setenv("DATABASE_URL", f"postgresql+psycopg://u:{sentinel}@db:5432/m")
        reset_settings()
        response = world.client.get("/api/v1/settings")
        assert sentinel not in response.text
        row = response.json()["settings"]["millm_trainer_token"]
        assert row["env_value"] == "<redacted>"
        assert "<redacted>" in response.json()["settings"]["database_url"]["env_value"]

    def test_a_now_setting_reaches_the_api_without_a_restart(self, world: World) -> None:
        """``millm_base_url`` states ``now``: the next health read reports miLLM configured."""
        before = world.client.get("/api/health").json()["integrations"]["millm"]
        assert before["configured"] is False
        put = world.client.put(
            "/api/v1/settings/millm_base_url", json={"value": "http://millm.test:8000"}
        )
        assert put.status_code == 200, put.text
        after = world.client.get("/api/health").json()["integrations"]["millm"]
        assert after["configured"] is True

    def test_a_log_level_write_applies_to_the_api_at_once(self, world: World) -> None:
        root = logging.getLogger()
        try:
            assert world.client.put(
                "/api/v1/settings/log_level", json={"value": "ERROR"}
            ).is_success
            assert root.level == logging.ERROR
        finally:
            from core.logging import configure_logging

            configure_logging("INFO")


# =================================================================== G1: after a restart


def test_a_restart_applies_a_stored_log_level_override(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The served app (``create_asgi_app``) starts over a database holding ``log_level=DEBUG``
    and the API logs at DEBUG - not at the environment's INFO until the next write."""
    from fastapi.testclient import TestClient

    from api.app import create_asgi_app
    from core import runtime_settings
    from core.logging import configure_logging
    from db import session as db_session
    from tests.support.db import session_for

    assert world.client.put("/api/v1/settings/log_level", json={"value": "DEBUG"}).is_success
    # A fresh process: nothing cached, and logging configured from the environment.
    runtime_settings.reset()
    configure_logging("INFO")

    @contextmanager
    def scope() -> Any:
        with session_for(world.engine) as session:
            yield session

    monkeypatch.setattr(db_session, "session_scope", scope)
    root = logging.getLogger()
    try:
        app = create_asgi_app()
        assert root.level == logging.INFO, "create_app configured from the environment"
        with TestClient(app) as client:
            assert client.get("/api/v1/approvals/actions").status_code == 200
            assert runtime_settings.overrides() == {"log_level": "DEBUG"}
            assert root.level == logging.DEBUG
    finally:
        runtime_settings.reset()
        configure_logging("INFO")
