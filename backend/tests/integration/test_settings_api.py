"""The settings surface, and the JUNCTION into the readers  (F12; FR-012.7, .8; D5).

The test that matters most is :class:`TestAnOverrideReachesTheCodeThatUsesIt`. A settings
route that stores a value is half a capability; the other half is every reader seeing it, in
the process that reads it. So the override is written through the route, the in-process cache
is **cleared** (the worker is another process), a real Celery task is run with ``apply()`` so
the ``task_prerun`` hook fires, and inside it the verifier ``AttachAcquirer`` builds is asked
what tolerance it judges with. Removing the hook's connection turns it red.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.support.approval_world import AGENT, World

SENTINEL = "s3ntinel-v4lue-f012"


def _core(row: dict[str, Any]) -> dict[str, Any]:
    """F12's three fields of a listed setting; 013's catalog fields are pinned in
    ``test_settings_history.py``."""
    return {name: row[name] for name in ("value", "source", "runtime_writable")}


class TestReading:
    def test_no_secret_reaches_the_response(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from core.config import reset_settings

        monkeypatch.setenv("MILLM_TRAINER_TOKEN", SENTINEL)
        monkeypatch.setenv("DATABASE_URL", f"postgresql+psycopg://miforge:{SENTINEL}@db:5432/m")
        monkeypatch.setenv("REDIS_URL", f"redis://:{SENTINEL}@redis:6379/0")
        reset_settings()
        response = world.client.get("/api/v1/settings")
        assert response.status_code == 200
        assert SENTINEL not in response.text
        settings = response.json()["settings"]
        assert settings["millm_trainer_token"]["value"] == "<redacted>"
        assert (
            settings["database_url"]["value"] == "postgresql+psycopg://miforge:<redacted>@db:5432/m"
        )
        assert "<redacted>" in settings["redis_url"]["value"]
        # positive control: the scan reads a real body, not an empty one
        assert settings["log_level"]["value"] == "INFO"

    def test_every_field_is_reported_with_its_source(self, world: World) -> None:
        from core.config import RUNTIME_KEYS, Settings

        body = world.client.get("/api/v1/settings").json()
        assert set(body["settings"]) == set(Settings.model_fields)
        writable = {k for k, v in body["settings"].items() if v["runtime_writable"]}
        assert writable == set(RUNTIME_KEYS)
        assert {v["source"] for v in body["settings"].values()} == {"env"}


class TestWriting:
    def test_an_override_is_stored_served_and_cleared(self, world: World) -> None:
        put = world.client.put(
            "/api/v1/settings/checkpoint_lifecycle_keep_last_n", json={"value": 3}
        )
        assert put.status_code == 200, put.text
        assert put.json() == {
            "key": "checkpoint_lifecycle_keep_last_n",
            "value": 3,
            "previous": 10,
            "source": "override",
        }
        read = world.client.get("/api/v1/settings").json()["settings"]
        assert _core(read["checkpoint_lifecycle_keep_last_n"]) == {
            "value": 3,
            "source": "override",
            "runtime_writable": True,
        }
        cleared = world.client.delete("/api/v1/settings/checkpoint_lifecycle_keep_last_n")
        assert cleared.status_code == 200
        assert cleared.json()["value"] == 10
        again = world.client.get("/api/v1/settings").json()["settings"]
        assert again["checkpoint_lifecycle_keep_last_n"]["source"] == "env"
        missing = world.client.delete("/api/v1/settings/checkpoint_lifecycle_keep_last_n")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "setting_not_overridden"

    @pytest.mark.parametrize(
        "key",
        sorted(
            {
                "database_url",
                "redis_url",
                "checkpoint_root",
                "trainer_state_root",
                "parts_root",
                "checkpoint_read_roots",
                "millm_trainer_token",
                "environment",
                "no_such_setting",
                "approval_ttl_hours",
            }
        ),
    )
    def test_a_key_off_the_allow_list_is_refused(self, world: World, key: str) -> None:
        response = world.client.put(f"/api/v1/settings/{key}", json={"value": "x"})
        assert response.status_code == 422, response.text
        assert response.json()["error"]["code"] == "setting_not_runtime"
        assert world.count("mf_settings") == 0

    def test_the_never_writable_list_is_disjoint_from_the_allow_list(self) -> None:
        from core.config import NEVER_WRITABLE, RUNTIME_KEYS, Settings

        assert not NEVER_WRITABLE & RUNTIME_KEYS
        assert NEVER_WRITABLE <= set(Settings.model_fields)
        assert RUNTIME_KEYS <= set(Settings.model_fields)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("log_level", "LOUD"),
            ("millm_base_url", "http://user:pw@millm:8000"),
            ("checkpoint_lifecycle_keep_last_n", "many"),
        ],
    )
    def test_an_invalid_value_is_refused_without_echoing_it(
        self, world: World, key: str, value: str
    ) -> None:
        response = world.client.put(f"/api/v1/settings/{key}", json={"value": value})
        assert response.status_code == 422, response.text
        assert response.json()["error"]["code"] == "setting_value_invalid"
        assert value not in response.text
        assert world.count("mf_settings") == 0


class TestTheAgentWaits:
    def test_an_agent_setting_write_waits(self, world: World) -> None:
        response = world.client.put(
            "/api/v1/settings/checkpoint_lifecycle_enabled", json={"value": True}, headers=AGENT
        )
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["action"] == "config_write"
        assert body["summary"] == {
            "route": "PUT /api/v1/settings/{key}",
            "key": "checkpoint_lifecycle_enabled",
            "old": False,
            "new": True,
            "reason": None,
        }
        assert world.count("mf_settings") == 0
        read = world.client.get("/api/v1/settings").json()["settings"]
        assert read["checkpoint_lifecycle_enabled"]["value"] is False
        # The operator approves; the override is then stored and in force.
        decided = world.client.post(f"/api/v1/approvals/{body['approval_id']}/approve")
        assert decided.json()["status"] == "executed", decided.text
        read = world.client.get("/api/v1/settings").json()["settings"]
        assert _core(read["checkpoint_lifecycle_enabled"]) == {
            "value": True,
            "source": "override",
            "runtime_writable": True,
        }

    def test_an_agent_setting_reset_waits(self, world: World) -> None:
        assert (
            world.client.put("/api/v1/settings/log_level", json={"value": "DEBUG"}).status_code
            == 200
        )
        response = world.client.delete("/api/v1/settings/log_level", headers=AGENT)
        assert response.status_code == 202, response.text
        assert response.json()["action"] == "config_write"
        assert response.json()["summary"]["old"] == "DEBUG"
        assert world.count("mf_settings") == 1

    def test_an_agent_request_for_a_secret_key_is_refused_not_stored(self, world: World) -> None:
        response = world.client.put(
            "/api/v1/settings/millm_trainer_token", json={"value": SENTINEL}, headers=AGENT
        )
        assert response.status_code == 422
        assert world.count("mf_approvals") == 0
        assert SENTINEL not in response.text


class TestAnOverrideReachesTheCodeThatUsesIt:
    """FR-012.8's acceptance criterion: an override reaches a WORKER's reader."""

    @pytest.fixture
    def worker_process(self, world: World, monkeypatch: pytest.MonkeyPatch) -> Any:
        """The worker's view: its own cache (empty) and a session on the same database."""
        from contextlib import contextmanager

        from core import runtime_settings
        from db import session as db_session
        from tests.support.db import session_for

        @contextmanager
        def scope() -> Any:
            with session_for(world.engine) as session:
                yield session

        monkeypatch.setattr(db_session, "session_scope", scope)
        runtime_settings.reset()
        return scope

    def _run_a_task_and_look(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Run the REAL `run_step_loop` task through Celery; look from inside the run."""
        from training import run_driver
        from training.hotupdate import settings as hotupdate_settings
        from training.hotupdate import verify
        from training.hotupdate.attach import AttachAcquirer
        from workers.celery_app import celery_app
        from workers.checkpoint_lifecycle import policy_from_settings

        seen: dict[str, Any] = {}

        class RecordingVerifier:
            def __init__(self, **kwargs: Any) -> None:
                seen["verifier"] = kwargs

        def drive(run_id: str, **kwargs: Any) -> dict[str, Any]:
            live = hotupdate_settings.load()
            seen["load"] = live
            monkeypatch.setattr(verify, "Verifier", RecordingVerifier)
            AttachAcquirer()._verifier(None, "served-model", live, tokenizer=object())
            seen["policy"] = policy_from_settings()
            return {"run_id": run_id}

        monkeypatch.setattr(run_driver, "drive_recorded_run", drive)
        task = celery_app.tasks["miforge.train.run_step_loop"]
        task.apply(args=["run-junction"]).get()
        return seen

    def test_a_tolerance_set_through_the_route_reaches_the_verifier(
        self, world: World, worker_process: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from core import runtime_settings

        for key, value in (
            ("hotupdate_verification_tolerance_float32", 0.0123),
            ("hotupdate_verification_tolerance", 0.25),
            ("millm_base_url", "http://millm.test:8000"),
            ("checkpoint_lifecycle_keep_last_n", 4),
        ):
            put = world.client.put(f"/api/v1/settings/{key}", json={"value": value})
            assert put.status_code == 200, put.text
        runtime_settings.reset()  # the worker is another process: nothing cached here
        assert runtime_settings.overrides() == {}

        seen = self._run_a_task_and_look(monkeypatch)

        assert seen["load"].verification_tolerance_float32 == 0.0123
        assert seen["verifier"]["tolerance"] == 0.0123  # float32 is the default precision
        assert seen["verifier"]["native_tolerance"] == 0.25
        assert seen["verifier"]["provider"].base_url == "http://millm.test:8000"
        assert seen["policy"].keep_last_n == 4

    def test_without_an_override_the_environment_value_reaches_it(
        self, world: World, worker_process: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from core.config import FLOAT32_VERIFICATION_TOLERANCE

        seen = self._run_a_task_and_look(monkeypatch)
        assert seen["verifier"]["tolerance"] == FLOAT32_VERIFICATION_TOLERANCE
        assert seen["policy"].keep_last_n == 10

    def test_the_prerun_hook_is_connected_in_the_live_signal(self) -> None:
        import weakref

        from celery.signals import task_prerun

        from workers import celery_app

        receivers = [
            ref() if isinstance(ref, weakref.ReferenceType) else ref
            for _, ref in task_prerun.receivers
        ]
        assert receivers, "task_prerun has no receivers at all - the scan read nothing"
        assert celery_app.refresh_runtime_settings in receivers


def test_the_served_app_reads_the_overrides_once_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``create_asgi_app`` - the object uvicorn serves - refreshes the cache on startup."""
    from fastapi.testclient import TestClient

    from api.app import create_asgi_app
    from core import runtime_settings

    calls: list[str] = []
    monkeypatch.setattr(
        runtime_settings, "refresh_from_database", lambda: calls.append("refresh") or True
    )
    app = create_asgi_app()
    assert calls == [], "the refresh ran at construction rather than at startup"
    with TestClient(app) as client:
        assert client.get("/api/v1/approvals/actions").status_code == 200
    assert calls == ["refresh"]
