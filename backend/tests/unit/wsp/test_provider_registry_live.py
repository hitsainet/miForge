"""FR-016.8 through the mounted app (FTASKS 7.1): availability follows ``WSP_PROVIDERS_AVAILABLE``;
``provider_unknown`` and ``provider_unavailable`` are named; nothing falls back to another provider.
And
FR-016.22's route (7.8): ``POST /wsp/runs/validate`` freezes exactly what ``wsp.freeze`` freezes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from tests.integration.wsp.wsp_db import NOW, engine_at_head, insert
from tests.support.jobs import session_for
from tests.support.wsp_fixtures import enable_wsp, make_bundle
from wsp.settings import get_wsp_settings
from wsp.weight_ops.hashing import jcs_sha256


@pytest.fixture()
def env(monkeypatch) -> Iterator[None]:
    get_wsp_settings.cache_clear()
    yield
    get_wsp_settings.cache_clear()


def _client(monkeypatch, available: str, engine=None, unset: tuple[str, ...] = ()) -> TestClient:
    enable_wsp(monkeypatch, available)
    for name in unset:
        monkeypatch.setenv(name, "")
    get_wsp_settings.cache_clear()
    app = create_app()
    if engine is not None:

        def one_session():
            with session_for(engine) as session:
                yield session

        app.dependency_overrides[session_dependency] = one_session
    return TestClient(app)


def _listing(monkeypatch, available: str) -> list[tuple[str, bool, str]]:
    providers = _client(monkeypatch, available).get("/api/v1/wsp/providers").json()["providers"]
    return [(p["id"], p["available"], p["version"]) for p in providers]


def test_the_listing_follows_the_setting(env, monkeypatch) -> None:
    """FR-018.1: the provider is listed live, available only when named. Since 018-P4
    (FR-018.18) it is the only one: ``weightspace`` is listed under no setting, not even one that
    still names it."""
    assert _listing(monkeypatch, "") == [("miwsp_engine", False, "0.1.0")]
    assert _listing(monkeypatch, "miwsp_engine") == [("miwsp_engine", True, "0.1.0")]
    assert _listing(monkeypatch, "weightspace,miwsp_engine") == [("miwsp_engine", True, "0.1.0")]
    assert _listing(monkeypatch, "weightspace") == [("miwsp_engine", False, "0.1.0")]


def test_weightspace_is_retired_not_registered() -> None:
    """FR-018.18's removal point in the live registry (the module the app's routes read), not in
    its source: the engine is the one registered provider and ``weightspace`` is a retired id."""
    from wsp.providers import registry

    assert sorted(registry.PROVIDERS) == ["miwsp_engine"]
    assert "weightspace" in registry.RETIRED and "weightspace" not in registry.PROVIDERS


def test_the_engine_descriptor_is_complete_and_its_version_is_the_engines(env, monkeypatch) -> None:
    """The listed operations and styles, and the version the worker stamps in every patch."""
    from wsp_worker import miwsp_engine

    client = _client(monkeypatch, "miwsp_engine")
    [engine] = [
        p
        for p in client.get("/api/v1/wsp/providers").json()["providers"]
        if p["id"] == "miwsp_engine"
    ]
    assert engine == {
        "id": "miwsp_engine",
        "version": miwsp_engine.VERSION,
        "operations": ["create", "inspect", "repair"],
        "styles": ["program", "session"],
        "available": True,
    }


@pytest.fixture()
def world(tmp_path) -> Iterator[dict[str, Any]]:
    fx = make_bundle(tmp_path / "bundle", rule="mean")
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        insert(
            conn,
            "mf_wsp_bundles",
            bundle_sha256=fx.manifest["bundle_sha256"],
            owner="miworkbench",
            bundle_id=fx.manifest["bundle_id"],
            bundle_kind="linear_probe",
            artifact_kind="probe",
            artifact_id="p",
            detector_digest=None,
            model="m",
            layer=12,
            sensitivity_value="not_sensitive",
            manifest=json.dumps(fx.manifest),
            bytes=1,
            import_status="imported",
            received_at=NOW,
            received_by="operator",
        )
        insert(
            conn,
            "mf_wsp_endpoints",
            id="ep-local",
            kind="openai_compatible",
            base_url="http://miserver-backend.miserver.svc:8000/v1",
            model="qwen",
            locality="local",
            output_cap_field="max_tokens",
            serving="miserver",
            created_at=NOW,
            created_by="operator",
        )
    body = {
        "bundle": {
            "owner": "miworkbench",
            "bundle_id": fx.manifest["bundle_id"],
            "bundle_sha256": fx.manifest["bundle_sha256"],
        },
        "kind": "repair",
        "mode": "session",
        "task": "fewer false alarms",
        "endpoint_id": "ep-local",
        "budgets": {"tokens": 1000},
        "change_limits": {},
        "observation_tools": [],
        "primary_sets": ["eval"],
        "sensitivity": "not_sensitive",
    }
    yield {"engine": engine, "body": body}
    # Each test's engine holds a pooled connection until disposed: on PostgreSQL a serial run of the
    # whole tree otherwise reaches max_connections here (12.2 round 3, finding R3-2).
    engine.dispose()


def test_validate_freezes_with_an_available_provider(env, monkeypatch, world) -> None:
    """A body naming no provider takes the default, ``miwsp_engine`` (018-P3's switch; the code
    default since 018-P4)."""
    client = _client(monkeypatch, "miwsp_engine", world["engine"])
    response = client.post("/api/v1/wsp/runs/validate", json=world["body"])
    assert response.status_code == 200, response.text
    out = response.json()
    assert out["request_sha256"] == jcs_sha256(out["request"])
    assert out["request"]["provider"] == {"id": "miwsp_engine", "version": "0.1.0"}
    assert out["approval_action"] == "wsp_session" and out["locality"] == "local"


def test_validate_freezes_the_engine_when_it_is_named_and_available(
    env, monkeypatch, world
) -> None:
    """The start path takes ``miwsp_engine`` from the body's ``provider`` (US-1) and freezes it,
    so the worker's dispatch (``spec.request.provider.id``) routes the run to the engine."""
    client = _client(monkeypatch, "miwsp_engine", world["engine"])
    body = dict(world["body"], provider={"id": "miwsp_engine"})
    response = client.post("/api/v1/wsp/runs/validate", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["request"]["provider"] == {"id": "miwsp_engine", "version": "0.1.0"}
    unavailable = _client(monkeypatch, "", world["engine"])
    refused = unavailable.post("/api/v1/wsp/runs/validate", json=body)
    assert refused.status_code == 503
    assert refused.json()["error"]["code"] == "provider_unavailable"


@pytest.mark.parametrize("route", ["/api/v1/wsp/runs/validate", "/api/v1/wsp/runs"])
@pytest.mark.parametrize("available", ["miwsp_engine", "weightspace,miwsp_engine", "weightspace"])
def test_a_new_run_naming_weightspace_is_refused_as_unknown_and_retired(
    env, monkeypatch, world, route, available
) -> None:
    """FR-018.18: a start naming the retired provider is refused ``provider_unknown`` (422) and
    the message says it was retired, even where ``WSP_PROVIDERS_AVAILABLE`` still names it (a
    stale ConfigMap); nothing is stored, and no other provider runs in its place."""
    client = _client(monkeypatch, available, world["engine"])
    body = dict(world["body"], provider={"id": "weightspace"})
    response = client.post(route, json=body)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "provider_unknown", error
    assert "'weightspace'" in error["message"] and "retired" in error["message"], error
    assert _runs(world["engine"]) == 0


def test_validate_refuses_an_unavailable_provider(env, monkeypatch, world) -> None:
    client = _client(monkeypatch, "", world["engine"])
    response = client.post("/api/v1/wsp/runs/validate", json=world["body"])
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_unavailable"


def test_validate_refuses_an_unknown_provider_without_fallback(env, monkeypatch, world) -> None:
    client = _client(monkeypatch, "miwsp_engine", world["engine"])
    body = dict(world["body"], provider={"id": "wsplab"})
    response = client.post("/api/v1/wsp/runs/validate", json=body)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "provider_unknown" and "retired" not in error["message"]


# ------------------------------------------------- R3 review fix (OR3-016-2), 2026-11-16


def _runs(engine) -> int:
    import sqlalchemy as sa

    with engine.connect() as conn:
        return int(conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_runs")).scalar_one())


@pytest.mark.parametrize(
    "unset",
    [("WSP_WORKER_URL",), ("WSP_GATEWAY_URL",), ("WSP_WORKER_URL", "WSP_GATEWAY_URL")],
)
@pytest.mark.parametrize("route", ["/api/v1/wsp/runs/validate", "/api/v1/wsp/runs"])
def test_an_enabled_provider_without_its_urls_is_refused_wsp_not_configured(
    env, monkeypatch, world, unset, route
) -> None:
    """Before the fix a start stored a ``queued`` run that never moved and said nothing. Now the
    start answers 503 ``wsp_not_configured`` naming each unset setting and its value, and writes
    no row."""
    client = _client(monkeypatch, "miwsp_engine", world["engine"], unset=unset)
    response = client.post(route, json=world["body"])
    assert response.status_code == 503, response.text
    error = response.json()["error"]
    assert error["code"] == "wsp_not_configured"
    for name in ("WSP_WORKER_URL", "WSP_GATEWAY_URL"):
        assert (name in error["message"]) == (name in unset), error
    assert (
        "http://miforge-wsp-worker-{index}.miforge-wsp-workers.miforge.svc.cluster.local:9000"
        in error["message"]
        or "WSP_WORKER_URL" not in unset
    )
    assert "miforge-config" in error["message"]
    assert _runs(world["engine"]) == 0


def test_a_configured_start_is_stored_queued(env, monkeypatch, world) -> None:
    """The positive half: the same body with both URLs set is stored (so the refusal above is the
    configuration, not the body)."""
    client = _client(monkeypatch, "miwsp_engine", world["engine"])
    response = client.post("/api/v1/wsp/runs", json=world["body"])
    assert response.status_code == 201, response.text
    assert response.json()["status"] == "queued" and _runs(world["engine"]) == 1


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"provider": "miwsp_engine"}, "provider"),
        ({"provider": ["miwsp_engine"]}, "provider"),
        ({"provider": {"id": 5}}, "provider.id"),
        ({"bundle": "wsb_x"}, "bundle"),
    ],
    ids=["provider-string", "provider-list", "provider-id-int", "bundle-string"],
)
def test_a_malformed_start_body_is_refused_by_name_never_a_500(
    env, monkeypatch, world, change, field
) -> None:
    """Accept016 finding A16-6 (2026-11-19 05:40 ET): ``"provider": "miwsp_engine"`` raised
    ``AttributeError`` in ``prepare_start`` and the live route answered ``500 internal_error``.
    A body field of the wrong shape is the caller's mistake: ``400 validation_failed`` naming it,
    and nothing stored."""
    client = _client(monkeypatch, "miwsp_engine", world["engine"])
    response = client.post("/api/v1/wsp/runs", json={**world["body"], **change})
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "validation_failed" and field in error["message"], error
    assert _runs(world["engine"]) == 0


@pytest.mark.parametrize("make", ["replay", "rerun"])
def test_a_replay_or_rerun_is_refused_wsp_not_configured(env, monkeypatch, world, make) -> None:
    from sqlalchemy.orm import Session

    from core.errors import Refusal
    from wsp import runs
    from wsp.settings import WSPSettings

    unset = WSPSettings(wsp_providers_available="miwsp_engine", wsp_worker_url="")
    with Session(world["engine"]) as session, pytest.raises(Refusal) as refused:
        if make == "replay":
            runs.create_replay(session, "wsr_x", "operator", settings=unset)
        else:
            runs.create_rerun(session, "0" * 64, "eval", "operator", settings=unset)
    assert refused.value.code == "wsp_not_configured"
    assert "WSP_WORKER_URL" in refused.value.message
