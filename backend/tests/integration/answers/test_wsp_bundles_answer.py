"""``api.v1.endpoints.wsp_bundles``'s routes answer as the contract says (021 FR-021.13).

The bundle is requested through the route (its Celery enqueue recorded, as
``test_bundle_import`` records it), then imported by the task's own body
(``wsp.bundles.import_bundle``) against the recording fake miWorkbench, so the list and get routes
answer an imported row with its full manifest.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.integration.wsp.test_bundle_import import Env
from tests.integration.wsp.test_bundle_import import env as env
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.fake_miworkbench import BASE
from wsp import bundles
from wsp.settings import get_wsp_settings

ROUTER = "api.v1.endpoints.wsp_bundles"


@pytest.fixture()
def api(env: Env, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Env, TestClient, list[str]]]:
    from api.app import create_app
    from api.deps import session_dependency
    from tests.support.jobs import session_for
    from wsp import tasks

    monkeypatch.setenv("MIWORKBENCH_API_URL", BASE)
    monkeypatch.setenv("WSP_BUNDLE_ROOT", str(env.settings.wsp_bundle_root))
    get_wsp_settings.cache_clear()
    queued: list[str] = []
    monkeypatch.setattr(tasks.import_bundle_task, "delay", lambda sha: queued.append(sha))
    app = create_app()

    def one_session() -> Iterator[Any]:
        with session_for(env.engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    yield env, TestClient(app, raise_server_exceptions=False), queued
    get_wsp_settings.cache_clear()


def _imported(env: Env, client: TestClient, queued: list[str]) -> str:
    """Request the import through the route, then run the queued task's body."""
    answered(
        "POST /api/v1/wsp/bundles/import",
        client.post("/api/v1/wsp/bundles/import", json=env.body()),
    )
    (sha,) = queued
    with env.factory() as session:
        outcome = bundles.import_bundle(
            session, sha, settings=env.settings, transport=env.fake.transport()
        )
        session.commit()
    assert outcome == "imported", outcome
    return sha


@case(ROUTER, "POST /api/v1/wsp/bundles/import")
def test_import_bundle(api: tuple[Env, TestClient, list[str]]) -> None:
    """The first request queues the import (``importing``); the second, once imported, is a no-op
    answering the imported row (``queued: false``)."""
    env, client, queued = api
    first = answered(
        "POST /api/v1/wsp/bundles/import",
        client.post("/api/v1/wsp/bundles/import", json=env.body()),
    )
    assert first["queued"] is True
    with env.factory() as session:
        bundles.import_bundle(
            session, queued[0], settings=env.settings, transport=env.fake.transport()
        )
        session.commit()
    again = answered(
        "POST /api/v1/wsp/bundles/import",
        client.post("/api/v1/wsp/bundles/import", json=env.body()),
    )
    assert again["queued"] is False and again["import_status"] == "imported"


@case(ROUTER, "GET /api/v1/wsp/bundles")
def test_list_bundles(api: tuple[Env, TestClient, list[str]]) -> None:
    env, client, queued = api
    _imported(env, client, queued)
    listed = answered("GET /api/v1/wsp/bundles", client.get("/api/v1/wsp/bundles"))
    assert len(listed["bundles"]) == 1


@case(ROUTER, "GET /api/v1/wsp/bundles/{bundle_sha256}")
def test_get_bundle(api: tuple[Env, TestClient, list[str]]) -> None:
    env, client, queued = api
    sha = _imported(env, client, queued)
    got = answered(
        "GET /api/v1/wsp/bundles/{bundle_sha256}", client.get(f"/api/v1/wsp/bundles/{sha}")
    )
    assert got["manifest"] is not None


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
