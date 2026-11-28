"""``api.health``'s route answers as the contract says (021 FR-021.13).

The two dependency probes are replaced at the table the route reads
(``api.health.DEPENDENCY_PROBES``, by ``monkeypatch.setitem`` as the reachability tests do),
each returning the real probe's ``{"status", "detail"}``. One sibling is configured, so the
integrations block carries both its configured and its unconfigured form. The same body
answers ``503`` when a required dependency is down.
"""

from __future__ import annotations

import pytest

from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World

ROUTER = "api.health"
OK = {"status": "ok", "detail": None}
DOWN = {"status": "unavailable", "detail": "OperationalError: connection refused"}


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch) -> None:
    from api import health as health_module
    from core.config import reset_settings

    monkeypatch.setenv("MIDATAWORKS_BASE_URL", "http://dw.invalid:8000")
    reset_settings()
    monkeypatch.setitem(health_module.DEPENDENCY_PROBES, "postgres", lambda: OK)
    monkeypatch.setitem(health_module.DEPENDENCY_PROBES, "redis", lambda: OK)


@case(ROUTER, "GET /api/health")
def test_health(world: World, probes: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from api import health as health_module

    body = answered("GET /api/health", world.client.get("/api/health"))
    assert body["status"] == "ok"
    monkeypatch.setitem(health_module.DEPENDENCY_PROBES, "postgres", lambda: DOWN)
    response = world.client.get("/api/health")
    assert response.status_code == 503, response.text
    assert answered("GET /api/health", response)["degraded_dependencies"] == ["postgres"]


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
