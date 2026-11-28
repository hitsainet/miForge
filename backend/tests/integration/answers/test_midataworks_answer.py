"""``api.v1.endpoints.midataworks``'s routes answer as the contract says (021 FR-021.13).

miDataworks is the independently written fake (``tests.support.fake_midataworks``), reached
through the real ``MiDataworksClient`` - only its ``httpx`` transport is the fake's, threaded in
where the route builds the client. ``GET /midataworks/available-parts`` had no HTTP-level test
before this one: here it lists one ``miforge_set`` export (and filters out a ``trl`` one).
"""

from __future__ import annotations

import functools

import pytest

from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World
from tests.support.fake_midataworks import FakeMiDataworks

ROUTER = "api.v1.endpoints.midataworks"


def _configure(monkeypatch: pytest.MonkeyPatch) -> FakeMiDataworks:
    """miDataworks configured, answering from the fake through the production client."""
    from api.v1.endpoints import midataworks as endpoints
    from clients.midataworks import MiDataworksClient
    from core.config import reset_settings

    monkeypatch.setenv("MIDATAWORKS_BASE_URL", "http://dw.invalid:8000")
    reset_settings()
    surface = FakeMiDataworks()
    monkeypatch.setattr(
        endpoints,
        "MiDataworksClient",
        functools.partial(MiDataworksClient, transport=surface.transport()),
    )
    return surface


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeMiDataworks:
    return _configure(monkeypatch)


@case(ROUTER, "GET /api/v1/midataworks/status")
def test_get_status(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unconfigured (the standalone default), then configured and probed reachable."""
    key = "GET /api/v1/midataworks/status"
    body = answered(key, world.client.get("/api/v1/midataworks/status"))
    assert body["configured"] is False
    surface = _configure(monkeypatch)
    body = answered(key, world.client.get("/api/v1/midataworks/status", params={"probe": True}))
    assert body["configured"] is True and body["reachable"] is True
    assert surface.call_count("GET", "/api/health") == 1


@case(ROUTER, "GET /api/v1/midataworks/available-parts")
def test_available_parts(world: World, fake: FakeMiDataworks) -> None:
    fake.add_export("exp-1", files={"data.jsonl": b"{}\n"}, miforge_set_kind="prompt_set")
    fake.add_export("exp-2", files={"train.jsonl": b"{}\n"}, target="trl", miforge_set_kind=None)
    body = answered(
        "GET /api/v1/midataworks/available-parts",
        world.client.get("/api/v1/midataworks/available-parts"),
    )
    assert [row["export_id"] for row in body["items"]] == ["exp-1"]


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
