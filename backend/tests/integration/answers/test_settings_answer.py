"""``api.v1.endpoints.settings``'s routes answer as the contract says (021 FR-021.13).

``PUT`` and ``DELETE /settings/{key}`` are gated ``config_write``: each is driven as an agent
(the gate's pending ``202``) and as the operator (the write's own ``200``).
"""

from __future__ import annotations

from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import AGENT, World

ROUTER = "api.v1.endpoints.settings"


def _override(world: World, key: str, value: object) -> None:
    response = world.client.put(f"/api/v1/settings/{key}", json={"value": value, "reason": "why"})
    assert response.status_code == 200, response.text


@case(ROUTER, "GET /api/v1/settings")
def test_read_settings(world: World) -> None:
    """With one override in force, so a row carries ``updated_at`` and its ``last_change``."""
    _override(world, "checkpoint_lifecycle_keep_last_n", 3)
    body = answered("GET /api/v1/settings", world.client.get("/api/v1/settings"))
    assert body["settings"]["checkpoint_lifecycle_keep_last_n"]["last_change"] is not None


@case(ROUTER, "GET /api/v1/settings/history")
def test_read_history(world: World) -> None:
    """A set and a reset, so both actions' rows are in the answer."""
    _override(world, "log_level", "DEBUG")
    assert world.client.delete("/api/v1/settings/log_level").status_code == 200
    body = answered("GET /api/v1/settings/history", world.client.get("/api/v1/settings/history"))
    assert body["count"] == 2


@case(ROUTER, "PUT /api/v1/settings/{key}")
def test_write_setting(world: World) -> None:
    key = "PUT /api/v1/settings/{key}"
    url = "/api/v1/settings/checkpoint_lifecycle_keep_last_n"
    answered(key, world.client.put(url, json={"value": 3}, headers=AGENT), pending=True)
    answered(key, world.client.put(url, json={"value": 4, "reason": "keep more"}))


@case(ROUTER, "DELETE /api/v1/settings/{key}")
def test_reset_setting(world: World) -> None:
    key = "DELETE /api/v1/settings/{key}"
    _override(world, "log_level", "DEBUG")
    url = "/api/v1/settings/log_level"
    answered(key, world.client.delete(url, headers=AGENT), pending=True)
    answered(key, world.client.delete(url, params={"reason": "back to the default"}))


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
