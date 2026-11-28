"""``api.v1.endpoints.runs_loop``'s routes answer as the contract says (021 FR-021.13)."""

from __future__ import annotations

import pytest

from tests.integration.answers import _runs_scenario as scenario
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import AGENT, World, definition

ROUTER = "api.v1.endpoints.runs_loop"


@case(ROUTER, "GET /api/v1/refresh-modes")
def test_list_refresh_modes(world: World) -> None:
    answered("GET /api/v1/refresh-modes", world.client.get("/api/v1/refresh-modes"))


@case(ROUTER, "GET /api/v1/stop-conditions")
def test_list_stop_conditions(world: World) -> None:
    answered("GET /api/v1/stop-conditions", world.client.get("/api/v1/stop-conditions"))


@case(ROUTER, "POST /api/v1/runs/validate")
def test_validate_definition(world: World) -> None:
    body = answered(
        "POST /api/v1/runs/validate", world.client.post("/api/v1/runs/validate", json=definition())
    )
    assert body["ok"] is True, body


@case(ROUTER, "POST /api/v1/runs/define")
def test_define_run(world: World) -> None:
    """As an agent (the gate's pending 202), then as the operator (the run accepted)."""
    key = "POST /api/v1/runs/define"
    answered(
        key,
        world.client.post("/api/v1/runs/define", json=definition("run-a"), headers=AGENT),
        pending=True,
    )
    answered(key, world.client.post("/api/v1/runs/define", json=definition("run-o")))
    assert world.enqueued == ["run-o"]


@case(ROUTER, "GET /api/v1/runs/{run_id}/definition")
def test_read_run_definition(world: World) -> None:
    run_id = scenario.define(world)
    answered(
        "GET /api/v1/runs/{run_id}/definition",
        world.client.get(f"/api/v1/runs/{run_id}/definition"),
    )


@case(ROUTER, "GET /api/v1/runs/{run_id}/stop-conditions")
def test_read_stop_conditions(world: World) -> None:
    """A defined run two steps in: its step-count condition evaluated live."""
    run_id = scenario.define(world)
    scenario.take_steps(world, run_id)
    body = answered(
        "GET /api/v1/runs/{run_id}/stop-conditions",
        world.client.get(f"/api/v1/runs/{run_id}/stop-conditions"),
    )
    assert body["conditions"], body


@case(ROUTER, "GET /api/v1/runs/{run_id}/boundaries")
def test_list_boundaries(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two boundary blocks written by F6's boundary into the run's own log."""
    run_id = scenario.define(world)
    scenario.take_steps(world, run_id, batch=scenario.corpus_batch(world, monkeypatch, run_id))
    body = answered(
        "GET /api/v1/runs/{run_id}/boundaries",
        world.client.get(f"/api/v1/runs/{run_id}/boundaries"),
    )
    assert body["count"] == 2


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
