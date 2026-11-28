"""``api.v1.endpoints.runs_index``'s routes answer as the contract says (021 FR-021.13)."""

from __future__ import annotations

from tests.integration.answers import _runs_scenario as scenario
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World

ROUTER = "api.v1.endpoints.runs_index"


@case(ROUTER, "GET /api/v1/runs")
def test_list_runs(world: World) -> None:
    """Two defined runs, so the page carries rows."""
    scenario.define(world, "run-answers-a")
    scenario.define(world, "run-answers-b")
    body = answered("GET /api/v1/runs", world.client.get("/api/v1/runs"))
    assert body["total"] == 2


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
