"""``api.v1.endpoints.runs``'s routes answer as the contract says (021 FR-021.13)."""

from __future__ import annotations

import pytest

from tests.integration.answers import _runs_scenario as scenario
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World

ROUTER = "api.v1.endpoints.runs"


@case(ROUTER, "GET /api/v1/runs/{run_id}")
def test_get_run(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """A defined run with a batch, two steps and two boundaries: the loop block is filled."""
    run_id = scenario.define(world)
    scenario.take_steps(world, run_id, batch=scenario.corpus_batch(world, monkeypatch, run_id))
    answered("GET /api/v1/runs/{run_id}", world.client.get(f"/api/v1/runs/{run_id}"))


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
