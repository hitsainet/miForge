"""``api.v1.endpoints.runs_acquisition``'s routes answer as the contract says (021 FR-021.13)."""

from __future__ import annotations

import pytest

from tests.integration.answers import _runs_scenario as scenario
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import AGENT, World, submit_body

ROUTER = "api.v1.endpoints.runs_acquisition"


@case(ROUTER, "GET /api/v1/models/sources")
def test_model_sources(world: World) -> None:
    answered("GET /api/v1/models/sources", world.client.get("/api/v1/models/sources"))


@case(ROUTER, "GET /api/v1/runs/{run_id}/steps")
def test_list_steps(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two steps written by F1's step writer, each citing a real batch."""
    run_id = scenario.define(world)
    scenario.take_steps(world, run_id, batch=scenario.corpus_batch(world, monkeypatch, run_id))
    body = answered(
        "GET /api/v1/runs/{run_id}/steps", world.client.get(f"/api/v1/runs/{run_id}/steps")
    )
    assert body["count"] == 2


@case(ROUTER, "POST /api/v1/runs")
def test_submit_run(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """As an agent (the gate's pending 202), then as the operator (the submit's own 202)."""
    scenario.production_shaped_preflight(monkeypatch)
    key = "POST /api/v1/runs"
    answered(
        key, world.client.post("/api/v1/runs", json=submit_body(), headers=AGENT), pending=True
    )
    answered(key, world.client.post("/api/v1/runs", json=submit_body()))
    assert world.enqueued == ["run-f1-submit"]


@case(ROUTER, "POST /api/v1/runs/{run_id}/cancel")
def test_cancel_run(world: World) -> None:
    run_id = scenario.define(world)
    answered(
        "POST /api/v1/runs/{run_id}/cancel", world.client.post(f"/api/v1/runs/{run_id}/cancel")
    )


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
