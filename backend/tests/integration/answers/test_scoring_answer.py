"""``api.v1.endpoints.scoring``'s routes answer as the contract says (021 FR-021.13)."""

from __future__ import annotations

import pytest

from tests.integration.answers import _runs_scenario as scenario
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import AGENT, World

ROUTER = "api.v1.endpoints.scoring"


@case(ROUTER, "GET /api/v1/providers")
def test_list_providers(world: World) -> None:
    answered("GET /api/v1/providers", world.client.get("/api/v1/providers"))


@case(ROUTER, "POST /api/v1/providers/{name}/check")
def test_check_provider(world: World) -> None:
    """The in-process row bound to the shipped fake model handle (the worker's handle; the API
    process holds no parameters), as ``test_scoring_endpoints.py`` binds it."""
    from providers import registry
    from providers.fakes import FakeModelHandle

    registry.bind_model_handle(FakeModelHandle(answers="hello there"))
    try:
        response = world.client.post("/api/v1/providers/in_process/check", json={"prompt": "p"})
    finally:
        registry.bind_model_handle(None)
    answered("POST /api/v1/providers/{name}/check", response)


@case(ROUTER, "POST /api/v1/runs/{run_id}/evaluate")
def test_evaluate_now(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """As an agent (the gate's pending 202), then as the operator (the evaluation's 200)."""
    key = "POST /api/v1/runs/{run_id}/evaluate"
    operator = scenario.evaluated(world, monkeypatch)
    agent = world.client.post(
        "/api/v1/runs/run-scored/evaluate", json={"kinds": ["test"]}, headers=AGENT
    )
    answered(key, agent, pending=True)
    answered(key, operator)


@case(ROUTER, "GET /api/v1/runs/{run_id}/scores")
def test_run_scores(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    assert scenario.evaluated(world, monkeypatch).status_code == 200
    body = answered(
        "GET /api/v1/runs/{run_id}/scores", world.client.get("/api/v1/runs/run-scored/scores")
    )
    assert body["scores"], body


@case(ROUTER, "GET /api/v1/runs/{run_id}/scores/{step}")
def test_run_scores_at_step(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    assert scenario.evaluated(world, monkeypatch).status_code == 200
    body = answered(
        "GET /api/v1/runs/{run_id}/scores/{step}",
        world.client.get("/api/v1/runs/run-scored/scores/4"),
    )
    assert body["scores"][0]["items"], body


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
