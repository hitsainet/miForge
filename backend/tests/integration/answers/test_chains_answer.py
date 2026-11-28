"""``api.v1.endpoints.chains``'s routes answer as the contract says (021 FR-021.13)."""

from __future__ import annotations

from typing import Any

import pytest

from tests.integration.answers import _runs_scenario as scenario
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import AGENT, World, chain_spec

ROUTER = "api.v1.endpoints.chains"


@pytest.fixture(autouse=True)
def _production_acquire() -> Any:
    """Production's ``lease.acquire``, captured before ``world`` (autouse runs first)."""
    from workers import lease as lease_module

    assert lease_module.acquire.__module__ == "workers.lease", lease_module.acquire
    return lease_module.acquire


@pytest.fixture(autouse=True)
def _real_lease(_production_acquire: Any, world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario.real_lease(world, monkeypatch, _production_acquire)


def _chain(world: World) -> dict[str, Any]:
    """The operator's create: stage 0 leased and enqueued through the chains service."""
    response = world.client.post("/api/v1/chains", json=chain_spec())
    assert response.status_code == 202, response.text
    return dict(response.json())


@case(ROUTER, "POST /api/v1/chains")
def test_create_chain(world: World) -> None:
    """As an agent (the gate's pending 202), then as the operator (the chain accepted)."""
    key = "POST /api/v1/chains"
    answered(
        key, world.client.post("/api/v1/chains", json=chain_spec(), headers=AGENT), pending=True
    )
    answered(key, world.client.post("/api/v1/chains", json=chain_spec()))
    assert len(world.enqueued) == 1


@case(ROUTER, "GET /api/v1/chains")
def test_list_chains(world: World) -> None:
    _chain(world)
    body = answered("GET /api/v1/chains", world.client.get("/api/v1/chains"))
    assert len(body["chains"]) == 1, body


@case(ROUTER, "GET /api/v1/chains/{chain_id}")
def test_read_chain(world: World) -> None:
    chain_id = _chain(world)["chain_id"]
    answered("GET /api/v1/chains/{chain_id}", world.client.get(f"/api/v1/chains/{chain_id}"))


@case(ROUTER, "POST /api/v1/chains/{chain_id}/stop")
def test_stop_chain(world: World) -> None:
    chain_id = _chain(world)["chain_id"]
    answered(
        "POST /api/v1/chains/{chain_id}/stop",
        world.client.post(f"/api/v1/chains/{chain_id}/stop", json={"requested_by": "operator"}),
    )


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
