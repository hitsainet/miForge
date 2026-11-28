"""``api.v1.endpoints.signal_batches``'s routes answer as the contract says (021 FR-021.13)."""

from __future__ import annotations

import pytest

from tests.integration.answers import _runs_scenario as scenario
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World

ROUTER = "api.v1.endpoints.signal_batches"


@case(ROUTER, "GET /api/v1/runs/{run_id}/batches")
def test_list_batches(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch from the real corpus source, recorded to the index and the run log."""
    run_id = scenario.define(world)
    scenario.corpus_batch(world, monkeypatch, run_id)
    body = answered(
        "GET /api/v1/runs/{run_id}/batches", world.client.get(f"/api/v1/runs/{run_id}/batches")
    )
    assert body["count"] == 1, body


@case(ROUTER, "GET /api/v1/runs/{run_id}/batches/{batch_sha256}")
def test_get_batch(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = scenario.define(world)
    batch = scenario.corpus_batch(world, monkeypatch, run_id)
    answered(
        "GET /api/v1/runs/{run_id}/batches/{batch_sha256}",
        world.client.get(f"/api/v1/runs/{run_id}/batches/{batch}"),
    )


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
