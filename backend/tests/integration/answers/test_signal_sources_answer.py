"""``api.v1.endpoints.signal_sources``'s routes answer as the contract says (021 FR-021.13)."""

from __future__ import annotations

import pytest

from tests.integration.answers import _runs_scenario as scenario
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World

ROUTER = "api.v1.endpoints.signal_sources"


@case(ROUTER, "GET /api/v1/signal-sources")
def test_list_signal_sources(world: World) -> None:
    answered("GET /api/v1/signal-sources", world.client.get("/api/v1/signal-sources"))


@case(ROUTER, "POST /api/v1/signal-sources/preflight")
def test_preflight_signal_source(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """A corpus part imported through F10's real ingest, resolved by the corpus source."""
    part = scenario.corpus_part(world, monkeypatch)
    spec = {"source": "corpus", "parts": [{"kind": "corpus", "part_id": str(part["id"])}]}
    answered(
        "POST /api/v1/signal-sources/preflight",
        world.client.post("/api/v1/signal-sources/preflight", json={"spec": spec}),
    )


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
