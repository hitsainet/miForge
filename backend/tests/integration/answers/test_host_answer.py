"""``api.v1.endpoints.host``'s routes answer as the contract says (021 FR-021.13)."""

from __future__ import annotations

from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World

ROUTER = "api.v1.endpoints.host"


@case(ROUTER, "GET /api/v1/host/resources")
def test_host_resources(world: World) -> None:
    """The real collector on a host without an NVIDIA driver: each GPU field named, not zeroed."""
    answered("GET /api/v1/host/resources", world.client.get("/api/v1/host/resources"))


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
