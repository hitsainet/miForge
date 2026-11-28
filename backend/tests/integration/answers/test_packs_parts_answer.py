"""``api.v1.endpoints.packs_parts``'s routes answer as the contract says (021 FR-021.13)."""

from __future__ import annotations

from tests.support import pack_fixtures as fx
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World

ROUTER = "api.v1.endpoints.packs_parts"


@case(ROUTER, "GET /api/v1/pack-part-refs/{part_id}")
def test_pinnable_reference(world: World) -> None:
    """Both of the ref's sources: a miDataworks contract part and an external part."""
    with world.session() as session:
        contract_id = fx.import_contract_part(
            session, parts_root=world.root / "parts", export_root=world.root / "export-contract"
        )
        external_id, _provenance = fx.import_external_part(
            session, parts_root=world.root / "parts", export_root=world.root / "export-external"
        )
    key = "GET /api/v1/pack-part-refs/{part_id}"
    contract = answered(key, world.client.get(f"/api/v1/pack-part-refs/{contract_id}"))
    external = answered(key, world.client.get(f"/api/v1/pack-part-refs/{external_id}"))
    assert {contract["source"], external["source"]} == {"midataworks", "external"}


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
