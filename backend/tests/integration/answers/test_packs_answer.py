"""``api.v1.endpoints.packs``'s routes answer as the contract says (021 FR-021.13).

The pack is committed through the real writer (``tests.support.pack_fixtures``) and its three
required slots resolve against parts imported through F10's real ingest path, so the registry
row, the validity block and the lineage carry resolved parts rather than empty slots.
"""

from __future__ import annotations

import json

from packs import transfer
from tests.support import pack_fixtures as fx
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import AGENT, World

ROUTER = "api.v1.endpoints.packs"

#: The parts the three slots of ``fx.valid_slots()`` pin, as ``(set kind, reference)``.
_PARTS = (
    ("prompt_set", ("prompt-set", "support-tickets", "3")),
    ("test_set", ("test-set", "notary-test", "1")),
    ("retention_set", ("retention-set", "general", "2")),
)


def _pack(world: World) -> None:
    """``notary-md@1``: valid, every required slot resolved to an imported part."""
    with world.session() as session:
        for kind, reference in _PARTS:
            fx.import_contract_part(
                session,
                parts_root=world.root / "parts",
                export_root=world.root / f"export-{kind}",
                set_kind=kind,
                reference=reference,
            )
        fx.commit(session, fx.definition(slots=fx.valid_slots()))


@case(ROUTER, "GET /api/v1/packs")
def test_list_packs(world: World) -> None:
    _pack(world)
    body = answered("GET /api/v1/packs", world.client.get("/api/v1/packs"))
    assert body["packs"]


@case(ROUTER, "POST /api/v1/packs")
def test_create_pack(world: World) -> None:
    response = world.client.post(
        "/api/v1/packs",
        json={
            "name": "notary-md",
            "version": "1",
            "domain": "maryland notary language",
            "slots": fx.valid_slots(),
        },
    )
    answered("POST /api/v1/packs", response)


@case(ROUTER, "GET /api/v1/packs/{pack_id}/versions/{version}")
def test_get_pack_version(world: World) -> None:
    _pack(world)
    answered(
        "GET /api/v1/packs/{pack_id}/versions/{version}",
        world.client.get("/api/v1/packs/notary-md/versions/1"),
    )


@case(ROUTER, "GET /api/v1/packs/{pack_id}/versions/{version}/validity")
def test_get_validity(world: World) -> None:
    """The valid pack: the shape that also carries ``next_steps``."""
    _pack(world)
    body = answered(
        "GET /api/v1/packs/{pack_id}/versions/{version}/validity",
        world.client.get("/api/v1/packs/notary-md/versions/1/validity"),
    )
    assert body["state"] == "valid", body


@case(ROUTER, "GET /api/v1/packs/{pack_id}/versions/{version}/lineage")
def test_get_lineage(world: World) -> None:
    _pack(world)
    body = answered(
        "GET /api/v1/packs/{pack_id}/versions/{version}/lineage",
        world.client.get("/api/v1/packs/notary-md/versions/1/lineage"),
    )
    assert body["slots"]


@case(ROUTER, "GET /api/v1/packs/{pack_id}/versions/{version}/export")
def test_export_pack(world: World) -> None:
    _pack(world)
    answered(
        "GET /api/v1/packs/{pack_id}/versions/{version}/export",
        world.client.get("/api/v1/packs/notary-md/versions/1/export"),
    )


@case(ROUTER, "POST /api/v1/packs/import")
def test_import_pack(world: World) -> None:
    """A document written by the real exporter for a pack this installation never held."""
    document = json.loads(transfer.export_bytes(fx.definition(slots=fx.valid_slots())))
    answered("POST /api/v1/packs/import", world.client.post("/api/v1/packs/import", json=document))


@case(ROUTER, "POST /api/v1/packs/{pack_id}/versions/{version}/clone")
def test_clone_pack(world: World) -> None:
    _pack(world)
    response = world.client.post(
        "/api/v1/packs/notary-md/versions/1/clone", json={"into": "notary-va"}
    )
    answered("POST /api/v1/packs/{pack_id}/versions/{version}/clone", response)


@case(ROUTER, "DELETE /api/v1/packs/{pack_id}/versions/{version}")
def test_delete_pack(world: World) -> None:
    """Gated: the agent's call waits (202 pending); the operator's deletes (204, no body)."""
    _pack(world)
    key = "DELETE /api/v1/packs/{pack_id}/versions/{version}"
    path = "/api/v1/packs/notary-md/versions/1"
    answered(key, world.client.delete(path, headers=AGENT), pending=True)
    response = world.client.delete(path)
    assert response.status_code == 204, response.text
    answered(key, response)


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
