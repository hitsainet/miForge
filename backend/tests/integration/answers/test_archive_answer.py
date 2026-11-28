"""``api.v1.endpoints.archive``'s routes answer as the contract says (021 FR-021.13).

A run is inserted through the production table (``tests.support.jobs.insert_run``), completed,
as ``test_archive.py`` seeds it; a part is imported through F10's real ingest path. Each
unarchive follows a real archive, so it answers for an item that was archived.
"""

from __future__ import annotations

from services.parts import acquire
from tests.support import part_fixtures as part_fx
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World
from tests.support.jobs import insert_run

ROUTER = "api.v1.endpoints.archive"


def _run(world: World, run_id: str = "done") -> str:
    with world.session() as session:
        insert_run(session, run_id, status="completed", last_step=2)
    return run_id


def _part(world: World) -> str:
    manifest, bodies = part_fx.miforge_set_manifest()
    directory = part_fx.write_export(world.root / "export", manifest, bodies)
    with world.session() as session:
        return str(acquire.from_dir(session, directory, parts_root=world.root / "parts").part_id)


@case(ROUTER, "POST /api/v1/runs/{run_id}/archive")
def test_archive_run(world: World) -> None:
    run_id = _run(world)
    answered(
        "POST /api/v1/runs/{run_id}/archive", world.client.post(f"/api/v1/runs/{run_id}/archive")
    )


@case(ROUTER, "POST /api/v1/runs/{run_id}/unarchive")
def test_unarchive_run(world: World) -> None:
    run_id = _run(world)
    assert world.client.post(f"/api/v1/runs/{run_id}/archive").status_code == 200
    answered(
        "POST /api/v1/runs/{run_id}/unarchive",
        world.client.post(f"/api/v1/runs/{run_id}/unarchive"),
    )


@case(ROUTER, "POST /api/v1/parts/{part_id}/archive")
def test_archive_part(world: World) -> None:
    part_id = _part(world)
    answered(
        "POST /api/v1/parts/{part_id}/archive",
        world.client.post(f"/api/v1/parts/{part_id}/archive"),
    )


@case(ROUTER, "POST /api/v1/parts/{part_id}/unarchive")
def test_unarchive_part(world: World) -> None:
    part_id = _part(world)
    assert world.client.post(f"/api/v1/parts/{part_id}/archive").status_code == 200
    answered(
        "POST /api/v1/parts/{part_id}/unarchive",
        world.client.post(f"/api/v1/parts/{part_id}/unarchive"),
    )


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
