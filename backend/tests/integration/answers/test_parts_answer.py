"""``api.v1.endpoints.parts``'s routes answer as the contract says (021 FR-021.13).

Every part is a conformant export written by ``tests.support.part_fixtures`` and taken in by
F10's real ingest path. The queued import (``202``) stubs one seam: Celery's ``apply_async`` on
``services.parts.tasks.part_import``, so ``enqueue_import`` itself (the argument assembly and
the job id it returns) runs as in production and nothing is published to a broker.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from services.parts import acquire
from tests.support import part_fixtures as part_fx
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World

ROUTER = "api.v1.endpoints.parts"


def _export(world: World, name: str = "support-tickets") -> Any:
    """A conformant export directory on disk; returns its path."""
    manifest, bodies = part_fx.miforge_set_manifest(reference=("prompt-set", name, "1"))
    return part_fx.write_export(world.root / f"export-{name}", manifest, bodies)


def _imported(world: World, name: str = "support-tickets") -> str:
    with world.session() as session:
        result = acquire.from_dir(session, _export(world, name), parts_root=world.root / "parts")
    return str(result.part_id)


@case(ROUTER, "GET /api/v1/parts")
def test_list_parts(world: World) -> None:
    _imported(world)
    body = answered("GET /api/v1/parts", world.client.get("/api/v1/parts"))
    assert body["items"]


@case(ROUTER, "GET /api/v1/parts/{part_id}")
def test_get_part(world: World) -> None:
    part_id = _imported(world)
    answered("GET /api/v1/parts/{part_id}", world.client.get(f"/api/v1/parts/{part_id}"))


@case(ROUTER, "POST /api/v1/parts/import")
def test_import_part(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both shapes: ``wait`` (200, the import's summary) and queued (202, the job id)."""
    from services.parts import tasks

    queued: list[dict[str, Any]] = []

    def apply_async(*, args: list[Any], kwargs: dict[str, Any], queue: str) -> Any:
        queued.append({"args": args, "kwargs": kwargs, "queue": queue})
        return SimpleNamespace(id="job-0123")

    monkeypatch.setattr(tasks.part_import, "apply_async", apply_async)
    key = "POST /api/v1/parts/import"
    waited = world.client.post(
        "/api/v1/parts/import",
        json={"source": "dir", "locator": str(_export(world, "waited")), "wait": True},
    )
    assert waited.status_code == 200, waited.text
    answered(key, waited)
    later = world.client.post(
        "/api/v1/parts/import", json={"source": "dir", "locator": str(_export(world, "later"))}
    )
    assert later.status_code == 202, later.text
    answered(key, later)
    assert len(queued) == 1, queued


@case(ROUTER, "POST /api/v1/parts/validate")
def test_validate_manifest(world: World) -> None:
    manifest, _bodies = part_fx.miforge_set_manifest()
    response = world.client.post("/api/v1/parts/validate", json={"manifest": manifest})
    answered("POST /api/v1/parts/validate", response)


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
