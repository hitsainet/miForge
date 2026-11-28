"""``api.v1.endpoints.purge``'s routes answer as the contract says (021 FR-021.13).

Runs are inserted through the production table (``tests.support.jobs.insert_run``) with steps,
a checkpoint row and real files under the configured roots, as ``test_purge.py`` seeds them;
parts are imported through F10's real ingest path. Both are archived over the archive routes,
so the purge reads the archive the operator's own action wrote. Every route here is
operator-only: no agent header is sent. The batch and plan routes are driven over one item
that goes and one that is blocked (``not_archived``), so both of an item's shapes answer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa

from models.checkpoint import Checkpoint
from models.run import Step
from services.parts import acquire
from services.scoring.tables import SCORES
from tests.support import part_fixtures as part_fx
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World
from tests.support.jobs import insert_run

ROUTER = "api.v1.endpoints.purge"
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


def _write(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _run(world: World, run_id: str, *, archived: bool = True) -> None:
    """A completed run with steps, a checkpoint, a score and files; archived over the route."""
    with world.session() as session:
        insert_run(session, run_id, status="completed", last_step=2)
        for step in range(3):
            session.execute(
                sa.insert(Step.__table__).values(run_id=run_id, step=step, loss=1.0 / (step + 1))
            )
        session.execute(
            sa.insert(Checkpoint.__table__).values(
                name=f"base:{run_id}:step-2",
                run_id=run_id,
                step=2,
                domain_state="absent",
                signal_source="corpus",
                parent=None,
                base_model="base",
                weights_sha256="0" * 64,
                manifest_sha256="1" * 64,
                path=str(world.root / "checkpoints" / run_id / "step-000002"),
                created_at=NOW,
            )
        )
        session.execute(
            sa.insert(SCORES).values(
                run_id=run_id,
                step=2,
                kind="test",
                set_rendered="prompt-set/held-out@1",
                score=0.5,
                n=1,
                provider="fake",
                provider_kind="fake",
                provider_model="fake",
                aggregation="mean",
                decode={},
                created_at=NOW,
            )
        )
    _write(world.root / "checkpoints" / run_id / "step-000002" / "model.safetensors", 4096)
    _write(world.root / "state" / "trainer-state" / run_id / "step-000002" / "optimizer.pt", 2048)
    _write(world.root / "state" / "runs" / run_id / "run.jsonl", 777)
    if archived:
        response = world.client.post(f"/api/v1/runs/{run_id}/archive")
        assert response.status_code == 200, response.text


def _part(world: World, name: str, *, archived: bool = True) -> str:
    manifest, bodies = part_fx.miforge_set_manifest(reference=("prompt-set", name, "1"))
    directory = part_fx.write_export(world.root / f"export-{name}", manifest, bodies)
    with world.session() as session:
        part_id = str(acquire.from_dir(session, directory, parts_root=world.root / "parts").part_id)
    if archived:
        response = world.client.post(f"/api/v1/parts/{part_id}/archive")
        assert response.status_code == 200, response.text
    return part_id


@case(ROUTER, "POST /api/v1/runs/purge/plan")
def test_preview_runs(world: World) -> None:
    _run(world, "gone")
    _run(world, "listed", archived=False)
    response = world.client.post("/api/v1/runs/purge/plan", json={"ids": ["gone", "listed"]})
    body = answered("POST /api/v1/runs/purge/plan", response)
    assert len(body["items"]) == 2


@case(ROUTER, "POST /api/v1/runs/purge")
def test_purge_runs(world: World) -> None:
    _run(world, "gone")
    _run(world, "listed", archived=False)
    response = world.client.post(
        "/api/v1/runs/purge", json={"ids": ["gone", "listed"], "reason": "tidy"}
    )
    body = answered("POST /api/v1/runs/purge", response)
    assert body["purged"] == 1 and body["blocked"] == 1, body


@case(ROUTER, "POST /api/v1/runs/{run_id}/purge")
def test_purge_run(world: World) -> None:
    _run(world, "gone")
    response = world.client.post("/api/v1/runs/gone/purge", json={"reason": "test clutter"})
    answered("POST /api/v1/runs/{run_id}/purge", response)


@case(ROUTER, "POST /api/v1/parts/purge/plan")
def test_preview_parts(world: World) -> None:
    gone = _part(world, "gone-set")
    kept = _part(world, "kept-set", archived=False)
    response = world.client.post("/api/v1/parts/purge/plan", json={"ids": [gone, kept]})
    body = answered("POST /api/v1/parts/purge/plan", response)
    assert len(body["items"]) == 2


@case(ROUTER, "POST /api/v1/parts/purge")
def test_purge_parts(world: World) -> None:
    gone = _part(world, "gone-set")
    kept = _part(world, "kept-set", archived=False)
    response = world.client.post("/api/v1/parts/purge", json={"ids": [gone, kept]})
    body = answered("POST /api/v1/parts/purge", response)
    assert body["purged"] == 1 and body["blocked"] == 1, body


@case(ROUTER, "POST /api/v1/parts/{part_id}/purge")
def test_purge_part(world: World) -> None:
    gone = _part(world, "gone-set")
    answered("POST /api/v1/parts/{part_id}/purge", world.client.post(f"/api/v1/parts/{gone}/purge"))


@case(ROUTER, "GET /api/v1/purges")
def test_list_purges(world: World) -> None:
    """A run's and a part's tombstone, so the log answers rows of both kinds."""
    _run(world, "gone")
    gone = _part(world, "gone-set")
    assert world.client.post("/api/v1/runs/gone/purge").status_code == 200
    assert world.client.post(f"/api/v1/parts/{gone}/purge").status_code == 200
    body = answered("GET /api/v1/purges", world.client.get("/api/v1/purges"))
    assert body["total"] == 2, body


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
