"""Purge: permanently delete ARCHIVED runs and parts  (operator decision 2026-10-20; 0023).

Every request goes through the **mounted** app (``create_app()``) over a database built by
``alembic upgrade head`` (SQLite here, PostgreSQL with ``MIFORGE_TEST_DATABASE_URL``), and
every file is a real file under the configured roots, so removing the router's include line,
the operator-only dependency, the archived check, the reference check, the file staging or
migration 0023 turns this file red.

Pinned, each by its own test:

* an archived run is removed - row, steps, checkpoint/adapter rows, scores, batches, merges -
  and its checkpoint, trainer-state and run-log directories are **gone from disk**; one
  tombstone records the rows, files and bytes, and the bytes are the bytes that were there;
* a run that is not archived is refused ``409 not_archived`` and nothing changes;
* a run something outside the purge still needs is refused ``409 purge_blocked`` naming the
  blocker - a live run resuming from it, a pending run naming it, a checkpoint whose lineage
  runs through it, a checkpoint miLLM is serving - and an ARCHIVED referrer does not block;
* an agent is refused ``403 purge_operator_only`` on every purge route, and nothing changes;
* no MCP tool reaches a purge route, and every one is exempt with the operator's reason;
* a batch purges each item independently: a blocked item is reported and the rest go;
* a part: index row, splits, archive row and its directory go; a reindex does not bring it
  back; a pack that pins it blocks;
* the tombstone table refuses UPDATE and DELETE in the database itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from core.config import reset_settings
from models.checkpoint import Checkpoint
from models.pack import Pack, PackPart
from models.run import Step
from services.parts import acquire, index
from services.parts.tables import CONTRACT_PARTS, PART_ARCHIVE, PART_SPLITS
from services.purge.tables import PURGES
from services.scoring.tables import SCORES
from tests.support import db as support
from tests.support import part_fixtures as fx
from tests.support.jobs import insert_run
from tests.support.mcp_ledger import EXPECTED_CALLS, PURGE_EXEMPT_REASON, ROUTE_EXEMPT, normalise
from workers.run_record import RUNS

AGENT = {"X-Miforge-Agent": "agent:purge-test"}
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)

PURGE_ROUTES: tuple[tuple[str, str, Any], ...] = (
    ("POST", "/api/v1/runs/purge/plan", {"ids": ["gone"]}),
    ("POST", "/api/v1/runs/purge", {"ids": ["gone"]}),
    ("POST", "/api/v1/runs/gone/purge", None),
    ("POST", "/api/v1/parts/purge/plan", {"ids": ["part-x"]}),
    ("POST", "/api/v1/parts/purge", {"ids": ["part-x"]}),
    ("POST", "/api/v1/parts/part-x/purge", None),
    ("GET", "/api/v1/purges", None),
)
PURGE_TEMPLATES = (
    ("POST", "/api/v1/runs/purge/plan"),
    ("POST", "/api/v1/runs/purge"),
    ("POST", "/api/v1/runs/{run_id}/purge"),
    ("POST", "/api/v1/parts/purge/plan"),
    ("POST", "/api/v1/parts/purge"),
    ("POST", "/api/v1/parts/{part_id}/purge"),
    ("GET", "/api/v1/purges"),
)


class Rig:
    def __init__(self, client: TestClient, engine: Any, root: Path) -> None:
        self.client = client
        self.engine = engine
        self.root = root
        self.checkpoints = root / "checkpoints"
        self.trainer_state = root / "trainer-state"
        self.run_logs = root / "runs"  # `tasks.run_log_root()`: trainer_state_root.parent/runs
        self.parts = root / "parts"

    def count(self, table: Any, *where: Any) -> int:
        with support.session_for(self.engine) as session:
            return int(
                session.execute(
                    sa.select(sa.func.count()).select_from(table).where(*where)
                ).scalar_one()
            )

    def tombstones(self) -> list[dict[str, Any]]:
        with support.session_for(self.engine) as session:
            return [dict(r) for r in session.execute(sa.select(PURGES)).mappings().all()]


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    monkeypatch.setenv("CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("PARTS_ROOT", str(tmp_path / "parts"))
    monkeypatch.setenv("TRAINER_STATE_ROOT", str(tmp_path / "trainer-state"))
    monkeypatch.delenv("CHECKPOINT_READ_ROOTS", raising=False)
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    monkeypatch.delenv("MILLM_BASE_URL", raising=False)
    reset_settings()
    engine = support.migrated_engine(tmp_path / "db")
    app = create_app()

    def one_session():  # type: ignore[no-untyped-def]
        with support.session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    (tmp_path / "parts").mkdir()
    try:
        yield Rig(TestClient(app, raise_server_exceptions=False), engine, tmp_path)
    finally:
        app.dependency_overrides.clear()
        reset_settings()


def write(path: Path, size: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return size


def seed_run(
    rig: Rig,
    run_id: str,
    *,
    status: str = "completed",
    archived: bool = True,
    steps: int = 3,
    parent: str | None = None,
    files: bool = True,
    **values: Any,
) -> int:
    """A run with rows and files. Returns the bytes written under its three directories."""
    with support.session_for(rig.engine) as session:
        insert_run(session, run_id, status=status, last_step=steps - 1, **values)
        for step in range(steps):
            session.execute(
                sa.insert(Step.__table__).values(run_id=run_id, step=step, loss=1.0 / (step + 1))
            )
        session.execute(
            sa.insert(Checkpoint.__table__).values(
                name=f"base:{run_id}:step-{steps - 1}",
                run_id=run_id,
                step=steps - 1,
                domain_state="absent",
                signal_source="corpus",
                parent=parent,
                base_model="base",
                weights_sha256="0" * 64,
                manifest_sha256="1" * 64,
                path=str(rig.checkpoints / run_id / f"step-{steps - 1:06d}"),
                created_at=NOW,
            )
        )
        session.execute(
            sa.insert(SCORES).values(
                run_id=run_id,
                step=steps - 1,
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
        if archived:
            session.execute(sa.update(RUNS).where(RUNS.c.id == run_id).values(archived_at=NOW))
    if not files:
        return 0
    size = 0
    size += write(rig.checkpoints / run_id / f"step-{steps - 1:06d}" / "model.safetensors", 4096)
    size += write(rig.checkpoints / run_id / f"step-{steps - 1:06d}" / "manifest.json", 100)
    size += write(rig.checkpoints / run_id / "adapters" / "step-000001" / "adapter.bin", 300)
    size += write(rig.trainer_state / run_id / f"step-{steps - 1:06d}" / "optimizer.pt", 2048)
    size += write(rig.run_logs / run_id / "run.jsonl", 777)
    return size


def run_gone(rig: Rig, run_id: str) -> bool:
    return (
        rig.count(RUNS, RUNS.c.id == run_id) == 0
        and rig.count(Step.__table__, Step.__table__.c.run_id == run_id) == 0
        and rig.count(Checkpoint.__table__, Checkpoint.__table__.c.run_id == run_id) == 0
        and not (rig.checkpoints / run_id).exists()
        and not (rig.trainer_state / run_id).exists()
        and not (rig.run_logs / run_id).exists()
    )


def run_intact(rig: Rig, run_id: str, *, steps: int = 3) -> bool:
    return (
        rig.count(RUNS, RUNS.c.id == run_id) == 1
        and rig.count(Step.__table__, Step.__table__.c.run_id == run_id) == steps
        and rig.count(Checkpoint.__table__, Checkpoint.__table__.c.run_id == run_id) == 1
        and (rig.checkpoints / run_id).is_dir()
        and (rig.trainer_state / run_id).is_dir()
        and (rig.run_logs / run_id / "run.jsonl").is_file()
    )


def codes(body: dict[str, Any]) -> list[str]:
    return [b["code"] for b in body["error"]["details"]["blockers"]]


# ======================================================================================
# Runs
# ======================================================================================


class TestPurgingARun:
    def test_an_archived_run_is_removed_with_its_files_and_a_tombstone(self, rig: Rig) -> None:
        size = seed_run(rig, "gone")
        seed_run(rig, "kept")
        preview = rig.client.post("/api/v1/runs/purge/plan", json={"ids": ["gone"]})
        assert preview.status_code == 200, preview.text
        planned = preview.json()["items"][0]
        assert planned["purgeable"] is True and planned["blockers"] == []
        assert planned["removes"]["bytes"] == size and planned["removes"]["files"] == 5
        assert planned["removes"]["rows"]["mf_steps"] == 3
        assert (rig.checkpoints / "gone").is_dir(), "a preview must change nothing"

        response = rig.client.post("/api/v1/runs/gone/purge", json={"reason": "test clutter"})

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["outcome"] == "purged" and body["files_left"] == []
        assert body["bytes_removed"] == size and body["files_removed"] == 5
        assert body["rows_removed"]["mf_runs"] == 1 and body["rows_removed"]["mf_steps"] == 3
        assert body["rows_removed"]["mf_checkpoints"] == 1
        assert run_gone(rig, "gone")
        assert run_intact(rig, "kept"), "the purge reached another run"
        # Nothing staged is left behind under any root.
        for root in (rig.checkpoints, rig.trainer_state, rig.run_logs):
            assert [p.name for p in root.iterdir()] == ["kept"]
        stones = rig.tombstones()
        assert len(stones) == 1
        stone = stones[0]
        assert stone["kind"] == "run" and stone["item_id"] == "gone"
        assert stone["purged_by"] == "operator" and stone["reason"] == "test clutter"
        assert stone["bytes_removed"] == size and stone["files_removed"] == 5
        assert stone["rows_removed"]["mf_steps"] == 3
        listed = rig.client.get("/api/v1/purges").json()["items"]
        assert [item["id"] for item in listed] == ["gone"]
        assert rig.client.get("/api/v1/runs/gone").status_code == 404

    @pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
    def test_a_run_that_is_not_archived_is_refused_and_untouched(
        self, rig: Rig, status, monkeypatch
    ) -> None:
        from services.purge import service

        seed_run(rig, "listed", status=status, archived=False)
        moved: list[Any] = []
        real_stage = service._stage
        monkeypatch.setattr(service, "_stage", lambda dirs: moved.append(dirs) or real_stage(dirs))

        response = rig.client.post("/api/v1/runs/listed/purge")

        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "not_archived"
        assert moved == [], "a refused purge must be refused before anything moves on disk"
        assert run_intact(rig, "listed")
        assert rig.tombstones() == []
        preview = rig.client.post("/api/v1/runs/purge/plan", json={"ids": ["listed"]}).json()
        assert preview["items"][0]["purgeable"] is False
        assert preview["items"][0]["not_archived"]["code"] == "not_archived"

    def test_the_delete_itself_refuses_a_run_that_is_not_archived(self, rig: Rig, monkeypatch):
        """The second holder: a run continued (so unarchived) after the plan was read. The
        conditional DELETE matches nothing, the purge is refused and the files go back."""
        from services.purge import service

        seed_run(rig, "raced", archived=False)
        monkeypatch.setattr(service, "_refuse_unless_purgeable", lambda plan: None)

        response = rig.client.post("/api/v1/runs/raced/purge")

        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "not_archived"
        assert run_intact(rig, "raced")
        assert rig.tombstones() == []

    def test_an_active_run_is_blocked_even_if_its_row_says_archived(self, rig: Rig) -> None:
        seed_run(rig, "live", status="running", files=False)  # archived_at set directly

        response = rig.client.post("/api/v1/runs/live/purge")

        assert response.status_code == 409
        assert codes(response.json()) == ["run_active"]
        assert rig.count(RUNS, RUNS.c.id == "live") == 1

    def test_an_unknown_run_is_a_404(self, rig: Rig) -> None:
        response = rig.client.post("/api/v1/runs/nope/purge")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "run_not_found"

    def test_a_live_run_resuming_from_it_blocks_and_names_itself(self, rig: Rig) -> None:
        seed_run(rig, "gone")
        seed_run(
            rig,
            "child",
            archived=False,
            files=False,
            definition_submitted={"model_source": {"resume_from": {"run_id": "gone", "step": 2}}},
        )

        response = rig.client.post("/api/v1/runs/gone/purge")

        assert response.status_code == 409, response.text
        body = response.json()
        assert body["error"]["code"] == "purge_blocked"
        assert codes(body) == ["run_reference"]
        assert body["error"]["details"]["blockers"][0]["run_id"] == "child"
        assert "child" in body["error"]["message"]
        assert run_intact(rig, "gone") and rig.tombstones() == []

    def test_a_pending_run_loading_its_checkpoint_by_path_blocks(self, rig: Rig) -> None:
        seed_run(rig, "gone")
        path = str(rig.checkpoints / "gone" / "step-000002")
        with support.session_for(rig.engine) as session:
            insert_run(
                session,
                "queued",
                status="pending",
                definition_resolved={"model_source": {"mode": "load", "path": path}},
            )

        response = rig.client.post("/api/v1/runs/gone/purge")

        assert response.status_code == 409
        assert codes(response.json()) == ["run_active_reference"]
        assert run_intact(rig, "gone")

    def test_a_checkpoint_descending_from_it_blocks(self, rig: Rig) -> None:
        seed_run(rig, "gone")
        seed_run(rig, "branch", archived=False, files=False, parent="base:gone:step-2")

        response = rig.client.post("/api/v1/runs/gone/purge")

        assert response.status_code == 409
        assert codes(response.json()) == ["lineage_reference"]
        assert run_intact(rig, "gone")

    def test_an_archived_referrer_does_not_block(self, rig: Rig) -> None:
        """Precision: the reference check is about LIVE referrers, not every mention."""
        seed_run(rig, "gone")
        seed_run(
            rig,
            "old-child",
            files=False,
            parent="base:gone:step-2",
            definition_submitted={"model_source": {"resume_from": {"run_id": "gone", "step": 2}}},
        )
        # A run whose id merely CONTAINS the purged id is not a reference either.
        seed_run(rig, "gone-2", archived=False, files=False, parent="base:gone-2:step-1")

        response = rig.client.post("/api/v1/runs/gone/purge")

        assert response.status_code == 200, response.text
        assert run_gone(rig, "gone")

    def test_an_active_chain_containing_it_blocks(self, rig: Rig) -> None:
        """Contributed by the chain package through `services.purge.blockers` (FR-006.46:
        the purge service never imports chain code). Pins that the registration is live."""
        from services.chains.tables import CHAINS, STAGES

        seed_run(rig, "gone")
        with support.session_for(rig.engine) as session:
            session.execute(
                sa.insert(CHAINS).values(id="ch-1", name="curriculum", status="running")
            )
            session.execute(
                sa.insert(STAGES).values(
                    chain_id="ch-1",
                    stage_index=0,
                    definition={},
                    advance_on=[],
                    run_id="gone",
                    started_at=NOW,
                )
            )

        response = rig.client.post("/api/v1/runs/gone/purge")

        assert response.status_code == 409, response.text
        assert codes(response.json()) == ["chain_active"]
        assert "curriculum" in response.json()["error"]["message"]
        assert run_intact(rig, "gone")

    def test_a_checkpoint_miLLM_is_serving_blocks(self, rig: Rig, monkeypatch) -> None:
        from training import acquisition

        seed_run(rig, "gone")
        monkeypatch.setattr(
            acquisition,
            "attachment_status",
            lambda: {"configured": True, "reachable": True, "resident_model": "base:gone:step-2"},
        )

        response = rig.client.post("/api/v1/runs/gone/purge")

        assert response.status_code == 409
        assert codes(response.json()) == ["loaded_in_millm"]
        assert run_intact(rig, "gone")

    def test_a_symlinked_run_directory_is_refused_and_its_target_kept(self, rig: Rig) -> None:
        seed_run(rig, "victim", archived=False)
        seed_run(rig, "gone", files=False)
        rig.checkpoints.mkdir(exist_ok=True)
        (rig.checkpoints / "gone").symlink_to(rig.checkpoints / "victim")

        response = rig.client.post("/api/v1/runs/gone/purge")

        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "purge_path_unsafe"
        assert run_intact(rig, "victim")
        assert rig.count(RUNS, RUNS.c.id == "gone") == 1

    def test_a_database_failure_puts_the_files_back(self, rig: Rig, monkeypatch) -> None:
        from services.purge import service

        seed_run(rig, "gone")

        def broken(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("database went away")

        monkeypatch.setattr(service, "_tombstone", broken)
        response = rig.client.post("/api/v1/runs/gone/purge")

        assert response.status_code == 500
        assert run_intact(rig, "gone")
        assert [p.name for p in rig.checkpoints.iterdir()] == ["gone"]


class TestBatch:
    def test_each_item_is_independent_and_blocked_items_are_reported(self, rig: Rig) -> None:
        seed_run(rig, "a")
        seed_run(rig, "b")
        seed_run(rig, "c")
        seed_run(rig, "listed", archived=False, files=False)
        seed_run(
            rig,
            "child",
            archived=False,
            files=False,
            definition_submitted={"resume_from": {"run_id": "b", "step": 2}},
        )

        response = rig.client.post(
            "/api/v1/runs/purge", json={"ids": ["a", "b", "listed", "c", "a"], "reason": "tidy"}
        )

        assert response.status_code == 200, response.text
        body = response.json()
        outcome = {r["id"]: r for r in body["results"]}
        assert [r["id"] for r in body["results"]] == ["a", "b", "listed", "c"]
        assert outcome["a"]["outcome"] == "purged" and outcome["c"]["outcome"] == "purged"
        assert outcome["b"]["outcome"] == "blocked" and outcome["b"]["code"] == "purge_blocked"
        assert outcome["b"]["blockers"][0]["run_id"] == "child"
        assert "child" in outcome["b"]["reason"]
        assert outcome["listed"]["outcome"] == "blocked"
        assert outcome["listed"]["code"] == "not_archived"
        assert body["purged"] == 2 and body["blocked"] == 2
        assert run_gone(rig, "a") and run_gone(rig, "c")
        assert run_intact(rig, "b")
        assert sorted(s["item_id"] for s in rig.tombstones()) == ["a", "c"]

    def test_a_selection_is_bounded(self, rig: Rig) -> None:
        assert rig.client.post("/api/v1/runs/purge", json={"ids": []}).status_code == 422
        too_many = {"ids": [f"r{i}" for i in range(201)]}
        assert rig.client.post("/api/v1/runs/purge", json=too_many).status_code == 422


# ======================================================================================
# Parts
# ======================================================================================


def import_part(rig: Rig, name: str) -> str:
    manifest, bodies = fx.miforge_set_manifest(reference=("prompt-set", name, "1"))
    directory = fx.write_export(rig.root / f"export-{name}", manifest, bodies)
    with support.session_for(rig.engine) as session:
        return str(acquire.from_dir(session, directory, parts_root=rig.parts).part_id)


def archive_part(rig: Rig, part_id: str) -> None:
    assert rig.client.post(f"/api/v1/parts/{part_id}/archive").status_code == 200


class TestPurgingAPart:
    def test_an_archived_part_is_removed_with_its_directory(self, rig: Rig) -> None:
        gone = import_part(rig, "gone-set")
        kept = import_part(rig, "kept-set")
        archive_part(rig, gone)
        directory = rig.parts / gone
        assert directory.is_dir()
        size = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())

        response = rig.client.post(f"/api/v1/parts/{gone}/purge")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["bytes_removed"] == size and body["rows_removed"]["mf_contract_parts"] == 1
        assert body["rows_removed"]["mf_part_archive"] == 1
        assert not directory.exists()
        assert rig.count(CONTRACT_PARTS, CONTRACT_PARTS.c.id == gone) == 0
        assert rig.count(PART_SPLITS, PART_SPLITS.c.part_id == gone) == 0
        assert rig.count(PART_ARCHIVE, PART_ARCHIVE.c.part_id == gone) == 0
        assert (rig.parts / kept).is_dir()
        assert [s["item_id"] for s in rig.tombstones()] == [gone]
        # The index is rebuilt from disk; a purged part must not come back.
        with support.session_for(rig.engine) as session:
            index.reindex_parts(session, root=rig.parts)
        assert rig.count(CONTRACT_PARTS, CONTRACT_PARTS.c.id == gone) == 0
        assert rig.client.get(f"/api/v1/parts/{gone}").status_code == 404

    def test_a_part_that_is_not_archived_is_refused(self, rig: Rig) -> None:
        part = import_part(rig, "listed-set")

        response = rig.client.post(f"/api/v1/parts/{part}/purge")

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "not_archived"
        assert (rig.parts / part).is_dir()

    def test_the_delete_itself_refuses_a_part_that_is_not_archived(
        self, rig: Rig, monkeypatch
    ) -> None:
        """The second holder for parts: unarchived after the plan was read. The archive row's
        DELETE matches nothing, so the purge is refused and the directory goes back."""
        from services.purge import service

        part = import_part(rig, "raced-set")
        monkeypatch.setattr(service, "_refuse_unless_purgeable", lambda plan: None)

        response = rig.client.post(f"/api/v1/parts/{part}/purge")

        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "not_archived"
        assert (rig.parts / part).is_dir()
        assert rig.count(CONTRACT_PARTS, CONTRACT_PARTS.c.id == part) == 1
        assert rig.tombstones() == []

    def test_a_pack_that_pins_it_blocks(self, rig: Rig) -> None:
        part = import_part(rig, "pinned-set")
        archive_part(rig, part)
        with support.session_for(rig.engine) as session:
            session.execute(
                sa.insert(Pack.__table__).values(id="pack-1", name="ops", version="1", valid=True)
            )
            session.execute(
                sa.insert(PackPart.__table__).values(
                    pack_id="pack-1",
                    part_kind="prompt_set",
                    source="midataworks",
                    ref_type="prompt-set",
                    ref_name="pinned-set",
                    ref_version="1",
                )
            )

        response = rig.client.post(f"/api/v1/parts/{part}/purge")

        assert response.status_code == 409
        body = response.json()
        assert codes(body) == ["pack_pins_it"]
        assert "ops@1" in body["error"]["message"]
        assert (rig.parts / part).is_dir()

    def test_a_live_run_that_used_it_blocks(self, rig: Rig) -> None:
        part = import_part(rig, "used-set")
        archive_part(rig, part)
        with support.session_for(rig.engine) as session:
            insert_run(
                session,
                "user",
                status="completed",
                definition_submitted={"parts": [{"part_id": part, "kind": "prompt_set"}]},
            )

        response = rig.client.post(f"/api/v1/parts/{part}/purge")

        assert response.status_code == 409
        assert codes(response.json()) == ["run_reference"]

    def test_the_part_batch_reports_each_item(self, rig: Rig) -> None:
        gone = import_part(rig, "gone-set")
        archive_part(rig, gone)
        listed = import_part(rig, "listed-set")

        body = rig.client.post("/api/v1/parts/purge", json={"ids": [gone, listed]}).json()

        assert [(r["id"], r["outcome"]) for r in body["results"]] == [
            (gone, "purged"),
            (listed, "blocked"),
        ]
        assert not (rig.parts / gone).exists() and (rig.parts / listed).is_dir()


def test_the_part_walk_skips_staged_and_in_flight_directories(rig: Rig) -> None:
    """A purge stages a directory as `.tmp-save-purge-*` before removing it; an ingest writes
    into `.tmp-save-*`. A reindex in that window must not index either (pre-existing gap)."""
    part = import_part(rig, "staged-set")
    (rig.parts / part).rename(rig.parts / ".tmp-save-purge-abc")
    with support.session_for(rig.engine) as session:
        report = index.reindex_parts(session, root=rig.parts)
    assert report.parts == 0


# ======================================================================================
# Operator only, and wiring
# ======================================================================================


@pytest.mark.parametrize(("method", "path", "body"), PURGE_ROUTES)
def test_an_agent_is_refused_on_every_purge_route(rig: Rig, method, path, body) -> None:
    seed_run(rig, "gone")

    response = rig.client.request(method, path, json=body, headers=AGENT)

    assert response.status_code == 403, response.text
    assert response.json()["error"]["code"] == "purge_operator_only"
    assert run_intact(rig, "gone")
    assert rig.tombstones() == []


@pytest.mark.parametrize(("method", "template"), PURGE_TEMPLATES)
def test_the_route_is_served_and_no_mcp_tool_reaches_it(method: str, template: str) -> None:
    from mcp_server.config import MCPSettings
    from mcp_server.server import build_server

    document = create_app().openapi()
    operation = document["paths"].get(template, {}).get(method.lower())
    assert operation is not None, f"{method} {template} is not served"
    assert not operation.get("x-approval-action"), "a purge is refused for agents, not gated"
    key = (method, normalise(template))
    assert ROUTE_EXEMPT.get(key) == PURGE_EXEMPT_REASON
    assert all(
        not (call[0] == method and "purge" in call[1]) for call in EXPECTED_CALLS.values()
    ), "an MCP tool calls a purge route"
    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert not any("purge" in name for name in names), sorted(names)


class TestTombstonesAreAppendOnly:
    def test_update_and_delete_are_refused_by_the_database(self, rig: Rig) -> None:
        seed_run(rig, "gone")
        assert rig.client.post("/api/v1/runs/gone/purge").status_code == 200

        for statement in (
            sa.update(PURGES).values(reason="rewritten"),
            sa.delete(PURGES),
        ):
            with (
                pytest.raises(sa.exc.DBAPIError, match="append-only"),
                support.session_for(rig.engine) as session,
            ):
                session.execute(statement)
        assert len(rig.tombstones()) == 1
        assert rig.tombstones()[0]["reason"] is None
