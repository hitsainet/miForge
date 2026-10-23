"""Archive and unarchive, runs and parts  (operator decision 2026-10-20; migration 0022).

Every request goes through the **mounted** app (``create_app()``) over a database built by
``alembic upgrade head``, so deleting ``api_v1_router.include_router(archive.router)``, the
``visible`` filter in ``runs_index``, the route-level part filter or migration 0022 turns this
file red - not a unit test of the service with its own tables.

What is pinned, each by its own test because two guards over one line mask each other:

* archived items leave the default lists and come back with ``include_archived``;
* every read by id still answers, and says ``archived_at``;
* a pending or running run is refused, and stays listed;
* a continue unarchives (``run_state.reopen_for_continue``);
* the run list never hides a non-terminal row even when the column says archived;
* archiving deletes nothing: steps, the row's own record and files on disk are unchanged;
* a part's archive survives ``reindex_parts``, which empties and rebuilds the part index;
* an agent's call runs at once (F12 D9: a non-deleting write is ungated).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from core.config import reset_settings
from models.run import Step
from services.parts import acquire, index
from tests.support import db as support
from tests.support import part_fixtures as fx
from tests.support.jobs import insert_run
from workers.run_record import RUNS

AGENT = {"X-Miforge-Agent": "agent:archive-test"}


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, Any, Path]]:
    monkeypatch.setenv("CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("PARTS_ROOT", str(tmp_path / "parts"))
    monkeypatch.setenv("TRAINER_STATE_ROOT", str(tmp_path / "trainer-state"))
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    reset_settings()
    engine = support.migrated_engine(tmp_path / "db")
    app = create_app()

    def one_session():  # type: ignore[no-untyped-def]
        with support.session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    parts_root = tmp_path / "parts"
    parts_root.mkdir()
    try:
        yield TestClient(app, raise_server_exceptions=False), engine, parts_root
    finally:
        app.dependency_overrides.clear()
        reset_settings()


def seed(engine: Any, run_id: str, status: str, **values: Any) -> None:
    with support.session_for(engine) as session:
        insert_run(session, run_id, status=status, **values)


def listed(client: TestClient, **params: Any) -> list[str]:
    response = client.get("/api/v1/runs", params=params)
    assert response.status_code == 200, response.text
    return [row["run_id"] for row in response.json()["runs"]]


def column(engine: Any, run_id: str) -> Any:
    with support.session_for(engine) as session:
        return session.execute(
            sa.select(RUNS.c.archived_at).where(RUNS.c.id == run_id)
        ).scalar_one()


# ======================================================================================
# Runs
# ======================================================================================


class TestArchivingARun:
    def test_an_archived_run_leaves_the_default_list_and_returns_on_request(self, rig) -> None:
        client, engine, _ = rig
        seed(engine, "kept", "completed")
        seed(engine, "gone", "failed", failure_code="weights_rebuild_failed")

        response = client.post("/api/v1/runs/gone/archive")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["run_id"] == "gone"
        assert body["archived"] is True and body["changed"] is True
        assert body["archived_at"] is not None
        assert listed(client) == ["kept"]
        page = client.get("/api/v1/runs").json()
        assert page["total"] == 1 and page["include_archived"] is False
        everything = client.get("/api/v1/runs", params={"include_archived": "true"}).json()
        assert {row["run_id"] for row in everything["runs"]} == {"kept", "gone"}
        assert everything["total"] == 2
        rows = {row["run_id"]: row for row in everything["runs"]}
        assert rows["gone"]["archived_at"] == body["archived_at"]
        assert rows["kept"]["archived_at"] is None

    def test_the_run_view_still_opens_an_archived_run_and_says_so(self, rig) -> None:
        client, engine, _ = rig
        seed(engine, "gone", "failed")
        archived_at = client.post("/api/v1/runs/gone/archive").json()["archived_at"]

        response = client.get("/api/v1/runs/gone")

        assert response.status_code == 200, response.text
        assert response.json()["archived_at"] == archived_at
        assert client.get("/api/v1/runs/gone/definition").status_code != 404

    def test_unarchive_returns_it_and_both_calls_are_idempotent(self, rig) -> None:
        client, engine, _ = rig
        seed(engine, "gone", "cancelled")
        first = client.post("/api/v1/runs/gone/archive").json()
        again = client.post("/api/v1/runs/gone/archive").json()
        assert again["changed"] is False and again["archived_at"] == first["archived_at"]

        back = client.post("/api/v1/runs/gone/unarchive")

        assert back.status_code == 200, back.text
        assert back.json()["archived"] is False and back.json()["changed"] is True
        assert listed(client) == ["gone"]
        assert column(engine, "gone") is None
        assert client.post("/api/v1/runs/gone/unarchive").json()["changed"] is False

    @pytest.mark.parametrize("status", ["pending", "running"])
    def test_an_active_run_is_refused_and_stays_listed(self, rig, status: str) -> None:
        client, engine, _ = rig
        seed(engine, "live", status)

        response = client.post("/api/v1/runs/live/archive")

        assert response.status_code == 409, response.text
        error = response.json()["error"]
        assert error["code"] == "run_not_archivable"
        assert error["details"]["status"] == status
        assert column(engine, "live") is None
        assert listed(client) == ["live"]

    def test_an_unknown_run_is_refused_by_name(self, rig) -> None:
        client, _, _ = rig
        for verb in ("archive", "unarchive"):
            response = client.post(f"/api/v1/runs/nope/{verb}")
            assert response.status_code == 404, response.text
            assert response.json()["error"]["code"] == "run_not_found"

    def test_archiving_deletes_nothing(self, rig, tmp_path: Path) -> None:
        """Visibility only: the steps, the row's own record and the files on disk stay."""
        client, engine, _ = rig
        seed(engine, "gone", "completed", last_step=2)
        with support.session_for(engine) as session:
            for step in range(3):
                session.execute(
                    sa.insert(Step.__table__).values(
                        run_id="gone", step=step, loss=1.0 / (step + 1)
                    )
                )
        saved = tmp_path / "checkpoints" / "base" / "gone" / "step-000002"
        saved.mkdir(parents=True)
        (saved / "manifest.json").write_text("{}")
        state = tmp_path / "trainer-state" / "gone"
        state.mkdir(parents=True)
        (state / "optimizer.pt").write_bytes(b"x")
        with support.session_for(engine) as session:
            before = dict(
                session.execute(sa.select(RUNS).where(RUNS.c.id == "gone")).one()._mapping
            )

        assert client.post("/api/v1/runs/gone/archive").status_code == 200

        with support.session_for(engine) as session:
            after = dict(session.execute(sa.select(RUNS).where(RUNS.c.id == "gone")).one()._mapping)
            steps = session.execute(
                sa.select(sa.func.count()).select_from(Step.__table__)
            ).scalar_one()
        assert after.pop("archived_at") is not None
        before.pop("archived_at")
        assert after == before, "archiving changed a column besides archived_at"
        assert steps == 3
        assert (saved / "manifest.json").is_file()
        assert (state / "optimizer.pt").is_file()

    def test_a_continue_unarchives_the_run_it_reopens(self, rig) -> None:
        from training import run_state

        client, engine, _ = rig
        seed(engine, "again", "failed")
        assert client.post("/api/v1/runs/again/archive").status_code == 200

        with support.session_for(engine) as session:
            run_state.reopen_for_continue(session, "again")

        assert column(engine, "again") is None
        assert listed(client) == ["again"]

    def test_the_list_never_hides_an_active_row_whatever_the_column_says(self, rig) -> None:
        """The third holder of the rule, on its own: the row is set archived directly."""
        client, engine, _ = rig
        seed(engine, "live", "running")
        with support.session_for(engine) as session:
            session.execute(
                sa.update(RUNS).where(RUNS.c.id == "live").values(archived_at=sa.func.now())
            )

        assert listed(client) == ["live"]

    def test_an_agent_archives_at_once_without_an_approval(self, rig) -> None:
        """F12 D9: a non-deleting write is ungated. 200 and done, never 202 pending."""
        client, engine, _ = rig
        seed(engine, "gone", "failed")

        response = client.post("/api/v1/runs/gone/archive", headers=AGENT)

        assert response.status_code == 200, response.text
        assert "approval_id" not in response.json()
        assert column(engine, "gone") is not None
        assert client.post("/api/v1/runs/gone/unarchive", headers=AGENT).status_code == 200
        assert column(engine, "gone") is None


# ======================================================================================
# Parts
# ======================================================================================


def import_part(engine: Any, parts_root: Path, tmp_path: Path, name: str) -> str:
    manifest, bodies = fx.miforge_set_manifest(reference=("prompt-set", name, "1"))
    directory = fx.write_export(tmp_path / f"export-{name}", manifest, bodies)
    with support.session_for(engine) as session:
        return str(acquire.from_dir(session, directory, parts_root=parts_root).part_id)


def part_ids(client: TestClient, **params: Any) -> list[str]:
    response = client.get("/api/v1/parts", params=params)
    assert response.status_code == 200, response.text
    return [item["id"] for item in response.json()["items"]]


class TestArchivingAPart:
    def test_an_archived_part_leaves_the_list_and_returns_on_request(self, rig, tmp_path) -> None:
        client, engine, parts_root = rig
        kept = import_part(engine, parts_root, tmp_path, "kept-set")
        gone = import_part(engine, parts_root, tmp_path, "gone-set")

        response = client.post(f"/api/v1/parts/{gone}/archive")

        assert response.status_code == 200, response.text
        assert response.json()["archived"] is True and response.json()["changed"] is True
        assert part_ids(client) == [kept]
        assert sorted(part_ids(client, include_archived="true")) == sorted([kept, gone])
        everything = client.get("/api/v1/parts", params={"include_archived": "true"}).json()
        stamps = {item["id"]: item["archived_at"] for item in everything["items"]}
        assert stamps[gone] == response.json()["archived_at"] and stamps[kept] is None
        # The kind filter and the archive filter compose.
        assert part_ids(client, part_kind="prompt_set") == [kept]

    def test_the_part_still_answers_by_id_and_unarchive_returns_it(self, rig, tmp_path) -> None:
        client, engine, parts_root = rig
        gone = import_part(engine, parts_root, tmp_path, "gone-set")
        archived_at = client.post(f"/api/v1/parts/{gone}/archive").json()["archived_at"]

        one = client.get(f"/api/v1/parts/{gone}")
        assert one.status_code == 200, one.text
        assert one.json()["archived_at"] == archived_at

        assert client.post(f"/api/v1/parts/{gone}/archive").json()["changed"] is False
        back = client.post(f"/api/v1/parts/{gone}/unarchive")
        assert back.status_code == 200 and back.json()["archived"] is False
        assert part_ids(client) == [gone]
        assert client.get(f"/api/v1/parts/{gone}").json()["archived_at"] is None

    def test_the_archive_survives_a_reindex(self, rig, tmp_path) -> None:
        """`reindex_parts` empties and rebuilds `mf_contract_parts`; the archive must hold."""
        client, engine, parts_root = rig
        kept = import_part(engine, parts_root, tmp_path, "kept-set")
        gone = import_part(engine, parts_root, tmp_path, "gone-set")
        assert client.post(f"/api/v1/parts/{gone}/archive").status_code == 200

        with support.session_for(engine) as session:
            report = index.reindex_parts(session, root=parts_root)
        assert report.parts == 2

        assert part_ids(client) == [kept]
        assert gone in part_ids(client, include_archived="true")

    def test_an_unknown_part_is_refused_by_name_as_a_404(self, rig) -> None:
        client, _, _ = rig
        for verb in ("archive", "unarchive"):
            response = client.post(f"/api/v1/parts/part-nope/{verb}")
            assert response.status_code == 404, response.text
            assert response.json()["error"]["code"] == "part_not_found"
        assert client.get("/api/v1/parts/part-nope").status_code == 404

    def test_an_agent_archives_a_part_at_once(self, rig, tmp_path) -> None:
        client, engine, parts_root = rig
        gone = import_part(engine, parts_root, tmp_path, "gone-set")

        response = client.post(f"/api/v1/parts/{gone}/archive", headers=AGENT)

        assert response.status_code == 200, response.text
        assert part_ids(client) == []


# ======================================================================================
# Wiring
# ======================================================================================

ARCHIVE_ROUTES = (
    "/api/v1/runs/{run_id}/archive",
    "/api/v1/runs/{run_id}/unarchive",
    "/api/v1/parts/{part_id}/archive",
    "/api/v1/parts/{part_id}/unarchive",
)


@pytest.mark.parametrize("path", ARCHIVE_ROUTES)
def test_the_route_is_in_the_served_document_and_ungated(path: str) -> None:
    document = create_app().openapi()
    assert "post" in document["paths"].get(path, {}), f"{path} is not served"
    assert not document["paths"][path]["post"].get("x-approval-action"), path
