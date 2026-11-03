"""Reindex equivalence for checkpoints  (AC-005.9; ADR-004, foundation 4.4).

**The test that makes ADR-004 a claim with teeth.** Seed saves, capture every F5 response,
drop the index, rebuild from manifests alone, and assert the responses are **identical** -
including the absence of any surrogate id that would change across the rebuild.

Responses rather than rows, deliberately. Comparing rows would pass against an endpoint that
built its body a second way from the ORM object; comparing what a caller actually receives
is what makes *"the database may be lost without losing an artifact"* true for the surface
the operator uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from core.config import reset_settings
from services.reindex import drop_index, index_snapshot, reindex
from tests.support.checkpoints import create_feature_tables, saved_checkpoint, score, write_chain
from tests.support.jobs import insert_run, memory_engine, session_for


@pytest.fixture
def seeded(tmp_path: Path, monkeypatch):
    """Three runs' worth of real checkpoints, a mounted app, and an indexed database."""
    root = tmp_path / "checkpoints"
    root.mkdir()
    monkeypatch.setenv("CHECKPOINT_ROOT", str(root))
    reset_settings()

    engine = memory_engine()
    create_feature_tables(engine)
    with session_for(engine) as session:
        for run in ("run-a", "run-b", "run-merge"):
            insert_run(session, run, status="running")

    write_chain(root, run_id="run-a", steps=(100, 200, 300))
    saved_checkpoint(root, run_id="run-b", step=0, domain=None, optimizer_state="fresh")
    saved_checkpoint(
        root,
        run_id="run-merge",
        step=4,
        parent="LiquidAI/LFM2-1.2B:run-a:step-300",
        test=score(),
        retention=score("retention", value=0.7, floor=0.65, provider=None),
        merged_adapter={"name": "x:lora:step-800", "step": 800, "manifest_sha256": "f" * 64},
    )

    with session_for(engine) as session:
        reindex(session, root=root)

    app = create_app()

    def one_session():
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client, engine, root
    finally:
        app.dependency_overrides.clear()
        reset_settings()


#: Every F5 read the equivalence assertion covers. Enumerated in one place so adding a route
#: and forgetting to add it here is visible - a snapshot that silently covered a subset is
#: the construction-green failure this project keeps finding.
READ_PATHS = (
    "/api/v1/checkpoints",
    "/api/v1/runs/run-a/checkpoints",
    "/api/v1/runs/run-a/checkpoints/300",
    "/api/v1/runs/run-a/checkpoints/300/lineage",
    "/api/v1/runs/run-b/checkpoints/0",
    "/api/v1/runs/run-merge/checkpoints/4",
    "/api/v1/runs/run-merge/checkpoints/4/lineage",
)


def snapshot(client: TestClient) -> dict[str, object]:
    return {path: client.get(path).json() for path in READ_PATHS}


class TestTheIndexMayBeLostWithoutLosingAnAnswer:
    def test_every_response_is_identical_after_a_drop_and_rebuild(self, seeded):
        client, engine, root = seeded
        before = snapshot(client)

        with session_for(engine) as session:
            drop_index(session)
        with session_for(engine) as session:
            assert (
                index_snapshot(session, root=root)["checkpoints"] == []
            ), "the index was not dropped"
        with session_for(engine) as session:
            report = reindex(session, root=root)

        assert report.checkpoints == 5
        assert snapshot(client) == before

    def test_the_listings_are_empty_while_the_index_is_dropped(self, seeded):
        """The negative control for the test above: if the listings answered the same with
        **no** index at all, the equivalence assertion would be comparing two reads of the
        filesystem and would prove nothing about the rebuild."""
        client, engine, _ = seeded
        assert client.get("/api/v1/checkpoints").json()["count"] > 0

        with session_for(engine) as session:
            drop_index(session)
        assert client.get("/api/v1/checkpoints").json()["count"] == 0

    def test_the_disk_reads_still_answer_while_the_index_is_dropped(self, seeded):
        """The other half of ADR-004: a correctness read goes to the manifest, so it is
        unaffected by the index being gone. If **this** went empty too, the manifest would
        not be the authority the architecture says it is."""
        client, engine, _ = seeded
        with session_for(engine) as session:
            drop_index(session)

        body = client.get("/api/v1/runs/run-a/checkpoints/300").json()
        assert body["manifest"]["step"] == 300
        assert client.get("/api/v1/runs/run-a/checkpoints/300/lineage").json()["terminated"] == (
            "root"
        )

    def test_no_surrogate_id_appears_in_any_captured_response(self, seeded):
        """Ids are reassigned by a rebuild, so one anywhere in these bodies would make the
        equivalence above fail for a reason that has nothing to do with the rebuild being
        faithful - which is why the addressing key is ``(run_id, step)``."""
        client, _, _ = seeded
        for path, body in snapshot(client).items():
            assert '"id"' not in json.dumps(body), path

    def test_a_merged_checkpoints_second_edge_survives_the_rebuild(self, seeded):
        """R-02.45's two ancestors. Rebuilt from the manifest, so a merged node whose
        adapter edge was dropped by a rebuild would show here and nowhere else."""
        client, engine, root = seeded
        with session_for(engine) as session:
            drop_index(session)
        with session_for(engine) as session:
            reindex(session, root=root)

        rows = client.get("/api/v1/checkpoints").json()["checkpoints"]
        merged = [r for r in rows if r["merged_adapter"]]
        assert len(merged) == 1
        assert merged[0]["merged_adapter"]["name"] == "x:lora:step-800"

    def test_a_null_provider_survives_the_rebuild_as_null_and_not_as_absent(self, seeded):
        """``provider`` is declared and **not required**, so "present and null" and "absent"
        are two different conforming states. A rebuild that collapsed them would lose the
        distinction silently - which is why the score is stored as JSON rather than shredded
        into columns."""
        client, engine, root = seeded
        with session_for(engine) as session:
            drop_index(session)
        with session_for(engine) as session:
            reindex(session, root=root)

        rows = {r["name"]: r for r in client.get("/api/v1/checkpoints").json()["checkpoints"]}
        merged = next(r for r in rows.values() if r["step"] == 4)
        assert "provider" not in merged["retention"], (
            "a provider F4 never recorded must not appear; writing one in would be a "
            "fabricated provenance"
        )
        assert merged["test"]["provider"] == "in-process:lfm2-1.2b"

    def test_a_checkpoint_deleted_from_disk_disappears_on_the_next_rebuild(self, seeded):
        """E-13. **The manifest wins**: the row goes, rather than the index outliving the
        artifact and serving a stale answer."""
        import shutil

        client, engine, root = seeded
        shutil.rmtree(root / "run-b")

        with session_for(engine) as session:
            reindex(session, root=root)

        names = [r["run_id"] for r in client.get("/api/v1/checkpoints").json()["checkpoints"]]
        assert "run-b" not in names
        assert client.get("/api/v1/runs/run-b/checkpoints/0").json()["error"]["code"] == (
            "checkpoint_not_found"
        )
