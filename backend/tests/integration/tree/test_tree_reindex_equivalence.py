"""Reindex equivalence for the forest  (task 3.10; ADR-016 #4; foundation 4.4).

Foundation 4.4 asserts the equivalence at the **service** layer, and says in its own
docstring what was owed here: *"When ``/api/v1`` lands, the API-level test asserts the same
equality over the HTTP bodies, and those bodies must be built from these same functions. An
endpoint that assembles its own body from the ORM rows is outside this guarantee and would
pass this test while breaking the thing it protects."*

So this file asserts it over ``GET /api/v1/tree``'s **body**: seed, snapshot, drop the
index, ``reindex`` from the manifests alone, and require the response to be **identical**.
That is what makes ADR-004's claim - *the filesystem is the source of truth and the database
is a rebuildable index* - true of the checkpoint tree screen rather than asserted about it.

**Two traps, both from foundation 4.4 and both still live here.**

1. *Construction-green.* Comparing two bodies proves nothing if both are empty or if the
   drop was a no-op, so the fixture's coverage is asserted first and the drop is asserted to
   have emptied the index.
2. *A subset comparison.* The comparison is over the whole body, not a projection of it, so
   a field added to ``TreeNode`` later is covered by this test without anyone remembering
   to add it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from core.config import reset_settings
from services.reindex import drop_index, index_snapshot, reindex
from tests.support import db as support
from tests.support import tree as forest


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch):  # type: ignore[no-untyped-def]
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    monkeypatch.setenv("CHECKPOINT_ROOT", str(checkpoints))
    reset_settings()

    engine = support.migrated_engine(tmp_path / "db")
    app = create_app()

    def one_session():  # type: ignore[no-untyped-def]
        with support.session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    client = TestClient(app, raise_server_exceptions=False)

    # Every shape the index has to carry, **through the real writers**: a root, a child, an
    # adapter bound to that root by its own manifest, a MERGED checkpoint with both
    # ancestors, a from-scratch run asserting a null domain, a nonconforming manifest with
    # the `domain` key removed, and a truncated chain. Single-case coverage is one of the
    # named reasons a suite stays green over a real bug (ADR-016).
    base = forest.root_checkpoint(checkpoints, run_id="run-base", step=40, domain="support")
    forest.branch(checkpoints, parent=base.name, run_id="run-base", step=80)
    adapter = forest.adapter(checkpoints, base=base, run_id="run-adapter", step=50)
    forest.merged_checkpoint(
        checkpoints, base=base, adapter_document=adapter, run_id="run-base", step=120
    )
    forest.root_checkpoint(checkpoints, run_id="run-scratch", step=10, domain=None)
    forest.checkpoint_without_domain_key(checkpoints)
    forest.unresolvable_parent_checkpoint(checkpoints, run_id="run-orphan", step=700)
    forest.indexed(engine, checkpoints, run_ids=("run-base", "run-adapter"))

    try:
        yield client, checkpoints, engine
    finally:
        app.dependency_overrides.clear()
        reset_settings()


def body(client: TestClient) -> dict[str, Any]:
    response = client.get("/api/v1/tree", params={"limit": 2000})
    assert response.status_code == 200, response.text
    return response.json()


class TestTheFixtureCoversEveryShapeBeforeAnythingIsCompared:
    def test_it_holds_both_kinds_a_merge_and_all_three_checkpoint_domain_states(self, rig) -> None:  # type: ignore[no-untyped-def]
        """An equivalence test over a thin fixture is a green with no content."""
        client, _, _ = rig
        payload = body(client)
        assert len(payload["nodes"]) == 7
        assert {node["kind"] for node in payload["nodes"]} == {"checkpoint", "adapter"}
        assert {
            node["domain_state"] for node in payload["nodes"] if node["kind"] == "checkpoint"
        } == {"recorded", "asserted_null", "absent"}
        merged = [node for node in payload["nodes"] if node["merged"] is not None]
        assert len(merged) == 1
        # Both ancestors, which is the shape the equivalence has to survive.
        assert merged[0]["parent"]["state"] == "ok" and merged[0]["merged"]["state"] == "ok"
        assert {node["parent"]["state"] for node in payload["nodes"] if node["parent"]} >= {
            "root",
            "ok",
            "truncated",
        }
        assert {node["run_indexed"] for node in payload["nodes"]} == {True, False}


class TestTheForestSurvivesLosingTheIndex:
    def test_the_response_is_IDENTICAL_after_a_drop_and_a_rebuild(self, rig) -> None:  # type: ignore[no-untyped-def]
        """**The structural test.** Byte-for-byte the same body, from manifests alone."""
        client, root, engine = rig
        before = body(client)

        with support.session_for(engine) as session:
            drop_index(session)
            session.commit()
        with support.session_for(engine) as session:
            # Assert the drop actually dropped. Without this the comparison below would
            # pass against a `drop_index` that did nothing.
            assert index_snapshot(session, root=root)["checkpoints"] == []

        with support.session_for(engine) as session:
            report = reindex(session, root=root)
            session.commit()
        assert (report.checkpoints, report.adapters) == (6, 1)

        after = body(client)
        assert after == before

    def test_a_dropped_index_that_is_NOT_reindexed_differs(self, rig) -> None:  # type: ignore[no-untyped-def]
        """The control, as a permanent test: the equality above has to be falsifiable.

        If an empty forest compared equal to a populated one, the assertion above would be
        green no matter what ``reindex`` did.
        """
        client, _, engine = rig
        before = body(client)
        with support.session_for(engine) as session:
            drop_index(session)
            session.commit()
        after = body(client)
        assert after != before
        assert after["nodes"] == []

    def test_the_rebuild_does_not_resurrect_a_DELETED_artifact(self, rig) -> None:  # type: ignore[no-untyped-def]
        """ADR-004 in the other direction: the filesystem is authoritative for absence too.

        Not a restatement of the equality above - that one proves a rebuild loses nothing,
        and this one proves it invents nothing.
        """
        import shutil

        client, root, engine = rig
        before = {node["name"] for node in body(client)["nodes"]}
        shutil.rmtree(root / "run-scratch")
        with support.session_for(engine) as session:
            drop_index(session)
            reindex(session, root=root)
            session.commit()
        after = {node["name"] for node in body(client)["nodes"]}
        assert len(after) == len(before) - 1
        assert not any(":run-scratch:" in name for name in after)
