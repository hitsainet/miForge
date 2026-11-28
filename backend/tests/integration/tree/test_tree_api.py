"""``GET /api/v1/tree`` from the **mounted app**  (task 3.9; ADR-016 #1; foundation 5.3).

*A capability is not shipped until a test FAILS when its wiring is removed.* So nothing
here imports the handler. Every assertion goes through ``TestClient`` against
``create_app()``, and :class:`TestTheRouteIsReachableFromTheMountedApp` asserts the route
is in the **live registry** rather than that its module imports - which is the failure that
shipped sixteen fully-tested MCP tools nobody could call.

**Payload *and* call count.** A reachability test that only checks a call happened passes
against a call sending the wrong arguments (ADR-016 #1), so
:class:`TestTheFiltersReachTheQueryIntact` spies on the query function and asserts it was
reached **once**, with **the parsed filters**, and that the body is what came back.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from api.route_table import mounted_routes
from core.config import reset_settings
from tests.support import db as support
from tests.support import tree as forest


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch):  # type: ignore[no-untyped-def]
    """A mounted app over a migrated database and a real artifact root on disk."""
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
    try:
        yield client, checkpoints, engine
    finally:
        app.dependency_overrides.clear()
        reset_settings()


def seeded(root: Path, engine, *, known: tuple[str, ...] = ("run-base", "run-adapter")):  # type: ignore[no-untyped-def]
    """A forest covering every shape the screen renders, through the real writers."""
    base = forest.root_checkpoint(root, run_id="run-base", step=40, domain="support")
    child = forest.branch(root, parent=base.name, run_id="run-base", step=80)
    adapter = forest.adapter(root, base=base, run_id="run-adapter", step=50)
    merged = forest.merged_checkpoint(
        root, base=base, adapter_document=adapter, run_id="run-base", step=120
    )
    scratch = forest.root_checkpoint(root, run_id="run-scratch", step=10, domain=None)
    truncated = forest.unresolvable_parent_checkpoint(root, run_id="run-orphan", step=700)
    forest.indexed(engine, root, run_ids=known)
    return {
        "base": base.name,
        "child": child.name,
        "adapter": adapter["name"],
        "merged": merged.name,
        "scratch": scratch.name,
        "truncated": truncated.name,
    }


class TestTheRouteIsReachableFromTheMountedApp:
    """**The control the whole of ADR-016 #1 is about.**

    These assertions go red when ``api_v1_router.include_router(tree.router)`` is deleted -
    at which point the composer, the three edge resolutions and ``schemas/tree.py`` would
    all still be implemented, unit-tested and documented, and the checkpoint tree screen
    would have no data source.
    """

    def test_the_path_is_in_the_live_route_table(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, _, _ = rig
        paths = {getattr(route, "path", None) for route in mounted_routes(client.app)}
        assert "/api/v1/tree" in paths

    def test_the_route_is_in_the_generated_openapi_document(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, _, _ = rig
        assert "/api/v1/tree" in client.app.openapi()["paths"]

    def test_it_answers_200_over_the_transport(self, rig) -> None:  # type: ignore[no-untyped-def]
        """Registered is not routable. Only a request through the transport tells them
        apart."""
        client, _, _ = rig
        response = client.get("/api/v1/tree")
        assert response.status_code == 200
        assert set(response.json()) == {"nodes", "edges", "page", "index_stale"}


class TestThePayloadIsTheForest:
    def test_the_node_set_is_every_artifact_of_both_kinds(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, root, engine = rig
        names = seeded(root, engine)
        body = client.get("/api/v1/tree").json()
        assert {node["name"] for node in body["nodes"]} == set(names.values())
        kinds = {node["name"]: node["kind"] for node in body["nodes"]}
        assert kinds[names["adapter"]] == "adapter"
        assert kinds[names["base"]] == "checkpoint"

    def test_the_edge_states_are_on_the_wire_and_distinguish_root_from_truncated(
        self, rig
    ) -> None:  # type: ignore[no-untyped-def]
        client, root, engine = rig
        names = seeded(root, engine)
        nodes = {node["name"]: node for node in client.get("/api/v1/tree").json()["nodes"]}
        assert nodes[names["base"]]["parent"]["state"] == "root"
        assert nodes[names["truncated"]]["parent"]["state"] == "truncated"
        assert nodes[names["child"]]["parent"]["state"] == "ok"
        assert nodes[names["adapter"]]["base"]["state"] == "ok"

    def test_a_merged_checkpoint_arrives_with_BOTH_edges(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, root, engine = rig
        names = seeded(root, engine)
        nodes = {node["name"]: node for node in client.get("/api/v1/tree").json()["nodes"]}
        merged = nodes[names["merged"]]
        assert merged["parent"]["target"] == names["base"]
        assert merged["merged"]["target"] == names["adapter"]
        assert merged["merged_verification"] == "verified"

    def test_the_four_facts_R_02_29_names_are_all_present(self, rig) -> None:  # type: ignore[no-untyped-def]
        """Name, parent, domain, final test score - plus kind and the canonical step."""
        client, root, engine = rig
        from tests.support.checkpoints import score

        published = forest.root_checkpoint(
            root, run_id="run-scored", step=10, domain="support", test=score(value=0.77, n=250)
        )
        forest.indexed(engine, root, run_ids=("run-scored",))
        node = client.get("/api/v1/tree").json()["nodes"][0]
        assert node["name"] == published.name
        assert node["parent"] == {"target": None, "state": "root"}
        assert node["domain"] == "support"
        assert node["test"]["score"] == 0.77 and node["test"]["n"] == 250
        assert node["kind"] == "checkpoint" and node["step"] == 10

    def test_there_is_no_depth_and_no_pack_on_the_wire(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, root, engine = rig
        seeded(root, engine)
        body = client.get("/api/v1/tree").json()
        assert "depth" not in body
        for node in body["nodes"]:
            assert "depth" not in node
            assert not any("pack" in key for key in node)


class TestTheThreeLegitimateStatesAreNotRefusals:
    """A 404 or an empty page for any of these would **hide an artifact**, and R-02.29
    promises the tree across runs (FTDD section 5.3)."""

    def test_a_node_whose_run_has_no_row_is_served_and_marked(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, root, engine = rig
        published = forest.root_checkpoint(root, run_id="run-copied-in", step=20)
        forest.indexed(engine, root, run_ids=())
        body = client.get("/api/v1/tree").json()
        assert body["nodes"][0]["name"] == published.name
        assert body["nodes"][0]["run_indexed"] is False

    def test_a_truncated_chain_is_served_200(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, root, engine = rig
        forest.unresolvable_parent_checkpoint(root)
        forest.indexed(engine, root)
        response = client.get("/api/v1/tree")
        assert response.status_code == 200
        assert response.json()["nodes"][0]["parent"]["state"] == "truncated"

    def test_an_unparented_adapter_is_served_200(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, root, engine = rig
        forest.adapter(root, base=None)
        forest.indexed(engine, root)
        response = client.get("/api/v1/tree")
        assert response.status_code == 200
        assert response.json()["nodes"][0]["base"]["state"] == "unparented"


class TestTheThreeRefusals:
    @pytest.mark.parametrize("value", ["", "   ", "None", "none", "NULL", "nil", "undefined", "~"])
    def test_an_unparseable_domain_filter_is_REFUSED_and_never_widened(
        self, rig, value: str
    ) -> None:  # type: ignore[no-untyped-def]
        """**The rule, stated as a test**: a silently widened query returns more than was
        asked for and looks like success. ``NULL`` is refused rather than taken as the null
        literal *or* as a domain called NULL, because either reading is a guess."""
        client, root, engine = rig
        seeded(root, engine)
        response = client.get("/api/v1/tree", params={"domain": value})
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "invalid_filter"
        assert error["details"]["filter"] == "domain"
        # The accepted forms are named, so the message says what to do next (ADR-021).
        assert "null" in error["details"]["accepted"]
        assert "nodes" not in response.json()

    def test_an_unknown_kind_filter_is_refused_rather_than_answered_with_both(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, root, engine = rig
        seeded(root, engine)
        response = client.get("/api/v1/tree", params={"kind": "adapters"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_filter"
        assert response.json()["error"]["details"]["accepted"] == ["checkpoint", "adapter"]

    def test_a_run_id_naming_nothing_at_all_is_run_not_found_naming_both_lookups(
        self, rig
    ) -> None:  # type: ignore[no-untyped-def]
        client, root, engine = rig
        seeded(root, engine)
        response = client.get("/api/v1/tree", params={"run_id": "nobody"})
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["code"] == "run_not_found"
        assert len(error["details"]["lookups_failed"]) == 2

    def test_a_run_id_with_artifacts_but_NO_run_row_is_NOT_refused(self, rig) -> None:  # type: ignore[no-untyped-def]
        """The second lookup is the point. Refusing on the run row alone would make
        FR-009.28's whole case unreachable through the one filter an operator would use to
        look at it."""
        client, root, engine = rig
        published = forest.root_checkpoint(root, run_id="run-copied-in", step=20)
        forest.indexed(engine, root, run_ids=())
        response = client.get("/api/v1/tree", params={"run_id": "run-copied-in"})
        assert response.status_code == 200
        assert [node["name"] for node in response.json()["nodes"]] == [published.name]

    def test_an_unreadable_index_refuses_by_name_and_names_reindex_as_the_recovery(
        self, rig
    ) -> None:  # type: ignore[no-untyped-def]
        """The index table is taken away by **renaming** it, not by dropping it.

        A ``DROP TABLE`` here was the first draft and it was a defect only PostgreSQL could
        show. On SQLite every test gets its own database file, so the drop is contained; on
        the shared PostgreSQL the dropped table outlives the test, ``alembic_version`` still
        reads head, and the **next** test's ``migrated_engine`` runs ``downgrade base`` and
        dies in ``0001``'s own downgrade with *"index ix_mf_checkpoints_run_id_step does not
        exist"* - which then reads as a broken migration rather than as a test that leaked
        its schema change. Every later test in the session errors at setup.

        A rename preserves the table's indexes and CHECK constraints under their own names
        on both engines, the request still fails from a real database error rather than a
        stubbed one, and the ``finally`` puts it back even when an assertion fails.
        """
        client, root, engine = rig
        seeded(root, engine)
        import sqlalchemy as sa

        with engine.begin() as connection:
            connection.execute(sa.text("ALTER TABLE mf_checkpoints RENAME TO mf_checkpoints_gone"))
        try:
            response = client.get("/api/v1/tree")
            assert response.status_code == 503
            error = response.json()["error"]
            assert error["code"] == "index_unavailable"
            assert error["details"]["recovery"] == "reindex"
            assert "mf_checkpoints" in error["details"]["dependency"]
        finally:
            with engine.begin() as connection:
                connection.execute(
                    sa.text("ALTER TABLE mf_checkpoints_gone RENAME TO mf_checkpoints")
                )

    def test_the_renamed_index_table_really_is_put_back(self, rig) -> None:  # type: ignore[no-untyped-def]
        """The positive check that the restore above happened.

        Without it, a ``finally`` that silently did nothing would look exactly like a clean
        test - which is this project's false-measurement family, and the reason the restore
        in every mutation sweep is confirmed rather than assumed.
        """
        import sqlalchemy as sa

        client, root, engine = rig
        seeded(root, engine)
        inspector = sa.inspect(engine)
        assert "mf_checkpoints" in inspector.get_table_names()
        assert "ix_mf_checkpoints_run_id_step" in {
            index["name"] for index in inspector.get_indexes("mf_checkpoints")
        }
        assert client.get("/api/v1/tree").status_code == 200


class TestTheFilters:
    def test_run_id_filters_to_one_runs_artifacts(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, root, engine = rig
        names = seeded(root, engine)
        body = client.get("/api/v1/tree", params={"run_id": "run-adapter"}).json()
        assert {node["name"] for node in body["nodes"]} == {names["adapter"]}

    def test_a_filtered_out_edge_target_is_NOT_LOADED_rather_than_truncated(self, rig) -> None:  # type: ignore[no-untyped-def]
        """A filtering artefact must not read as a broken lineage (FTDD section 5.2). The
        adapter's base is a checkpoint of another run, so filtering to the adapter's run
        puts the target outside the answer while leaving it in the index."""
        client, root, engine = rig
        names = seeded(root, engine)
        body = client.get("/api/v1/tree", params={"run_id": "run-adapter"}).json()
        edge = body["nodes"][0]["base"]
        assert edge == {"target": names["base"], "state": "not_loaded"}

    def test_domain_null_selects_only_the_artifacts_that_ASSERT_no_domain(self, rig) -> None:  # type: ignore[no-untyped-def]
        """``asserted_null`` only. ``absent`` is a missing required key and ``not_in_kind``
        is an adapter; neither is an artifact saying "I have no domain" (FR-009.25)."""
        client, root, engine = rig
        names = seeded(root, engine)
        forest.checkpoint_without_domain_key(root)
        forest.indexed(engine, root, run_ids=("run-base",))
        body = client.get("/api/v1/tree", params={"domain": "null"}).json()
        assert {node["name"] for node in body["nodes"]} == {names["scratch"]}

    def test_a_named_domain_selects_checkpoints_and_never_borrows_an_adapters(
        self, rig
    ) -> None:  # type: ignore[no-untyped-def]
        client, root, engine = rig
        names = seeded(root, engine)
        body = client.get("/api/v1/tree", params={"domain": "support"}).json()
        selected = {node["name"] for node in body["nodes"]}
        assert names["base"] in selected
        # The adapter's base HAS domain "support". The adapter does not, and must not be
        # swept in by it (ADR-018, FPRD F-11).
        assert names["adapter"] not in selected

    def test_kind_selects_one_artifact_kind(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, root, engine = rig
        names = seeded(root, engine)
        body = client.get("/api/v1/tree", params={"kind": "adapter"}).json()
        assert {node["name"] for node in body["nodes"]} == {names["adapter"]}

    def test_pagination_is_flat_and_reports_the_full_total(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, root, engine = rig
        names = seeded(root, engine)
        body = client.get("/api/v1/tree", params={"limit": 2, "offset": 0}).json()
        assert len(body["nodes"]) == 2
        assert body["page"] == {"limit": 2, "offset": 0, "total": len(names)}


class TestTheFiltersReachTheQueryIntact:
    """**Payload and call count** (ADR-016 #1), because "was called" passes against a call
    sending the wrong arguments - and a filter dropped on the way through would answer with
    the whole forest and look like a success."""

    def test_the_query_is_reached_ONCE_with_the_parsed_filters(self, rig, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        client, root, engine = rig
        seeded(root, engine)

        from api.v1.endpoints import tree as endpoint
        from services.tree import DomainFilterKind

        calls: list[dict[str, Any]] = []
        returned: list[Any] = []
        real = endpoint.checkpoint_tree

        def spy(session, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(kwargs)
            payload = real(session, **kwargs)
            returned.append(payload)
            return payload

        monkeypatch.setattr(endpoint, "checkpoint_tree", spy)
        body = client.get(
            "/api/v1/tree",
            params={"run_id": "run-base", "domain": "support", "kind": "checkpoint", "limit": 7},
        ).json()

        assert len(calls) == 1, calls
        (sent,) = calls
        assert sent["run_id"] == "run-base"
        assert sent["kind"] == "checkpoint"
        assert sent["domain"].kind is DomainFilterKind.NAMED
        assert sent["domain"].name == "support"
        assert (sent["limit"], sent["offset"]) == (7, 0)
        # And the body really is what came back, rather than a second rendering of it: a
        # route that re-derived the payload after the call would satisfy every assertion
        # above while serving something else.
        assert body == returned[0]
        assert body["page"]["limit"] == 7


class TestTheRequestLevelLogLine:
    def test_it_carries_the_node_count_and_the_index_stale_count(self, rig, caplog) -> None:  # type: ignore[no-untyped-def]
        """FR-009.29's divergences must be visible **in aggregate** and not only per node,
        so the count rides on one structured line per request."""
        client, root, engine = rig
        names = seeded(root, engine)
        with caplog.at_level(logging.INFO, logger="miforge.api.tree"):
            client.get("/api/v1/tree")
        records = [r for r in caplog.records if r.message == "tree_served"]
        assert len(records) == 1
        assert records[0].nodes == len(names)  # type: ignore[attr-defined]
        assert records[0].index_stale == 0  # type: ignore[attr-defined]
        assert records[0].total == len(names)  # type: ignore[attr-defined]
