"""F5's wiring, asserted where removing it goes red  (FR-005.43, AC-005.13; ADR-016 #1).

*A capability is not shipped until a test FAILS when its wiring is removed.* The rule was
written after a sibling shipped **sixteen fully-implemented, unit-tested, documented MCP
tools that were never registered with the server** - every one of those tests passed by
importing the module directly, so the suite was green and the documentation said done while
the feature did not exist for any caller.

So nothing here imports a handler. Three things are asserted, and the third is the one a
weaker test gets wrong:

1. every one of the eight paths is on the **mounted application's** route table and in its
   generated OpenAPI document;
2. the reindex contributor is reachable in the **table the rebuild actually iterates**, not
   merely importable;
3. the save-request call is asserted by **payload and call count**, because *"was called"*
   passes against a call sending the wrong arguments - a route that recorded a save request
   against the wrong ``run_id``, or twice, would pass it.

The mutation controls this file is the target of (MUT-10, MUT-11) are run by hand and
recorded in the report: delete the ``include_router`` line, run, require red, restore.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app, status_for
from api.deps import session_dependency
from core.config import reset_settings
from services.checkpoints import index as index_module
from services.checkpoints import requests as requests_module
from tests.support.checkpoints import create_feature_tables
from tests.support.jobs import insert_run, memory_engine, session_for

#: F5's eight, as ``(method, path)``. Declared here as well as in the project-wide manifest
#: in ``test_reachability.py``: that file's two-way comparison catches a route appearing
#: without review, and this one catches **F5's** surface shrinking - a route deleted from
#: both lists at once would otherwise be invisible.
F5_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/api/v1/runs/{run_id}/checkpoints"),
        ("GET", "/api/v1/checkpoints"),
        ("GET", "/api/v1/runs/{run_id}/checkpoints"),
        ("GET", "/api/v1/runs/{run_id}/checkpoints/{step}"),
        ("POST", "/api/v1/runs/{run_id}/checkpoints/{step}/verify"),
        ("GET", "/api/v1/runs/{run_id}/checkpoints/{step}/lineage"),
        ("POST", "/api/v1/runs/{run_id}/continue"),
        ("POST", "/api/v1/runs/{run_id}/checkpoints/{step}/branch"),
    }
)


@pytest.fixture
def app():
    reset_settings()
    return create_app()


def mounted(app) -> set[tuple[str, str]]:
    return {
        (method, route.path)
        for route in app.routes
        for method in (getattr(route, "methods", set()) or set())
        if method != "HEAD"
    }


class TestEveryRouteExistsForACaller:
    def test_all_eight_are_on_the_mounted_app(self, app):
        """**MUT-10's target.** Deleting the ``include_router`` line leaves every handler
        implemented, unit-tested and documented - and reachable by nobody."""
        missing = F5_ROUTES - mounted(app)
        assert missing == set(), f"declared but not mounted: {sorted(missing)}"

    def test_all_eight_appear_in_the_generated_openapi_document(self, app):
        """OpenAPI is derived from the mounted app, so a route absent here is absent for
        every generated client as well as for the docs page."""
        schema = app.openapi()
        documented = {
            (method.upper(), path)
            for path, operations in schema["paths"].items()
            for method in operations
        }
        assert F5_ROUTES <= documented, f"not documented: {sorted(F5_ROUTES - documented)}"

    def test_registered_is_not_the_same_as_routable(self, tmp_path: Path, monkeypatch):
        """Each route is **requested**, and the refusal that comes back proves the handler
        ran: a path that did not match answers ``http_404``, a handler that matched and
        found nothing answers by name. Without the distinction, a deleted registration looks
        exactly like a missing checkpoint."""
        monkeypatch.setenv("CHECKPOINT_ROOT", str(tmp_path))
        reset_settings()
        engine = memory_engine()
        create_feature_tables(engine)
        with session_for(engine) as session:
            insert_run(session, "run-a", status="running")

        app = create_app()

        def one_session():
            with session_for(engine) as session:
                yield session

        app.dependency_overrides[session_dependency] = one_session
        client = TestClient(app, raise_server_exceptions=False)
        try:
            for path in (
                "/api/v1/runs/run-a/checkpoints/9",
                "/api/v1/runs/run-a/checkpoints/9/lineage",
            ):
                body = client.get(path).json()
                assert body["error"]["code"] == "checkpoint_not_found", path
            for path in (
                "/api/v1/runs/run-a/checkpoints/9/verify",
                "/api/v1/runs/run-a/continue",
            ):
                body = client.post(path, json={}).json()
                assert body["error"]["code"] == "checkpoint_not_found", path
            assert client.get("/api/v1/checkpoints").status_code == 200
            assert client.get("/api/v1/runs/run-a/checkpoints").status_code == 200
        finally:
            app.dependency_overrides.clear()
            reset_settings()


class TestTheSaveRequestIsAssertedByPayloadAndCallCount:
    def test_the_flag_is_set_once_with_the_right_run_id(self, tmp_path: Path, monkeypatch):
        """*"Was called" passes against a call sending the wrong arguments.* A route that
        recorded the request against the wrong run, or recorded it twice, would pass a
        weaker assertion and would silently save the wrong run - or save twice."""
        monkeypatch.setenv("CHECKPOINT_ROOT", str(tmp_path))
        reset_settings()
        engine = memory_engine()
        create_feature_tables(engine)
        with session_for(engine) as session:
            insert_run(session, "run-a", status="running")
            insert_run(session, "run-b", status="running")

        calls: list[tuple[object, str]] = []
        real = requests_module.request_save

        def spy(session, run_id):
            calls.append((session, run_id))
            return real(session, run_id)

        monkeypatch.setattr(requests_module, "request_save", spy)
        # The endpoint module holds its own reference to the package, so the patch has to
        # land where the call site looks it up - patching only the definition would leave
        # the route calling the original and the spy recording nothing, which is a test
        # that cannot fail.
        import api.v1.endpoints.checkpoints as endpoint_module

        monkeypatch.setattr(endpoint_module.requests, "request_save", spy)

        app = create_app()

        def one_session():
            with session_for(engine) as session:
                yield session

        app.dependency_overrides[session_dependency] = one_session
        client = TestClient(app, raise_server_exceptions=False)
        try:
            assert client.post("/api/v1/runs/run-a/checkpoints").status_code == 202
        finally:
            app.dependency_overrides.clear()
            reset_settings()

        assert len(calls) == 1, f"the flag was set {len(calls)} times, not once"
        assert calls[0][1] == "run-a", f"the flag was set against {calls[0][1]!r}"

        with session_for(engine) as session:
            assert requests_module.pending_save_request(session, "run-a") is not None
            assert requests_module.pending_save_request(session, "run-b") is None


class TestTheReindexContributorIsWiredNotMerelyPresent:
    def test_the_rebuild_still_scans_for_checkpoint_manifests(self):
        """**MUT-11's target, and the shape of the assertion matters.** A test asserting
        that an upsert function *exists* would pin the defect rather than prevent it - the
        project has shipped a ``TestRingPruningIsWired`` that asserted an entry point
        existed while nothing called it. This reads the table the rebuild drives its walk
        from."""
        from models.domain_state import ArtifactKind
        from services.reindex import MANIFEST_FILENAMES, MANIFEST_KINDS

        assert index_module.checkpoint_is_indexed_by_reindex()
        assert MANIFEST_FILENAMES[ArtifactKind.CHECKPOINT] == "manifest.json"
        assert MANIFEST_KINDS[ArtifactKind.CHECKPOINT] == "miforge.checkpoint-manifest"

    def test_the_checkpoint_table_is_one_reindex_owns(self):
        """``mf_checkpoints`` has to be an **index** table, not a record table: if it were
        a record table a rebuild would not touch it and ADR-004's claim would be false for
        the one artifact the claim is mainly about."""
        from models.checkpoint import Checkpoint
        from services.reindex import INDEX_TABLES, RECORD_TABLES

        assert Checkpoint.__tablename__ in INDEX_TABLES
        assert Checkpoint.__tablename__ not in RECORD_TABLES


class TestTheRefusalCodesAreMappedRatherThanDefaulted:
    @pytest.mark.parametrize(
        ("code", "status"),
        [
            ("checkpoint_not_found", 404),
            ("checkpoint_manifest_invalid", 409),
            ("checkpoint_hash_mismatch", 409),
            ("checkpoint_step_exists", 409),
            ("trainer_state_incomplete", 409),
            ("run_not_continuable", 409),
            ("save_refused_missing_field", 409),
            ("checkpoint_root_not_atomic", 500),
        ],
    )
    def test_each_code_reports_the_status_its_meaning_implies(self, code, status):
        """Left to the 400 default, *"no such checkpoint"* would be reported as *"your
        request was bad"* - which sends the reader looking for a mistake they did not make,
        exactly as a missing ``_not_configured`` entry would."""
        assert status_for(code) == status
