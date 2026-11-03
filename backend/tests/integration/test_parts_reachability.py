"""F10's capabilities exist **for a caller**  (AC-010.11, AC-010.13; ADR-016 #1).

*A capability is not shipped until a test FAILS when its wiring is removed.* The rule was
written after a sibling shipped **sixteen fully-implemented, unit-tested, documented MCP
tools that were never registered with the server** - every one of their tests passed by
importing the module directly, so the suite was green and the docs said ✅ while the feature
did not exist for any caller.

So nothing here imports an endpoint function. Every assertion is made against the **mounted
application**, over the transport, or against a **live registry**. And ``test_wiring.py``'s
sibling in this file - :class:`TestTheControlsWereRun` - records the mutations that were
actually run against these assertions, because *a structural test that has never been seen
red is not known to work.*

The grep form is the weaker one and it is not sufficient: **a grep proves a call exists, not
that anything depends on it.**
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.route_table import mounted_routes
from core.config import reset_settings

#: Every route F10 adds, as ``(method, path)``. The same declaration
#: ``backend/tests/unit/test_reachability.py`` carries for the whole surface; repeated here
#: scoped to this feature so a failure points at F10 rather than at "the API".
F010_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/api/v1/parts/import"),
        ("POST", "/api/v1/parts/validate"),
        ("GET", "/api/v1/parts"),
        ("GET", "/api/v1/parts/{part_id}"),
        ("GET", "/api/v1/midataworks/status"),
        ("GET", "/api/v1/midataworks/available-parts"),
        ("GET", "/api/v1/runs/{run_id}/review"),
    }
)


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    reset_settings()
    return create_app()


@pytest.fixture()
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def mounted(app) -> set[tuple[str, str]]:
    return {
        (method, route.path)
        for route in mounted_routes(app)
        for method in getattr(route, "methods", set())
        if method != "HEAD"
    }


class TestTheRoutesAreMountedNotMerelyDefined:
    @pytest.mark.parametrize(("method", "path"), sorted(F010_ROUTES))
    def test_the_route_is_in_the_mounted_app(self, app, method: str, path: str):
        assert (method, path) in mounted(app)

    @pytest.mark.parametrize(("method", "path"), sorted(F010_ROUTES))
    def test_the_route_appears_in_the_generated_document(self, app, method: str, path: str):
        """A route present in the table but absent from OpenAPI is one nothing can call."""
        document = app.openapi()
        assert path in document["paths"]
        assert method.lower() in document["paths"][path]

    def test_the_parts_listing_answers_over_the_transport(self, client, monkeypatch):
        """Registered-but-unroutable is red too, which only a real request can show."""
        from api import deps

        def no_database():
            from core.errors import refuse

            raise refuse("database_unavailable", "no database in this test")

        client.app.dependency_overrides[deps.session_dependency] = no_database
        response = client.get("/api/v1/parts")
        # 503 from the named refusal, NOT a 404: the route answered.
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "database_unavailable"
        client.app.dependency_overrides.clear()

    def test_an_unrouted_path_under_the_same_prefix_is_a_different_404(self, client):
        """So the test above cannot pass because everything 404s the same way."""
        response = client.get("/api/v1/parts-not-a-route")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "http_404"


class TestTheValidatorRouteCarriesThePayloadAndNotJustAStatus:
    def test_a_conformant_manifest_validates_through_the_transport(self, client):
        from tests.support import part_fixtures as fx

        document, _ = fx.miforge_set_manifest()
        response = client.post("/api/v1/parts/validate", json={"manifest": document})
        assert response.status_code == 200
        body = response.json()
        assert body["valid"] is True
        assert body["part_kind"] == "prompt_set"
        assert body["contract_reference"] == {
            "type": "prompt-set",
            "name": "support-tickets",
            "version": "3",
        }

    def test_a_non_conformant_manifest_refuses_with_the_pointer(self, client):
        """AC-010.3 over the wire. **The refusal is the product.**"""
        from tests.support import part_fixtures as fx

        document, _ = fx.miforge_set_manifest()
        document["content"]["splits"][0].pop("sha256")
        response = client.post("/api/v1/parts/validate", json={"manifest": document})
        assert response.status_code in (400, 409)
        error = response.json()["error"]
        assert error["code"] == "dataset_version_invalid"
        assert error["details"]["errors"][0]["instance_path"] == "/content/splits/0"
        assert error["details"]["errors"][0]["schema_keyword"] == "required"

    def test_a_typo_in_the_request_body_refuses_rather_than_being_ignored(self, client):
        """``extra="forbid"``. An ignored field is a silent default."""
        response = client.post("/api/v1/parts/validate", json={"manifests": {}})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_failed"


class TestUnconfiguredCapabilitiesRefuseByNameOverTheTransport:
    def test_the_listing_refuses_naming_the_setting(self, client):
        """AC-010.1. Nothing is retried and no row is written against an empty address."""
        response = client.get("/api/v1/midataworks/available-parts")
        assert response.status_code == 503
        error = response.json()["error"]
        assert error["code"] == "midataworks_not_configured"
        assert error["details"]["setting"] == "MIDATAWORKS_BASE_URL"
        assert error["details"]["capability"] == "available_parts_listing"

    def test_the_status_route_is_available_unconfigured_and_says_so(self, client):
        """**Absent is normal.** The status of an optional integration must be readable
        precisely when it is absent - that is the case an operator is diagnosing."""
        response = client.get("/api/v1/midataworks/status")
        assert response.status_code == 200
        body = response.json()
        assert body["configured"] is False
        assert body["reachable"] is None, "null, not false - see FR-010.6"
        assert set(body["capabilities_unavailable"]) == set(body["capabilities"])

    def test_health_and_the_status_route_read_one_registry(self, client):
        """Two lists would drift, and health reporting a capability the guard does not gate
        is worse than no report at all."""
        from api.health import health_payload

        status = client.get("/api/v1/midataworks/status").json()
        health = health_payload()["integrations"]["midataworks"]
        assert status["configured"] == health["configured"]
        assert sorted(status["capabilities_unavailable"]) == sorted(
            {key for key in status["capabilities"] if key in status["capabilities_unavailable"]}
        )
        assert health["code"] == status["code"]


class TestTheImportJobIsRegisteredInTheLiveCeleryRegistry:
    def test_the_task_is_registered(self):
        """**The live registry, not the module.** An unregistered task is unreachable
        however well tested its body is."""
        from services.parts.tasks import TASK_PART_IMPORT
        from workers.celery_app import celery_app

        assert TASK_PART_IMPORT in celery_app.tasks

    def test_it_routes_to_the_default_queue_through_celerys_own_router(self):
        """``default``, never ``train``: nothing here touches a GPU, and a GPU queue is for
        what contends on the card. Resolved through Celery's router so the queue asserted is
        the queue the broker uses."""
        from services.parts.tasks import TASK_PART_IMPORT
        from workers.celery_app import QUEUE_DEFAULT, queue_for

        assert queue_for(TASK_PART_IMPORT) == QUEUE_DEFAULT

    def test_the_api_module_is_what_registers_it(self):
        """The production import path. Asserted so that removing the import from the route
        module is a red test rather than a task that exists only when a test imports it."""
        import ast

        source = (
            Path(__file__).resolve().parents[2] / "src" / "api" / "v1" / "endpoints" / "parts.py"
        ).read_text("utf-8")
        imports = {
            node.module
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert "services.parts.tasks" in imports


class TestMiForgeHoldsNoReviewQueue:
    """AC-010.11, FR-010.25. **Asserted against the live router and the live metadata.**

    Not against table names. The prohibition in R-02.36 is on the **thing** - the queue, the
    decision history, the review screen - and holding a queue id is not holding a queue:
    R-02.36's own closing sentence *requires* the ``external_id`` -> decision-id map. An
    earlier wording of this criterion read *"no review table"*, which this feature's own
    design contradicts, and a test written to it would have been **red against the design on
    the day it was written** - or made to pass by renaming the tables, which changes nothing
    about where the queue lives.
    """

    def test_there_is_no_adjudication_route_in_the_live_router(self, app):
        """A decision-recording endpoint here would be the second answer to the one question
        miDataworks already answers."""
        suspicious = [
            (method, path)
            for method, path in mounted(app)
            if method in {"POST", "PUT", "PATCH", "DELETE"}
            and ("decision" in path or "adjudicat" in path or "review-item" in path)
        ]
        assert suspicious == [], suspicious

    def test_the_only_review_route_is_read_only(self, app):
        review_routes = {(m, p) for m, p in mounted(app) if "review" in p}
        assert review_routes == {("GET", "/api/v1/runs/{run_id}/review")}

    def test_there_is_no_queue_or_decision_store(self):
        """No table of items **awaiting a decision** and no decision store.

        The two binding tables are asserted **present**, so this test distinguishes *holding
        a queue* from *holding a queue id* rather than banning a prefix.
        """
        from services.parts.tables import ALL_TABLES, REVIEW_STATE_TABLES

        assert set(REVIEW_STATE_TABLES) <= set(ALL_TABLES)
        assert set(REVIEW_STATE_TABLES) == {"mf_review_bindings", "mf_review_candidate_refs"}
        forbidden = [
            name
            for name in ALL_TABLES
            if name in {"mf_review_items", "mf_review_queues", "mf_review_decisions"}
        ]
        assert forbidden == [], forbidden

    def test_the_binding_tables_hold_a_pointer_and_not_content(self):
        """What miForge stores is exactly three things: the queue id, the candidates' external
        ids, and the decision ids and decisions it was told. ``reason`` and ``decided_by`` are
        kept for display and traceability - **the authority is miDataworks** - and there is
        no column for a queue's items or a decision's history."""
        from services.parts.tables import REVIEW_CANDIDATE_REFS

        columns = set(REVIEW_CANDIDATE_REFS.columns.keys())
        assert {"external_id", "item_id", "decision_id", "decision"} <= columns
        assert "payload" not in columns, "a candidate's text would make this a queue"
        assert "prompt" not in columns
        assert "completion" not in columns

    def test_no_frontend_component_is_shipped_by_this_feature(self):
        """F9 owns the Admin UI, F7 pack authoring, F3 the selector card. **F10 owns no
        screen** - and the mockup's *"12 waiting in the current run"* is miDataworks' queue
        surfaced, not rebuilt."""
        repo = Path(__file__).resolve().parents[3]
        frontend = repo / "frontend" / "src"
        if not frontend.is_dir():
            pytest.skip("no frontend tree in this checkout")
        offenders = [
            str(path.relative_to(frontend))
            for path in frontend.rglob("*.tsx")
            if "adjudicat" in path.read_text("utf-8").lower()
        ]
        assert offenders == [], offenders


class TestTheControlsWereRun:
    """**The record of the mutations, in the repository rather than only in a report.**

    *A structural test that has never been seen red is not known to work* (foundation 10.2).
    Each line below was produced by breaking the named line, running the named test, seeing
    red, restoring, and confirming ``git diff`` clean. The table is asserted non-empty so it
    cannot quietly become a comment nobody maintains.
    """

    #: ``(what was broken, which test went red)``
    CONTROLS: tuple[tuple[str, str], ...] = (
        (
            "deleted `api_v1_router.include_router(parts.router)`",
            "test_parts_reachability.py::TestTheRoutesAreMountedNotMerelyDefined",
        ),
        (
            "deleted `api_v1_router.include_router(midataworks.router)`",
            "test_parts_reachability.py + test_reachability.py",
        ),
        (
            "deleted `api_v1_router.include_router(runs_review.router)`",
            "test_parts_reachability.py::TestTheRoutesAreMountedNotMerelyDefined",
        ),
        (
            "planted `import httpx` in services/parts/index.py",
            "test_midataworks_one_call_site.py::test_no_f10_module_but_the_client_imports_httpx",
        ),
        (
            "planted a write to midataworks-dataset-version.json in services/parts/index.py",
            "test_midataworks_one_call_site.py::TestMiForgeWritesNoDatasetVersionManifest",
        ),
        (
            "perturbed one byte of a TEMPORARY copy of the vendored schema",
            "test_schema_sync.py::test_schema_is_byte_identical_in_every_repo",
        ),
        (
            "pointed the validator at an embedded copy instead of the file path",
            "test_dataset_version_contract.py::TestTheRuntimeValidatorReadsTheVendoredFile",
        ),
        (
            "defaulted an undecided candidate to accept in verdicts.verdict_for_absent",
            "test_review_queue_verdicts.py::TestNoDecisionYet",
        ),
        (
            "dropped `caveats` from part_row_values",
            "test_parts_ingest.py::test_caveats_and_licence_classes_are_on_the_summary",
        ),
        (
            "allowed a held_out split for training in reader.splits_for",
            "test_parts_reader.py::TestTraining",
        ),
        (
            "mapped 5xx to midataworks_not_configured",
            "test_midataworks_status_mapping.py::TestUnreachableIsNotUnconfigured",
        ),
        (
            "added a retry on 401",
            "test_midataworks_failure_modes.py::test_a_401_costs_exactly_one_call",
        ),
        (
            "dropped the index without reindexing",
            "test_parts_reindex_equivalence.py::test_a_dropped_index_that_is_not_reindexed_differs",
        ),
        (
            "synthesised a contract reference from the version id",
            "test_parts_ingest.py::TestANullContractReferenceIsSurfacedNotSynthesised",
        ),
        (
            "sent `since` and `external_ids` in one read-back request",
            "test_review_queue_round_trip.py::"
            "test_the_watermark_and_the_pending_ids_are_two_separate_requests",
        ),
    )

    def test_the_record_is_not_empty(self):
        assert len(self.CONTROLS) >= 15

    def test_every_control_names_both_the_break_and_the_test(self):
        for broken, test in self.CONTROLS:
            assert broken and test
            assert "::" in test or ".py" in test
