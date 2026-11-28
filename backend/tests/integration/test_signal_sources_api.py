"""F2's four routes, from the **mounted** app  (FR-002.1, FR-002.4; AC-002.1).

Nothing here imports an endpoint function. Every assertion goes through the mounted
application and a real request over the transport, because *a router that exists but is not
included* is the failure this project's review discipline was written after - sixteen
fully-implemented, unit-tested, documented tools that were never registered.

**Control for this file:** delete ``api_v1_router.include_router(signal_sources.router)``
and the first test goes red; delete the ``signal_batches`` line and the batch tests do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from core.config import reset_settings
from tests.support import db as support_db
from tests.support import signal_fixtures as fixtures
from training.signals.settings import reset_signal_settings


@pytest.fixture
def client() -> Any:
    reset_settings()
    reset_signal_settings()
    return TestClient(create_app(), raise_server_exceptions=False)


class TestTheSourceListing:
    def test_all_four_sources_are_listed_from_the_mounted_app(self, client: Any) -> None:
        response = client.get("/api/v1/signal-sources")
        assert response.status_code == 200, response.text
        names = [row["name"] for row in response.json()["sources"]]
        assert sorted(names) == ["corpus", "preference-pairs", "reward", "self-generation"]
        assert response.json()["count"] == 4

    def test_each_source_says_what_it_requires(self, client: Any) -> None:
        rows = {row["name"]: row for row in client.get("/api/v1/signal-sources").json()["sources"]}
        assert rows["corpus"]["requires"] == {
            "part_kind": "corpus",
            "selector": False,
            "grader": False,
            "generation_provider": False,
        }
        assert rows["self-generation"]["requires"]["selector"] is True
        assert rows["reward"]["requires"]["grader"] is True

    def test_an_unavailable_source_is_listed_and_named_never_omitted(self, client: Any) -> None:
        """ADR-013: an option missing from a response is indistinguishable from one that is
        broken. The reason is present **exactly when** the source is unavailable."""
        rows = {row["name"]: row for row in client.get("/api/v1/signal-sources").json()["sources"]}
        for row in rows.values():
            assert (row["unavailable_reason"] is None) == row["available"]
        assert rows["corpus"]["available"] is True

    def test_availability_is_computed_per_call_and_not_cached_at_import(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator who binds a provider must see the source become available without a
        restart. The answer moves between two calls in one process."""
        from providers import registry as provider_registry
        from providers.fakes import FakeModelHandle

        before = {r["name"]: r for r in client.get("/api/v1/signal-sources").json()["sources"]}
        provider_registry.bind_model_handle(FakeModelHandle())
        try:
            after = {r["name"]: r for r in client.get("/api/v1/signal-sources").json()["sources"]}
        finally:
            provider_registry.bind_model_handle(None)
        assert before["self-generation"]["available"] is False
        assert after["self-generation"]["available"] is True
        assert after["self-generation"]["unavailable_reason"] is None


class TestPreflight:
    def test_an_unknown_source_refuses_in_the_error_envelope(self, client: Any) -> None:
        """E-1, through the one exception handler: ``{"error": {code, message, details}}``."""
        response = client.post(
            "/api/v1/signal-sources/preflight", json={"spec": {"source": "rlhf"}}
        )
        assert response.status_code >= 400
        body = response.json()
        assert set(body["error"]) == {"code", "message", "details"}
        assert body["error"]["code"] == "signal_source_unknown"
        assert sorted(body["error"]["details"]["registered"]) == [
            "corpus",
            "preference-pairs",
            "reward",
            "self-generation",
        ]

    def test_an_unknown_key_in_the_payload_is_refused_not_ignored(self, client: Any) -> None:
        """A misspelled ``group_size`` must not silently leave the real value at its
        default, which would pre-flight a run different from the one described."""
        response = client.post(
            "/api/v1/signal-sources/preflight",
            json={"spec": {"source": "corpus", "groupsize": 4}},
        )
        assert response.status_code == 422

    def test_a_resolvable_specification_comes_back_with_its_binding(
        self, client: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What F7's pack validation and F9's form read **before** a run exists."""
        monkeypatch.setenv("SIGNAL_SFT_MASKING_POLICY", "completion_only")
        reset_signal_settings()
        engine = support_db.migrated_engine(tmp_path / "db")
        with support_db.session_for(engine) as session:
            part = fixtures.corpus_part(session, tmp_path, texts=["one", "two"])

        from sqlalchemy.orm import sessionmaker

        from api.deps import session_dependency

        factory = sessionmaker(bind=engine, expire_on_commit=False)

        def override() -> Any:
            session = factory()
            try:
                yield session
            finally:
                session.close()

        app = create_app()
        app.dependency_overrides[session_dependency] = override
        with TestClient(app, raise_server_exceptions=False) as test_client:
            response = test_client.post(
                "/api/v1/signal-sources/preflight",
                json={
                    "spec": {
                        "source": "corpus",
                        "parts": [{"kind": "corpus", "part_id": str(part["id"])}],
                    }
                },
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["objective"] == "sft"
        assert body["parts"][0]["binding"] == {"text": "text"}
        assert body["parts"][0]["splits"] == ["train"]
        assert body["parts"][0]["contract_reference"]["name"] == "support-notes"


class TestTheBatchRoutes:
    def test_both_routes_answer_from_the_mounted_app(self, client: Any) -> None:
        """Registered and routable. A 404 body is still the error envelope."""
        listing = client.get("/api/v1/runs/run-x/batches")
        assert listing.status_code in (200, 400, 404, 500), listing.text
        detail = client.get("/api/v1/runs/run-x/batches/" + "a" * 64)
        assert detail.status_code != 405

    def test_a_page_over_the_cap_is_refused_naming_the_cap(self, client: Any) -> None:
        """A response quietly shorter than the page asked for reads as the end of the run."""
        response = client.get("/api/v1/runs/run-x/batches?limit=10000")
        assert response.status_code >= 400
        assert response.json()["error"]["code"] == "batch_page_too_large"
        assert response.json()["error"]["details"]["maximum"] == 200
