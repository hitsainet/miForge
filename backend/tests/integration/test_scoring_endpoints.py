"""F4's five routes, answering from the mounted app  (FPRD §7; ADR-012, ADR-013, ADR-018).

Reachability is asserted centrally in ``test_reachability.py``, against the mounted route
table and the generated document. This file asserts the **bodies**, and in particular the one
obligation F4 owes the UI: *every number names its scale and sample.* A bare ``0.84`` in a
response is a defect in **this** feature's output, not a gap in F9's rendering, so the
payload is where it is enforced.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from providers.base import GenerationParams
from providers.fakes import FakeModelHandle, FakeProvider
from services.scoring import expectations, retention, service, sets
from services.scoring.tables import SCORES
from tests.support import scores as fixtures
from tests.support.db import migrated_engine, session_for

RULE = {
    "kind": "rule",
    "name": "len",
    "config_sha256": None,
    "expression": '{"min_length": 4}',
}


@pytest.fixture
def engine(tmp_path: Path):
    engine = migrated_engine(tmp_path / "db")
    fixtures.create_score_tables(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    with session_for(engine) as session:
        yield session


@pytest.fixture
def client(engine, monkeypatch: pytest.MonkeyPatch):
    for key in list(sa.__dict__ and __import__("os").environ):
        if key.startswith("PROVIDER_"):
            monkeypatch.delenv(key, raising=False)
    app = create_app()

    def override():
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = override
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def seed(
    session, log_path: Path, *, step: int = 4, floor: float = 0.2, answers: str = "longer"
) -> None:
    expectation = expectations.resolve_expectation(RULE, source="pack")

    def a_set(kind: str) -> sets.ResolvedSet:
        reference = sets.SetReference(
            rendered=f"{kind}_set/general@1",
            structured={"type": f"{kind}_set", "name": "general", "version": "1"},
        )
        return sets.resolve(
            kind, reference, [{"prompt": f"p{i}"} for i in range(4)], markers=("evaluation_only",)
        )

    service.evaluate(
        service.EvaluationSpec(
            run_id="run-1",
            provider_name="in_process",
            aggregation="mean",
            decode=GenerationParams(max_tokens=16),
            test_set=a_set("test"),
            retention_set=a_set("retention"),
            test_expectation=expectation,
            retention_expectation=expectation,
            floor=retention.ResolvedFloor(value=floor, layer="pack_slot"),
            batch_bound=4,
        ),
        step,
        session=session,
        log_path=log_path,
        provider=FakeProvider(answers=answers),
    )
    # Committed here, not at fixture teardown: the route runs on its own session, and an
    # uncommitted seed would make every assertion below test an empty database - which would
    # pass for `scores == []` and fail for everything else, i.e. exactly the confusing half.
    session.commit()


class TestTheProviderListing:
    def test_it_lists_the_in_process_row_even_when_unavailable(self, client) -> None:
        """ADR-013: *an option missing from a response is indistinguishable from a broken
        one.* The API process holds no parameters, so it says so by name."""
        body = client.get("/api/v1/providers").json()
        rows = {row["name"]: row for row in body["providers"]}
        assert "in_process" in rows
        assert rows["in_process"]["available"] is False
        assert rows["in_process"]["unavailable_reason"] == "inference_provider_unavailable"

    def test_it_says_the_in_process_row_is_never_judge_eligible(self, client) -> None:
        """R-02.11, on the listing, because the judge path is the one place a reader could
        reasonably expect it to serve."""
        body = client.get("/api/v1/providers").json()
        rows = {row["name"]: row for row in body["providers"]}
        assert rows["in_process"]["judge_eligible"] is False

    def test_it_publishes_the_capability_vocabulary_without_activations(self, client) -> None:
        body = client.get("/api/v1/providers").json()
        assert body["capability_vocabulary"] == ["embeddings", "generation"]
        assert "activations" not in body["capability_vocabulary"]

    def test_a_configured_endpoint_appears_without_restarting_the_layer(
        self, client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Computed from **live** settings on each call, as F3's capability listing is."""
        monkeypatch.setenv("PROVIDER_JUDGE_BASE_URL", "http://judge.test:8000")
        monkeypatch.setenv("PROVIDER_JUDGE_MODEL", "judge-7b")
        rows = {row["name"]: row for row in client.get("/api/v1/providers").json()["providers"]}
        assert rows["judge"]["base_url"] == "http://judge.test:8000"
        assert rows["judge"]["judge_eligible"] is True

    def test_no_credential_is_ever_serialised(
        self, client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PROVIDER_JUDGE_BASE_URL", "http://judge.test:8000")
        monkeypatch.setenv("PROVIDER_JUDGE_MODEL", "judge-7b")
        monkeypatch.setenv("PROVIDER_JUDGE_API_KEY", "sk-must-not-appear")
        assert "sk-must-not-appear" not in client.get("/api/v1/providers").text


class TestTheProbesRefusalIsTheProduct:
    def test_an_unavailable_provider_refuses_in_the_envelope(self, client) -> None:
        response = client.post("/api/v1/providers/in_process/check", json={"prompt": "ping"})
        assert response.status_code == 400
        body = response.json()
        assert body["error"]["code"] == "inference_provider_unavailable"
        assert "will not substitute" in body["error"]["message"]

    def test_an_unknown_provider_names_what_is_configured(self, client) -> None:
        body = client.post("/api/v1/providers/nope/check", json={"prompt": "p"}).json()
        assert body["error"]["code"] == "inference_provider_unavailable"
        assert "in_process" in body["error"]["details"]["configured"]

    def test_a_live_provider_returns_its_identity_and_no_completion_text(
        self, client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A completion read back from a provider is **untrusted data** and has no reason to
        be echoed into an API response."""
        from providers import registry

        registry.bind_model_handle(FakeModelHandle(answers="hello there"))
        try:
            body = client.post("/api/v1/providers/in_process/check", json={"prompt": "p"}).json()
        finally:
            registry.bind_model_handle(None)
        assert body["identity"]["kind"] == "in_process"
        assert body["capabilities"] == ["generation"]
        assert body["completion_chars"] == len("hello there")
        assert "hello there" not in str(body)


class TestTheScoreSeries:
    def test_every_element_carries_n_and_its_provider(self, client, session, tmp_path) -> None:
        """ADR-012, enforced by the payload so **F9 cannot render a bare number**."""
        seed(session, tmp_path / "run.jsonl")
        body = client.get("/api/v1/runs/run-1/scores").json()
        assert body["scores"]
        for score in body["scores"]:
            assert score["n"] == 4
            assert score["provider"] == "in_process:fake-model"
            assert "longer" not in str(score), "no completion text in a series response"

    def test_it_reports_the_cadences_and_the_exposure_window_as_numbers(
        self, client, session, tmp_path
    ) -> None:
        """FR-004.26: *never implied.*"""
        seed(session, tmp_path / "run.jsonl")
        body = client.get("/api/v1/runs/run-1/scores").json()
        assert body["retention_every"] == 1
        assert body["exposure_window_steps"] == 0

    def test_a_breach_message_says_what_happens_next(self, client, session, tmp_path) -> None:
        """One sentence, in one place, from the API (ADR-021)."""
        # "no" is below the rule's 4-character minimum, so nothing is admitted: score 0.0,
        # which is below the 0.9 floor. A breach needs a score, not just a high floor.
        seed(session, tmp_path / "run.jsonl", floor=0.9, answers="no")
        scores = client.get("/api/v1/runs/run-1/scores").json()["scores"]
        retention_score = next(s for s in scores if s["kind"] == "retention")
        assert retention_score["breached"] is True
        assert "stop condition" in retention_score["message"]

    def test_a_non_breach_message_still_names_the_sample_and_the_provider(
        self, client, session, tmp_path
    ) -> None:
        seed(session, tmp_path / "run.jsonl")
        scores = client.get("/api/v1/runs/run-1/scores").json()["scores"]
        assert "on 4 prompts" in scores[0]["message"]
        assert "in_process:fake-model" in scores[0]["message"]

    def test_discontinuities_are_listed_so_f9_can_break_the_curve(
        self, client, session, tmp_path
    ) -> None:
        seed(session, tmp_path / "run.jsonl")
        session.execute(sa.update(SCORES).values(provider_discontinuity=True))
        session.commit()
        body = client.get("/api/v1/runs/run-1/scores").json()
        assert body["provider_discontinuities"] == [4]

    def test_a_run_with_no_scores_returns_an_empty_series_rather_than_404(self, client) -> None:
        """An unevaluated run is normal, not an error about the run."""
        body = client.get("/api/v1/runs/never-scored/scores").json()
        assert body["scores"] == []


class TestOneStepsDetail:
    def test_it_returns_per_prompt_items_and_says_how_many_were_retained(
        self, client, session, tmp_path
    ) -> None:
        """*An empty ``items`` list means pruned or never retained, not scored nothing* -
        said in the payload rather than left for a reader to infer."""
        seed(session, tmp_path / "run.jsonl")
        body = client.get("/api/v1/runs/run-1/scores/4").json()
        assert body["step"] == 4
        first = body["scores"][0]
        assert first["items_retained"] == 4
        assert [item["verdict"] for item in first["items"]] == ["accept"] * 4

    def test_a_step_with_no_evaluation_refuses_with_a_non_alarming_message(
        self, client, session, tmp_path
    ) -> None:
        seed(session, tmp_path / "run.jsonl")
        body = client.get("/api/v1/runs/run-1/scores/99").json()
        assert body["error"]["code"] == "score_not_found"
        assert "is not an error about the run" in body["error"]["message"]


class TestTheOutOfCadenceRoute:
    def test_an_unknown_run_refuses(self, client) -> None:
        body = client.post("/api/v1/runs/nope/evaluate", json={}).json()
        assert body["error"]["code"] == "run_not_found"

    def test_a_run_with_no_scoring_configuration_refuses_rather_than_defaulting(
        self, client, engine
    ) -> None:
        """*A defaulted test set is a number about a set nobody chose, and a defaulted floor
        passes every run forever.*"""
        from models.run import Run
        from tests.support.db import utc_now

        with session_for(engine) as session:
            session.add(
                Run(id="run-bare", status="running", run_type="change", created_at=utc_now())
            )
        body = client.post("/api/v1/runs/run-bare/evaluate", json={}).json()
        assert body["error"]["code"] == "run_scoring_not_configured"
        assert "defaults" in body["error"]["message"]
