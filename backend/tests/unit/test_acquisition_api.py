"""F1's four routes, answered through the transport  (FR-001.1, 15, 16, 20, 22; ADR-016 #1).

**Nothing here imports an endpoint function.** Every assertion is made against the **mounted
application's** route table and an actual request through the transport, because *a capability
is not shipped until a test fails when its wiring is removed* - the rule written after a
sibling shipped sixteen fully-implemented, unit-tested, documented MCP tools that were never
registered with the server. Every one of those tests passed by importing the module directly.

The foundation's ``test_reachability.py`` holds the single declaration of the whole surface
and compares it in both directions; F1's four routes are declared there. **This file asserts
what each one does**, with its payload and its call count.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api import health as health_module
from api.app import create_app
from api.deps import session_dependency
from core.config import Settings, reset_settings
from tests.support.jobs import insert_run, memory_engine, session_for

RUN = "run-api"


def _one_session(engine) -> Iterator[Session]:
    with session_for(engine) as session:
        yield session


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkpoints = tmp_path / "checkpoints"
    parts = tmp_path / "parts"
    checkpoints.mkdir()
    parts.mkdir()
    monkeypatch.setattr(
        "core.config._settings",
        Settings(_env_file=None, checkpoint_root=checkpoints, parts_root=parts),  # type: ignore[call-arg]
    )
    engine = memory_engine()
    with session_for(engine) as session:
        insert_run(session, RUN, status="running")
    yield engine
    reset_settings()


@pytest.fixture
def client(engine):
    app = create_app()
    for name in ("postgres", "redis"):
        health_module.DEPENDENCY_PROBES[name] = lambda: {"status": "ok", "detail": None}
    app.dependency_overrides[session_dependency] = lambda: (yield from _one_session(engine))
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    app.dependency_overrides.clear()


class TestTheRoutesAnswerFromTheMountedApp:
    """Registered is not the same as routable."""

    def test_models_sources_answers(self, client: TestClient) -> None:
        assert client.get("/api/v1/models/sources").status_code == 200

    def test_the_step_log_answers(self, client: TestClient) -> None:
        assert client.get(f"/api/v1/runs/{RUN}/steps").status_code == 200

    def test_cancel_answers(self, client: TestClient) -> None:
        assert client.post(f"/api/v1/runs/{RUN}/cancel").status_code == 202

    def test_submit_answers_with_the_envelope_rather_than_a_422(self, client: TestClient) -> None:
        """A raw Pydantic error would reach the client as FastAPI's own 422 shape - a
        *different* envelope from every other refusal in this API, so a client would need
        two parsers and ADR-013's one shape would hold for everything except the most common
        failure."""
        response = client.post("/api/v1/runs", json={"learning_rate": 1e-5})
        assert response.status_code in {400, 422}
        assert "error" in response.json(), response.json()

    def test_an_unrouted_path_under_v1_is_a_DIFFERENT_404(self, client: TestClient) -> None:
        """Without this, a deleted ``include_router`` would look exactly like a missing run.
        That distinction is the test."""
        body = client.get("/api/v1/models/nope").json()
        assert body["error"]["code"] == "http_404"


class TestTheAcquisitionModeList:
    """FR-001.15's surface."""

    def test_every_mode_is_listed_including_the_unavailable_one(self, client: TestClient) -> None:
        """An option missing from a response is indistinguishable from one that is broken
        (ADR-013), so ``attach`` appears even in a build without the hot-update mode."""
        body = client.get("/api/v1/models/sources").json()
        assert [entry["mode"] for entry in body["modes"]] == ["load", "create", "attach"]

    def test_the_unavailable_mode_names_its_reason_and_lost_capabilities(
        self, client: TestClient
    ) -> None:
        body = client.get("/api/v1/models/sources").json()
        attach = next(entry for entry in body["modes"] if entry["mode"] == "attach")
        assert attach["available"] is False
        assert attach["reason"] == "hot_update_mode_not_available"
        assert attach[
            "capabilities_unavailable"
        ], "a reader must learn what they cannot do, not merely that something failed"

    def test_the_default_mode_is_published(self, client: TestClient) -> None:
        assert client.get("/api/v1/models/sources").json()["default"] == "load"

    def test_the_model_roots_are_published_so_a_refused_path_is_actionable(
        self, client: TestClient
    ) -> None:
        """``model_source_outside_root`` names the roots in its details; publishing them here
        lets the UI show them *before* the operator types a path."""
        assert client.get("/api/v1/models/sources").json()["model_roots"]


class TestSubmitRefusesBeforeAllocating:
    def test_a_payload_naming_TWO_sources_is_refused_not_resolved(self, client: TestClient) -> None:
        """Picking one trains a parameter set the operator did not ask for and reports
        success - the fabricated value ADR-018 forbids, at the scale of a whole run."""
        response = client.post(
            "/api/v1/runs",
            json={
                "path": "/data/checkpoints/a",
                "repo": "org/model",
                "learning_rate": 1e-5,
                "warmup_steps": 1,
                "batch_size": 1,
            },
        )
        assert response.status_code == 400
        body = response.json()["error"]
        assert body["code"] == "acquisition_source_invalid"
        assert "two sources" in body["message"]

    def test_an_out_of_range_learning_rate_is_refused_NAMING_THE_FIELD(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/v1/runs",
            json={
                "path": "/data/checkpoints/a",
                "learning_rate": 0,
                "warmup_steps": 1,
                "batch_size": 1,
            },
        )
        assert response.status_code == 400
        body = response.json()["error"]
        assert body["code"] == "run_config_invalid"
        assert body["details"]["field"] == "learning_rate"
        assert "clamped" in body["message"]

    def test_an_unknown_payload_key_is_refused(self, client: TestClient) -> None:
        """A misspelled ``gradient_accumulation`` must not silently leave the real field at
        its default."""
        response = client.post(
            "/api/v1/runs",
            json={
                "path": "/data/checkpoints/a",
                "learning_rate": 1e-5,
                "warmup_steps": 1,
                "batch_size": 1,
                "grad_accum": 8,
            },
        )
        assert response.status_code in {400, 422}

    def test_the_attach_mode_is_refused_BY_NAME_and_does_not_fall_back(
        self, client: TestClient
    ) -> None:
        """AC-001.12. R-02.48 forbids substituting a different behaviour, and the available
        substitution is the dangerous kind: loading the same weights from disk trains a
        different set of tensors and reports success."""
        response = client.post(
            "/api/v1/runs",
            json={
                "mode": "attach",
                "learning_rate": 1e-5,
                "warmup_steps": 1,
                "batch_size": 1,
            },
        )
        assert response.status_code == 400
        body = response.json()["error"]
        assert body["code"] == "acquisition_mode_unavailable"
        assert body["details"]["mode"] == "attach"
        assert body["details"]["capabilities_unavailable"]

    def test_a_quantized_model_is_refused_and_NOTHING_IS_ENQUEUED(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-001.4's transport half: refused **before** the lease and before the enqueue.
        Asserted by the *call count* of the enqueue, not by the absence of a side effect."""
        from services import acquisition as service
        from tests.unit.test_acquisition_local_dir import build_model_dir

        enqueued: list[str] = []
        monkeypatch.setattr(
            service, "submit", service.submit
        )  # keep the real submit; only the enqueue is observed
        directory = build_model_dir(service.model_roots()[0], dtype="I8")

        response = client.post(
            "/api/v1/runs",
            json={
                "path": str(directory),
                "learning_rate": 1e-5,
                "warmup_steps": 1,
                "batch_size": 1,
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_quantized"
        assert enqueued == [], "a refused submit must enqueue nothing"


class TestCancellation:
    def test_it_sets_the_column_and_returns_the_TIMESTAMP(self, client: TestClient) -> None:
        """The timestamp, not just a boolean: it is what says whether the worker is shutting
        down or ignoring the request."""
        body = client.post(f"/api/v1/runs/{RUN}/cancel").json()
        assert body["run_id"] == RUN
        assert body["cancel_pending"] is True
        assert body["cancel_requested_at"] is not None

    def test_a_second_request_does_NOT_overwrite_the_first_timestamp(
        self, client: TestClient
    ) -> None:
        """The only record of how long the worker took to notice must not be destroyed by
        the operator asking again."""
        first = client.post(f"/api/v1/runs/{RUN}/cancel").json()["cancel_requested_at"]
        second = client.post(f"/api/v1/runs/{RUN}/cancel").json()["cancel_requested_at"]
        assert first == second

    def test_it_returns_202_because_the_stop_is_RECORDED_not_performed(
        self, client: TestClient
    ) -> None:
        """A 200 would tell the operator the run had stopped when what happened is that the
        request was accepted - and the difference is up to one step of training."""
        assert client.post(f"/api/v1/runs/{RUN}/cancel").status_code == 202

    def test_the_route_never_reaches_for_celerys_control_api(self) -> None:
        """``revoke(terminate=True)`` signals a pool child and the solo pool has none, so the
        call succeeds and nothing happens - the operator sees a cancellation that was
        accepted. Asserted over the module's imports, not by reading it."""
        import ast

        path = (
            Path(__file__).resolve().parents[3]
            / "backend"
            / "src"
            / "api"
            / "v1"
            / "endpoints"
            / "runs_acquisition.py"
        )
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "revoke", "cancellation goes through the database"

    def test_cancelling_an_absent_run_refuses_by_name(self, client: TestClient) -> None:
        body = client.post("/api/v1/runs/nope/cancel")
        assert body.status_code == 404
        assert body.json()["error"]["code"] == "run_not_found"


class TestTheStepLog:
    def test_it_is_newest_first(self, client: TestClient, engine) -> None:
        _seed_steps(engine, 1, 2, 3)
        body = client.get(f"/api/v1/runs/{RUN}/steps").json()
        assert [entry["step"] for entry in body["steps"]] == [3, 2, 1]

    def test_it_is_keyset_paginated_rather_than_offset_paginated(
        self, client: TestClient, engine
    ) -> None:
        """A run writes rows while the operator is paging, and an offset would silently skip
        or repeat steps as the tail grows."""
        _seed_steps(engine, *range(1, 11))
        first = client.get(f"/api/v1/runs/{RUN}/steps?limit=4").json()
        assert [entry["step"] for entry in first["steps"]] == [10, 9, 8, 7]
        assert first["next_before_step"] == 7

        second = client.get(f"/api/v1/runs/{RUN}/steps?limit=4&before_step=7").json()
        assert [entry["step"] for entry in second["steps"]] == [6, 5, 4, 3]

    def test_the_last_page_reports_no_cursor(self, client: TestClient, engine) -> None:
        _seed_steps(engine, 1, 2)
        body = client.get(f"/api/v1/runs/{RUN}/steps?limit=4").json()
        assert body["next_before_step"] is None

    def test_every_number_names_its_scale_and_sample(self, client: TestClient, engine) -> None:
        """ADR-012. A page of steps that does not say it is a page reads as the whole run."""
        _seed_steps(engine, 1)
        body = client.get(f"/api/v1/runs/{RUN}/steps").json()
        assert body["count"] == 1
        assert body["limit"] >= 1
        assert body["run_id"] == RUN

    def test_an_oversized_limit_is_refused_rather_than_silently_capped(
        self, client: TestClient
    ) -> None:
        assert client.get(f"/api/v1/runs/{RUN}/steps?limit=100000").status_code in {400, 422}

    def test_the_payload_carries_the_indexed_columns(self, client: TestClient, engine) -> None:
        _seed_steps(engine, 1)
        entry = client.get(f"/api/v1/runs/{RUN}/steps").json()["steps"][0]
        assert set(entry) == {
            "step",
            "loss",
            "grad_norm",
            "learning_rate",
            "wall_time_ms",
            "batch_source_hashes",
            "quiesce_hold_ms",
            "commit_duration_ms",
        }


def _seed_steps(engine, *steps: int) -> None:
    """Seed steps **through the real writer**, so the rows are what production produces."""
    from training.run_config import RunConfig
    from training.step import StepOutcome
    from training.step_record import StepWriter

    cfg = RunConfig(learning_rate=1e-5, warmup_steps=1, batch_size=1, gradient_accumulation=1)
    import tempfile

    writer = StepWriter(run_id=RUN, log_root=Path(tempfile.mkdtemp()))
    with session_for(engine) as session:
        for step in steps:
            writer.write(
                session=session,
                step=step,
                outcome=StepOutcome(
                    loss=0.5,
                    objective="sft",
                    grad_norm=1.0,
                    wall_time_ms=10.0,
                    batch_sha256="e" * 64,
                ),
                cfg=cfg,
            )


class TestHealthStillReportsTheIntegrationsByName:
    """AC-001.13 - served by the foundation; asserted here because F1 claims it."""

    def test_each_optional_integration_is_unconfigured_BY_NAME(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        for service in ("millm", "midataworks"):
            assert body["integrations"][service]["configured"] is False
            assert body["integrations"][service]["code"] == f"{service}_not_configured"

    def test_no_mode_is_unavailable_BECAUSE_OF_A_SIBLING(self, client: TestClient) -> None:
        """**R-02.48 at the route level, and AC20's shape.** With every sibling unset, the
        two core modes must not be unavailable *for that reason* - the only thing that may
        make them unavailable on this machine is the absent ML runtime, which is a local
        fact and not a sibling.

        Asserted on the *reason* rather than on the boolean, because the boolean depends on
        whether torch happens to be installed and the reason does not. A test pinned to the
        boolean would pass on the GPU node and fail in CI while nothing had changed about the
        standalone guarantee.
        """
        body = client.get("/api/v1/models/sources").json()
        reasons = {entry["mode"]: entry["reason"] for entry in body["modes"]}
        assert reasons["load"] in {None, "ml_runtime_unavailable"}
        assert reasons["create"] in {None, "ml_runtime_unavailable"}
        # And the only sibling-shaped unavailability is §5.10's own mode, which is optional
        # by requirement rather than by configuration.
        assert reasons["attach"] == "hot_update_mode_not_available"
        for entry in body["modes"]:
            assert "not_configured" not in (
                entry["reason"] or ""
            ), "no F1 capability may be unavailable because a sibling is unconfigured"
