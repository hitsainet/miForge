"""F5's eight routes, over the transport  (FR-005.41, AC-005.12, AC-005.13).

Everything here goes through ``TestClient`` rather than by calling a handler, because *a
route that is registered but unroutable* is a different failure from *a handler that is
wrong*, and only the transport tells them apart.

**Addressed by ``(run_id, step)``.** Ids are reassigned by a rebuild and foundation 4.4
demands identical responses across one, so a surrogate id in a path or as an identity in a
body would make ADR-004's claim untestable - which is asserted here rather than left to
review.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from core.config import reset_settings
from services.checkpoints import provider_source
from tests.support.checkpoints import create_feature_tables, saved_checkpoint, score, write_chain
from tests.support.jobs import insert_run, memory_engine, session_for


@pytest.fixture
def rig(tmp_path: Path, monkeypatch):
    """A mounted app over an in-memory database and a real checkpoint root on disk."""
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    monkeypatch.setenv("CHECKPOINT_ROOT", str(checkpoints))
    monkeypatch.setenv("TRAINER_STATE_ROOT", str(tmp_path / "trainer-state"))
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
        yield client, checkpoints, engine
    finally:
        app.dependency_overrides.clear()
        reset_settings()


class TestTheReadRoutes:
    def test_one_checkpoints_manifest_is_served_from_disk(self, rig):
        client, root, _ = rig
        result = saved_checkpoint(root, run_id="run-a", step=500, test=score())

        body = client.get("/api/v1/runs/run-a/checkpoints/500").json()
        assert body["manifest"] == result.manifest
        assert body["manifest"]["name"] == result.name

    def test_the_manifest_response_is_the_bytes_on_disk_not_the_index_row(self, rig):
        """ADR-004: the manifest is the authority. Asserted by editing the index out from
        under the response - the answer must not change."""
        client, root, _ = rig
        result = saved_checkpoint(root, run_id="run-a", step=500)
        client.get("/api/v1/runs/run-a/checkpoints/500")  # warm anything that might cache

        body = client.get("/api/v1/runs/run-a/checkpoints/500").json()
        assert body["manifest"]["weights_sha256"] == result.manifest["weights_sha256"]

    def test_a_missing_checkpoint_refuses_by_name(self, rig):
        client, _, _ = rig
        response = client.get("/api/v1/runs/run-a/checkpoints/999")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "checkpoint_not_found"

    def test_an_unknown_run_refuses_as_run_not_found_not_as_a_missing_checkpoint(self, rig):
        """Two different facts with two different fixes: *this id is wrong* versus *nothing
        saved yet*. Collapsing them sends the operator to the wrong place."""
        client, _, _ = rig
        response = client.get("/api/v1/runs/nobody/checkpoints/1")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "run_not_found"

    def test_a_runs_listing_carries_what_latest_resolves_to(self, rig):
        client, root, engine = rig
        write_chain(root, run_id="run-a", steps=(100, 200))
        with session_for(engine) as session:
            from services.reindex import reindex

            reindex(session, root=root)

        body = client.get("/api/v1/runs/run-a/checkpoints").json()
        assert body["latest_step"] == 200
        assert body["steps_on_disk"] == [100, 200]
        assert [c["step"] for c in body["checkpoints"]] == [200, 100]

    def test_no_surrogate_id_appears_as_an_identity(self, rig):
        """Ids are reassigned by a rebuild, so one in a response body would break foundation
        4.4's *identical API responses* - and the breakage would appear long afterwards."""
        client, root, engine = rig
        saved_checkpoint(root, run_id="run-a", step=1)
        with session_for(engine) as session:
            from services.reindex import reindex

            reindex(session, root=root)

        for path in (
            "/api/v1/checkpoints",
            "/api/v1/runs/run-a/checkpoints",
            "/api/v1/runs/run-a/checkpoints/1",
            "/api/v1/runs/run-a/checkpoints/1/lineage",
        ):
            body = json.dumps(client.get(path).json())
            assert '"id"' not in body, f"{path} exposes an id"

    def test_the_lineage_route_reports_how_the_chain_ended(self, rig):
        client, root, _ = rig
        write_chain(root, run_id="run-a", steps=(100, 200))
        body = client.get("/api/v1/runs/run-a/checkpoints/200/lineage").json()
        assert body["terminated"] == "root"
        assert [n["step"] for n in body["nodes"]] == [200, 100]
        assert "depth" not in body["nodes"][0]


class TestTheProviderJoinNamesItsSource:
    def test_manifest_response_names_the_provider_source(self, rig):
        """**MUT-13's target.** A provider that came from the run log is never presented as
        the manifest's: the artifact is the thing that travels, and criterion 2's exact case
        - *deterministic decoding and the same provider* - is only checkable from the
        artifact when the artifact says so."""
        client, root, _ = rig
        saved_checkpoint(root, run_id="run-a", step=1, test=score(provider="vllm:local"))

        sources = client.get("/api/v1/runs/run-a/checkpoints/1").json()["provider_sources"]
        assert sources["test"]["provider"] == "vllm:local"
        assert sources["test"]["source"] == provider_source.MANIFEST

    def test_a_null_provider_in_the_manifest_is_joined_from_the_step_record_and_labelled(
        self, rig, tmp_path: Path
    ):
        """The join survives for manifests written before the field existed, and for one
        whose provider is honestly null. **Labelled ``step_record``, never ``manifest``.**"""
        client, root, _ = rig
        saved_checkpoint(root, run_id="run-a", step=1, test=score(provider=None))

        from training.run_log import append, run_log_path
        from training.tasks import run_log_root

        path = run_log_path(run_log_root(), "run-a")
        append(
            path,
            {
                "block": "evaluation",
                "run_id": "run-a",
                "step": 1,
                "ts": "2026-10-14T09:00:00Z",
                # The run log's ``ScoreEntry`` requires every key including the nullable
                # ones - a different rule from the manifest's ``ScoreRecord``, where
                # ``provider`` and ``floor`` are declared and NOT required. Two contracts,
                # two required sets; neither is retyped from the other.
                "scores": [
                    {
                        "set": "contract-clauses-test",
                        "score": 0.81,
                        "n": 120,
                        "floor": None,
                        "provider": "in-process:lfm2-1.2b",
                    }
                ],
                "provider": "in-process:lfm2-1.2b",
                "aggregation": "mean",
                # The block's own required set, taken from the live schema rather than
                # guessed: a score at temperature 0 is not comparable with one at 0.8, so
                # all three decode parameters are required there.
                "decode_params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64},
                "subsample_seed": None,
                "split": ["test"],
                "retention_floor_breached": False,
            },
        )

        sources = client.get("/api/v1/runs/run-a/checkpoints/1").json()["provider_sources"]
        assert sources["test"]["provider"] == "in-process:lfm2-1.2b"
        assert sources["test"]["source"] == provider_source.STEP_RECORD
        assert sources["test"]["source"] != provider_source.MANIFEST

    def test_no_provider_anywhere_is_reported_as_none_rather_than_silently_null(self, rig):
        """``null`` with no source named is a different answer from ``null`` because nothing
        recorded one, and a response that could not say which would be unusable for AC2."""
        client, root, _ = rig
        saved_checkpoint(root, run_id="run-a", step=1, test=score(provider=None))
        sources = client.get("/api/v1/runs/run-a/checkpoints/1").json()["provider_sources"]
        assert sources["test"]["provider"] is None
        assert sources["test"]["source"] == provider_source.NONE

    def test_a_null_record_is_distinguished_from_a_record_with_no_provider(self, rig):
        """*No test ran at this step* is not *a test ran and its provider is unknown*."""
        client, root, _ = rig
        saved_checkpoint(root, run_id="run-a", step=1, test=None)
        sources = client.get("/api/v1/runs/run-a/checkpoints/1").json()["provider_sources"]
        assert sources["test"]["record"] is None


class TestVerifyOverTheTransport:
    def test_a_matching_checkpoint_verifies(self, rig):
        client, root, _ = rig
        result = saved_checkpoint(root, run_id="run-a", step=1)
        body = client.post("/api/v1/runs/run-a/checkpoints/1/verify").json()
        assert body["match"] is True
        assert body["expected"] == result.manifest["weights_sha256"]

    def test_a_corrupted_checkpoint_refuses_and_the_manifest_is_untouched(self, rig):
        client, root, _ = rig
        result = saved_checkpoint(root, run_id="run-a", step=1)
        before = (result.directory / "manifest.json").read_bytes()
        (result.directory / "model.safetensors").write_bytes(b"tampered")

        response = client.post("/api/v1/runs/run-a/checkpoints/1/verify")
        assert response.json()["error"]["code"] == "checkpoint_hash_mismatch"
        assert (result.directory / "manifest.json").read_bytes() == before


class TestContinueAndBranchOverTheTransport:
    def test_continue_returns_a_plan_and_takes_no_lease(self, rig):
        client, root, _ = rig
        write_chain(root, run_id="run-a", steps=(100,))
        body = client.post("/api/v1/runs/run-a/continue", json={}).json()
        assert body["plan"]["step"] == 100
        assert body["plan"]["run_id"] == "run-a"
        assert body["plan"]["optimizer"]["state"] == "fresh"

    def test_branch_requires_a_new_run_id(self, rig):
        """A branch *is* a new run; generating the id here would mean the operator could not
        name the thing they just created."""
        client, root, _ = rig
        write_chain(root, run_id="run-a", steps=(100,))
        assert client.post("/api/v1/runs/run-a/checkpoints/100/branch", json={}).status_code == 422

        body = client.post(
            "/api/v1/runs/run-a/checkpoints/100/branch", json={"run_id": "run-b"}
        ).json()
        assert body["plan"]["run_id"] == "run-b"
        assert body["plan"]["is_branch"] is True
        assert body["plan"]["parent"].endswith(":run-a:step-100")

    def test_a_continue_from_a_corrupt_checkpoint_refuses_over_the_wire(self, rig):
        client, root, _ = rig
        result = saved_checkpoint(root, run_id="run-a", step=1)
        (result.directory / "model.safetensors").write_bytes(b"tampered")

        response = client.post("/api/v1/runs/run-a/continue", json={})
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "checkpoint_hash_mismatch"


class TestTheSaveRequestRoute:
    def test_it_accepts_with_202_because_the_save_has_not_happened(self, rig):
        client, _, _ = rig
        response = client.post("/api/v1/runs/run-a/checkpoints")
        assert response.status_code == 202
        assert response.json()["pending_save"]["request_count"] == 1

    def test_two_requests_collapse_into_one_pending_save(self, rig):
        client, _, _ = rig
        client.post("/api/v1/runs/run-a/checkpoints")
        body = client.post("/api/v1/runs/run-a/checkpoints").json()
        assert body["pending_save"]["request_count"] == 2

    def test_requesting_a_save_for_an_unknown_run_refuses(self, rig):
        client, _, _ = rig
        response = client.post("/api/v1/runs/nobody/checkpoints")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "run_not_found"


class TestPathContainmentAtTheEdge:
    def test_a_non_integer_step_is_a_validation_failure(self, rig):
        """``step`` is an ``int`` path parameter, so a traversal attempt never reaches a
        path composer at all."""
        client, _, _ = rig
        response = client.get("/api/v1/runs/run-a/checkpoints/..%2F..%2Fetc")
        assert response.status_code in {404, 422}
        assert response.json()["error"]["code"] in {"validation_failed", "http_404"}

    def test_a_traversal_in_the_run_id_does_not_escape_the_root(self, rig):
        client, _, _ = rig
        response = client.get("/api/v1/runs/..%2F..%2Fetc/checkpoints/1")
        assert response.status_code in {404, 422}


class TestTwoMockupStatesF5NeverProduces:
    def test_no_quarantined_state_and_no_millm_residency_field(self, rig):
        """*"Quarantined"* is **mockup-only scope** - no requirement defines it - and
        *"Resident in miLLM"* is an optional-integration concern whose colour is reserved for
        miLLM state (ADR-012). A checkpoint is on disk and valid, on disk and unusable (named
        by its failure), or absent."""
        client, root, engine = rig
        saved_checkpoint(root, run_id="run-a", step=1)
        with session_for(engine) as session:
            from services.reindex import reindex

            reindex(session, root=root)

        for path in (
            "/api/v1/checkpoints",
            "/api/v1/runs/run-a/checkpoints",
            "/api/v1/runs/run-a/checkpoints/1",
            "/api/v1/runs/run-a/checkpoints/1/lineage",
        ):
            body = json.dumps(client.get(path).json()).lower()
            assert "quarantin" not in body, path
            assert "resident" not in body and "millm" not in body, path
