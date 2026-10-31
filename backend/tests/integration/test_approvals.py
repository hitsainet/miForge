"""The approval queue's lifecycle, through the mounted app  (F12; FR-012.9, .12 - .14, .16, .22).

The vehicle is ``POST /api/v1/runs/define`` - the product's primary action and a ``gpu_start``
route - so every assertion about "executes exactly once" is about a real run row, a real
definition record, a recorded lease and a recorded enqueue, not about a stub executor.
"""

from __future__ import annotations

import threading
from datetime import timedelta
from typing import Any

import pytest
import sqlalchemy as sa

from tests.support.approval_world import AGENT, GPU, World, definition


def _request_run(world: World, run_id: str = "run-f12") -> dict[str, Any]:
    response = world.client.post("/api/v1/runs/define", json=definition(run_id), headers=AGENT)
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending" and body["approval_id"].startswith("apr_")
    return body


class TestARequestWaits:
    def test_an_agent_define_is_stored_and_nothing_runs(self, world: World) -> None:
        body = _request_run(world)
        assert body["action"] == "gpu_start"
        assert body["request_digest"].startswith("sha256:")
        assert body["summary"]["run_id"] == "run-f12"
        assert body["summary"]["gpu_uuid"] == GPU
        assert body["summary"]["step_budget"] == 40
        assert world.count("mf_runs") == 0
        assert world.enqueued == [] and world.leases == []

    def test_the_operator_call_runs_at_once(self, world: World) -> None:
        response = world.client.post("/api/v1/runs/define", json=definition("run-op"))
        assert response.status_code == 202, response.text
        assert response.json()["run_id"] == "run-op"
        assert "approval_id" not in response.json()
        assert world.enqueued == ["run-op"]
        assert world.leases == [("run-op", GPU)]
        assert world.count("mf_approvals") == 0


class TestApproveRunsItExactlyOnce:
    def test_approve_executes_the_stored_request(self, world: World) -> None:
        pending = _request_run(world)
        response = world.client.post(f"/api/v1/approvals/{pending['approval_id']}/approve")
        assert response.status_code == 200, response.text
        decided = response.json()
        assert decided["status"] == "executed", decided
        assert decided["result"]["run_id"] == "run-f12"
        assert world.enqueued == ["run-f12"]
        assert world.leases == [("run-f12", GPU)]
        assert world.count("mf_runs") == 1
        with world.engine.connect() as conn:
            submitted = conn.execute(
                sa.text("SELECT definition_submitted FROM mf_runs WHERE id = 'run-f12'")
            ).scalar_one()
        assert submitted is not None

    def test_a_second_approve_is_refused_and_runs_nothing(self, world: World) -> None:
        pending = _request_run(world)
        url = f"/api/v1/approvals/{pending['approval_id']}/approve"
        assert world.client.post(url).status_code == 200
        again = world.client.post(url)
        assert again.status_code == 409
        assert again.json()["error"]["code"] == "approval_not_pending"
        assert world.enqueued == ["run-f12"], "the second approve ran the request again"

    def test_two_concurrent_approves_execute_once(self, world: World, monkeypatch) -> None:
        """The claim is the guard, not the status read: both callers pass the read."""
        from services import approval_service, run_launch

        pending = _request_run(world)
        calls: list[str] = []
        real = run_launch.launch
        barrier = threading.Barrier(2)
        real_claim = approval_service._claim

        def slow_claim(session: Any, approval_id: str, to_status: str, **values: Any) -> bool:
            if to_status == "executing":
                barrier.wait(timeout=10)
            return real_claim(session, approval_id, to_status, **values)

        def counting(*args: Any, **kwargs: Any) -> Any:
            calls.append("launch")
            return real(*args, **kwargs)

        monkeypatch.setattr(approval_service, "_claim", slow_claim)
        monkeypatch.setattr(run_launch, "launch", counting)
        results: list[int] = []

        def approve() -> None:
            results.append(
                world.client.post(f"/api/v1/approvals/{pending['approval_id']}/approve").status_code
            )

        threads = [threading.Thread(target=approve) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert sorted(results) == [200, 409], results
        assert calls == ["launch"], f"the executor ran {len(calls)} times"
        assert world.enqueued == ["run-f12"]

    def test_the_executor_refusal_is_recorded_as_failed(self, world: World, monkeypatch) -> None:
        from core.errors import refuse
        from services import run_launch

        pending = _request_run(world)

        def lease_lost(*args: Any, **kwargs: Any) -> Any:
            raise refuse("gpu_lease_held", "Another run holds the card.", gpu_uuid=GPU)

        monkeypatch.setattr(run_launch, "launch", lease_lost)
        decided = world.client.post(f"/api/v1/approvals/{pending['approval_id']}/approve")
        assert decided.status_code == 200
        body = decided.json()
        assert body["status"] == "failed"
        assert body["error"]["code"] == "gpu_lease_held"
        assert body["result"] is None
        assert world.count("mf_runs") == 0


class TestTheOperatorDecides:
    @pytest.mark.parametrize("verb", ["approve", "reject"])
    def test_an_agent_cannot_decide(self, world: World, verb: str) -> None:
        pending = _request_run(world)
        response = world.client.post(
            f"/api/v1/approvals/{pending['approval_id']}/{verb}", headers=AGENT
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "agent_cannot_decide"
        status = world.client.get(f"/api/v1/approvals/{pending['approval_id']}").json()
        assert status["status"] == "pending"
        assert world.enqueued == []

    def test_reject_records_the_reason_and_never_runs(self, world: World) -> None:
        pending = _request_run(world)
        url = f"/api/v1/approvals/{pending['approval_id']}"
        rejected = world.client.post(url + "/reject", json={"reason": "not tonight"})
        assert rejected.status_code == 200
        assert rejected.json()["status"] == "rejected"
        assert rejected.json()["reason"] == "not tonight"
        assert world.client.post(url + "/approve").status_code == 409
        assert world.enqueued == []


class TestExpiryAndTamper:
    def test_an_expired_request_never_runs(self, world: World, monkeypatch) -> None:
        from services import approval_service

        pending = _request_run(world)
        later = approval_service.now() + timedelta(hours=24, seconds=1)
        monkeypatch.setattr(approval_service, "now", lambda: later)
        response = world.client.post(f"/api/v1/approvals/{pending['approval_id']}/approve")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "approval_expired"
        status = world.client.get(f"/api/v1/approvals/{pending['approval_id']}").json()
        assert status["status"] == "expired"
        assert world.enqueued == [] and world.count("mf_runs") == 0

    def test_listing_marks_due_rows_expired(self, world: World, monkeypatch) -> None:
        from services import approval_service

        _request_run(world)
        later = approval_service.now() + timedelta(hours=25)
        monkeypatch.setattr(approval_service, "now", lambda: later)
        rows = world.approvals()
        assert [row["status"] for row in rows] == ["expired"]

    def test_an_edited_payload_fails_as_tampered(self, world: World) -> None:
        pending = _request_run(world)
        with world.session() as session:
            from models.approval import Approval

            row = session.get(Approval, pending["approval_id"])
            assert row is not None
            changed = dict(row.payload)
            changed["definition"] = {**changed["definition"], "run_id": "run-swapped"}
            row.payload = changed
        response = world.client.post(f"/api/v1/approvals/{pending['approval_id']}/approve")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "approval_tampered"
        status = world.client.get(f"/api/v1/approvals/{pending['approval_id']}").json()
        assert status["status"] == "failed"
        assert status["error"]["code"] == "approval_tampered"
        assert world.enqueued == [] and world.count("mf_runs") == 0


class TestTheActionSet:
    def test_the_live_action_set_is_the_four(self, world: World) -> None:
        from models.approval import APPROVAL_ACTION_NAMES

        body = world.client.get("/api/v1/approvals/actions").json()
        assert set(body["actions"]) == {"gpu_start", "hotupdate_attach", "delete", "config_write"}
        assert set(body["actions"]) == set(APPROVAL_ACTION_NAMES)
        assert all(len(text) > 20 for text in body["actions"].values())


class TestIdentity:
    @pytest.mark.parametrize("value", ["", "agent:", "Agent:Forge", "forge-mcp", "agent:-x"])
    def test_a_malformed_identity_is_refused_on_a_read_too(self, world: World, value: str) -> None:
        response = world.client.get("/api/v1/approvals", headers={"X-Miforge-Agent": value})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_agent_identity"

    def test_a_malformed_identity_never_reaches_a_gated_route(self, world: World) -> None:
        response = world.client.post(
            "/api/v1/runs/define", json=definition(), headers={"X-Miforge-Agent": "root"}
        )
        assert response.status_code == 400
        assert world.count("mf_approvals") == 0 and world.enqueued == []

    def test_the_requester_is_recorded(self, world: World) -> None:
        world.client.post(
            "/api/v1/runs/define",
            json=definition(),
            headers={"X-Miforge-Agent": "agent:mistudio-mcp"},
        )
        assert [row["requested_by"] for row in world.approvals()] == ["agent:mistudio-mcp"]
