"""``api.v1.endpoints.wsp_runs``'s routes answer as the contract says (021 FR-021.13).

Every run here is made by the production path: ``wsp.runs.prepare_start`` and ``create_run`` (or the
gated route itself), pushed and taken in by the REAL driver and intake against
``test_dispatch``'s fake supervisor, and the program sessions' patches are made by the worker's
own ``wsp_worker.program`` (``test_replay_and_rerun._session``). A running session's editor call
is metered through the gateway ledger's own ``reserve`` and ``settle``, as the gateway's internal
routes call them. The start gate runs on ``test_wsp_approvals``' world.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.integration.wsp.test_dispatch import World, _running
from tests.integration.wsp.test_dispatch import world as dispatch_world  # noqa: F401
from tests.integration.wsp.test_replay_and_rerun import PROGRAM, _session
from tests.integration.wsp.test_run_cause import _client
from tests.integration.wsp.test_wsp_approvals import World as GateWorld
from tests.integration.wsp.test_wsp_approvals import world as gate_world  # noqa: F401
from tests.support.answer_cases import answered, assert_covered, case
from wsp import gateway_ledger, runs

ROUTER = "api.v1.endpoints.wsp_runs"


@pytest.fixture()
def runs_world(request: pytest.FixtureRequest) -> World:
    """``test_dispatch``'s world (taken by name, so its imported fixture is not shadowed)."""
    world: World = request.getfixturevalue("dispatch_world")
    return world


@pytest.fixture()
def gate(request: pytest.FixtureRequest) -> GateWorld:
    """``test_wsp_approvals``' start-gate world."""
    world: GateWorld = request.getfixturevalue("gate_world")
    return world


@pytest.fixture()
def api(runs_world: World, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return _client(runs_world, monkeypatch)


def _metered_call(world: World, run_id: str) -> None:
    """One served editor call on a running session, through the ledger the gateway calls."""
    with world.factory() as session:
        row = runs.get_run(session, run_id)
        assert row.status == "running" and row.gateway_token_sha256, row.status
        reservation = gateway_ledger.reserve(
            session,
            run_id=run_id,
            token_sha256=row.gateway_token_sha256,
            worst_case_usd=Decimal("0"),
            worst_case_tokens=300,
            external=False,
            settings=world.settings,
        )
        gateway_ledger.settle(
            session,
            reservation_id=reservation,
            usage={"input_tokens": 120, "output_tokens": 40, "cost_usd": 0},
            call={
                "outcome": "served",
                "http_status": 200,
                "latency_ms": 85,
                "finish_reason": "stop",
                "sent_bytes": 2048,
                "prompt_messages": 3,
                "conversation_sha256": "e" * 64,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
        )
        session.commit()


def _program_run(world: World, tmp_path: Path) -> tuple[str, str]:
    """A completed program session whose patch became a candidate: ``(run_id, program_sha256)``."""
    run_id, _ = _session(world, tmp_path, mode="program", source=PROGRAM)
    row = world.run(run_id)
    assert row["status"] == "completed" and row["outcome"] == "candidate", row["failure"]
    return run_id, str(row["program_sha256"])


@case(ROUTER, "POST /api/v1/wsp/runs")
def test_start_run(gate: GateWorld) -> None:
    """Gated: an agent's local repair waits for ``wsp_session``; the operator's is queued."""
    answered("POST /api/v1/wsp/runs", gate.start("ep-local"), pending=True)
    queued = answered("POST /api/v1/wsp/runs", gate.start("ep-local", agent=False))
    assert queued["status"] == "queued"


@case(ROUTER, "GET /api/v1/wsp/pool")
def test_get_pool(runs_world: World, api: TestClient) -> None:
    """One worker holding a pushed run and one run still queued behind it."""
    _running(runs_world)
    runs_world.queue()
    pool = answered("GET /api/v1/wsp/pool", api.get("/api/v1/wsp/pool"))
    assert pool["busy"] == 1 and pool["workers"]


@case(ROUTER, "GET /api/v1/wsp/runs")
def test_list_runs(runs_world: World, api: TestClient, tmp_path: Path) -> None:
    """A completed candidate run, a running one with a metered call (its live progress) and a
    queued one."""
    _program_run(runs_world, tmp_path)
    running, _ = _running(runs_world)
    _metered_call(runs_world, running)
    runs_world.queue()
    listed = answered("GET /api/v1/wsp/runs", api.get("/api/v1/wsp/runs"))
    assert len(listed["runs"]) == 3
    assert any(r["progress"] is not None for r in listed["runs"])


@case(ROUTER, "GET /api/v1/wsp/runs/{run_id}")
def test_get_run(runs_world: World, api: TestClient, tmp_path: Path) -> None:
    run_id, _ = _program_run(runs_world, tmp_path)
    document = answered("GET /api/v1/wsp/runs/{run_id}", api.get(f"/api/v1/wsp/runs/{run_id}"))
    assert document["candidate_id"] is not None


@case(ROUTER, "GET /api/v1/wsp/runs/{run_id}/record")
def test_get_run_record(runs_world: World, api: TestClient, tmp_path: Path) -> None:
    run_id, _ = _program_run(runs_world, tmp_path)
    record = answered(
        "GET /api/v1/wsp/runs/{run_id}/record", api.get(f"/api/v1/wsp/runs/{run_id}/record")
    )
    assert record["run_id"] == run_id and record["document"]["run_id"] == run_id


@case(ROUTER, "GET /api/v1/wsp/runs/{run_id}/ledger")
def test_get_run_ledger(runs_world: World, api: TestClient) -> None:
    """A running session with one metered call: one reservation and one usage row."""
    running, _ = _running(runs_world)
    _metered_call(runs_world, running)
    ledger = answered(
        "GET /api/v1/wsp/runs/{run_id}/ledger", api.get(f"/api/v1/wsp/runs/{running}/ledger")
    )
    assert ledger["reservations"] and ledger["usage"], ledger


@case(ROUTER, "GET /api/v1/wsp/runs/{run_id}/progress")
def test_get_progress(runs_world: World, api: TestClient, tmp_path: Path) -> None:
    """Both shapes: a running session's live progress, and a completed run's final figures."""
    key = "GET /api/v1/wsp/runs/{run_id}/progress"
    done, _ = _session(runs_world, tmp_path)  # first: the one worker is then free again
    running, _ = _running(runs_world)
    _metered_call(runs_world, running)
    assert answered(key, api.get(f"/api/v1/wsp/runs/{running}/progress"))["live"] is True
    assert answered(key, api.get(f"/api/v1/wsp/runs/{done}/progress"))["live"] is False


@case(ROUTER, "GET /api/v1/wsp/runs/{run_id}/calls")
def test_get_calls(runs_world: World, api: TestClient) -> None:
    running, _ = _running(runs_world)
    _metered_call(runs_world, running)
    calls = answered(
        "GET /api/v1/wsp/runs/{run_id}/calls", api.get(f"/api/v1/wsp/runs/{running}/calls")
    )
    assert len(calls["calls"]) == 1


@case(ROUTER, "GET /api/v1/wsp/runs/{run_id}/transcript")
def test_get_transcript(runs_world: World, api: TestClient, tmp_path: Path) -> None:
    run_id, _ = _session(runs_world, tmp_path)
    transcript = answered(
        "GET /api/v1/wsp/runs/{run_id}/transcript",
        api.get(f"/api/v1/wsp/runs/{run_id}/transcript"),
    )
    assert transcript["entries"] and transcript["transcript_sha256"]


@case(ROUTER, "POST /api/v1/wsp/runs/{run_id}/cancel")
def test_cancel_run(runs_world: World, api: TestClient) -> None:
    run_id = runs_world.queue()
    cancelled = answered(
        "POST /api/v1/wsp/runs/{run_id}/cancel", api.post(f"/api/v1/wsp/runs/{run_id}/cancel")
    )
    assert cancelled["status"] == "cancelled"


@case(ROUTER, "POST /api/v1/wsp/runs/{run_id}/replay")
def test_replay_run(runs_world: World, api: TestClient, tmp_path: Path) -> None:
    run_id, _ = _session(runs_world, tmp_path)
    replay = answered(
        "POST /api/v1/wsp/runs/{run_id}/replay", api.post(f"/api/v1/wsp/runs/{run_id}/replay")
    )
    assert replay["status"] == "queued"


@case(ROUTER, "GET /api/v1/wsp/programs/{program_sha256}")
def test_get_program(runs_world: World, api: TestClient, tmp_path: Path) -> None:
    """The program with one rerun of it already queued (``reruns`` non-empty)."""
    _, sha = _program_run(runs_world, tmp_path)
    answered(
        "POST /api/v1/wsp/programs/{program_sha256}/rerun",
        api.post(f"/api/v1/wsp/programs/{sha}/rerun"),
    )
    program = answered(
        "GET /api/v1/wsp/programs/{program_sha256}", api.get(f"/api/v1/wsp/programs/{sha}")
    )
    assert program["source"] == PROGRAM and len(program["reruns"]) == 1


@case(ROUTER, "POST /api/v1/wsp/programs/{program_sha256}/rerun")
def test_rerun_program(runs_world: World, api: TestClient, tmp_path: Path) -> None:
    _, sha = _program_run(runs_world, tmp_path)
    rerun = answered(
        "POST /api/v1/wsp/programs/{program_sha256}/rerun",
        api.post(
            f"/api/v1/wsp/programs/{sha}/rerun", json={"support_set": {"kind": "practice_train"}}
        ),
    )
    assert rerun["status"] == "queued"


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
