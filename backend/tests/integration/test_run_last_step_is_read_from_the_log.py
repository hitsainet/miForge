"""``GET /runs/{id}`` serves the last step the run LOG holds  (ADR-004; found live).

F11 window 2, 2026-10-18: run f11w2-r3 trained to step 650 and ``GET /runs/f11w2-r3`` said
``last_step: null`` throughout, because ``mf_runs.last_step`` has no production writer. The
log holds every step and is written first, so the route reads it there.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from core.config import reset_settings
from tests.support.jobs import insert_run, memory_engine, session_for
from training.run_config import RunConfig
from training.step import StepOutcome
from training.step_record import StepWriter

RUN = "run-last-step"
CFG = RunConfig(learning_rate=2e-5, warmup_steps=5, batch_size=4, gradient_accumulation=2)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRAINER_STATE_ROOT", str(tmp_path / "trainer-state"))
    reset_settings()
    from training.tasks import run_log_root

    engine = memory_engine()
    with session_for(engine) as session:
        insert_run(session, RUN, status="failed", failure_code="worker_lost")
        insert_run(session, "run-without-a-log", status="failed", failure_code="worker_lost")
    writer = StepWriter(run_id=RUN, log_root=run_log_root())
    for step in (1, 2, 3, 4):
        writer.write(
            session=None,  # the crash case: the index never got these rows
            step=step,
            outcome=StepOutcome(
                loss=0.5, objective="sft", grad_norm=1.0, wall_time_ms=1.0, batch_sha256="e" * 64
            ),
            cfg=CFG,
        )
    app = create_app()

    def override():  # type: ignore[no-untyped-def]
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = override
    with TestClient(app) as test_client:
        yield test_client
    reset_settings()


def test_the_route_serves_the_logs_last_step_when_the_index_has_none(client) -> None:
    body = client.get(f"/api/v1/runs/{RUN}").json()
    assert body["last_step"] == 4


def test_a_run_with_no_log_serves_null_not_zero(client) -> None:
    """``0`` would claim a step happened; ``null`` says nothing is recorded."""
    assert client.get("/api/v1/runs/run-without-a-log").json()["last_step"] is None
