"""A failed run says WHY: the refusal's message and details reach the row and the API.

Found live 2026-10-17 (F11 window 2): an attached run failed `weights_rebuild_failed` in the
cluster and nothing said which tensors or why - only the code reached `GET /runs/{id}`, and the
worker's own log lines are not in the container log. Driven through `drive_recorded_run`, with
the loop replaced only to raise the refusal a real run raised.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from core.errors import Refusal, refuse
from tests.support.jobs import insert_run, memory_engine, read_run, session_for
from training import run_driver, tasks

from .test_run_parts_reach_the_source import RESOLVED, SUBMITTED

RUN = SUBMITTED["run_id"]


class _Source:
    def __init__(self, context) -> None:  # type: ignore[no-untyped-def]
        self.context = context

    def preflight(self, spec) -> None:  # type: ignore[no-untyped-def]
        return None


@pytest.fixture
def failed_engine(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from training import signals

    signals.ensure_registered()
    monkeypatch.setattr(signals.registry, "get", lambda name: _Source)

    def boom(**kwargs):  # type: ignore[no-untyped-def]
        raise refuse(
            "weights_rebuild_failed",
            "2 of 149 parameter tensors could not be rebuilt",
            failed={"model.layers.0.w": "RuntimeError: mapping of buffer object failed"},
            rebuilt=0,
        )

    monkeypatch.setattr(tasks, "drive_run", boom)
    engine = memory_engine()
    with session_for(engine) as session:
        insert_run(
            session,
            RUN,
            definition_submitted=SUBMITTED,
            definition_resolved=RESOLVED,
            definition_sha256="x" * 64,
        )
        session.commit()
        with pytest.raises(Refusal):
            run_driver.drive_recorded_run(RUN, session=session)
    return engine


def test_the_row_carries_the_message_and_the_details(failed_engine) -> None:  # type: ignore[no-untyped-def]
    with session_for(failed_engine) as session:
        row = read_run(session, RUN)
    assert row["failure_code"] == "weights_rebuild_failed"
    assert "2 of 149" in row["failure_message"]
    assert "mapping of buffer object failed" in row["failure_message"]


def test_the_api_serves_it(failed_engine) -> None:  # type: ignore[no-untyped-def]
    app = create_app()

    def override():  # type: ignore[no-untyped-def]
        with session_for(failed_engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = override
    with TestClient(app, raise_server_exceptions=False) as client:
        body = client.get(f"/api/v1/runs/{RUN}").json()
    assert body["failure_code"] == "weights_rebuild_failed"
    assert "mapping of buffer object failed" in body["failure_message"]


def test_a_run_stopped_before_it_started_says_why(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Found live 2026-10-17: a pending run stopped through `POST /runs/{id}/cancel` read
    `completed` with `stop_reason: null`. FR-006.13 keeps it `completed` (the lifecycle has no
    `cancelled`); the REASON was validated and never written, and a run that never started
    has no boundary block to derive it from."""
    from services import stop_service

    engine = memory_engine()
    with session_for(engine) as session:
        insert_run(
            session,
            "queued",
            definition_submitted=SUBMITTED,
            definition_resolved=RESOLVED,
            definition_sha256="x" * 64,
        )
        session.commit()
        stop_service.request_stop(session, "queued")
        session.commit()
    app = create_app()

    def override():  # type: ignore[no-untyped-def]
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = override
    with TestClient(app, raise_server_exceptions=False) as client:
        body = client.get("/api/v1/runs/queued").json()
    assert body["status"] == "completed"
    assert body["stop_reason"] == "operator_stop"
    assert "before the run started" in body["stop_detail"]
