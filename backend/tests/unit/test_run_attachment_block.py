"""F11 16.4's data, served: the run's attachment block and the two §5.10 step series.

Asserted against the **mounted app**, never by importing the endpoint function. The block
exists so the shell can render the attachment chip; the secret-handling half matters as much
as the presence half - ``attachment_id`` authorises miLLM's holder-only calls (migration
0017) and must never appear in a response body.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api import health as health_module
from api.app import create_app
from api.deps import session_dependency
from core.config import Settings, reset_settings
from tests.support.jobs import insert_run, memory_engine, session_for

ATTACHED = "run-attached"
PLAIN = "run-plain"
SERVING = "http://millm-backend.millm.svc:8000"
SECRET_ID = "att-SECRET-0123456789abcdef"


def _one_session(engine) -> Iterator[Session]:  # type: ignore[no-untyped-def]
    with session_for(engine) as session:
        yield session


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "core.config._settings",
        Settings(_env_file=None, checkpoint_root=tmp_path, parts_root=tmp_path),  # type: ignore[call-arg]
    )
    from models.run import Run

    engine = memory_engine()
    with session_for(engine) as session:
        insert_run(session, ATTACHED, status="running")
        insert_run(session, PLAIN, status="running")
        session.execute(
            sa.update(Run.__table__)
            .where(Run.__table__.c.id == ATTACHED)
            .values(hotupdate_base_url=SERVING, attachment_id=SECRET_ID)
        )
    yield engine
    reset_settings()


@pytest.fixture
def client(engine):  # type: ignore[no-untyped-def]
    app = create_app()
    for name in ("postgres", "redis"):
        health_module.DEPENDENCY_PROBES[name] = lambda: {"status": "ok", "detail": None}
    app.dependency_overrides[session_dependency] = lambda: (yield from _one_session(engine))
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestTheAttachmentBlock:
    def test_an_attached_run_carries_where_it_attached(self, client: TestClient) -> None:
        body = client.get(f"/api/v1/runs/{ATTACHED}").json()
        assert body["attachment"] == {
            "serving_base_url": SERVING,
            "resident_model": None,
            "weights_state": None,
            "holder": None,
            "stale": None,
            "attached": None,
            "observed_at": None,
        }

    def test_the_trainers_observation_is_served_as_written(  # type: ignore[no-untyped-def]
        self, client: TestClient, engine
    ) -> None:
        """Operator Q1 (2026-10-16): the trainer writes the observed state into the run
        row; this read-only route serves that column and never asks miLLM itself."""
        from training.hotupdate import run_state

        observed = run_state.status_record(
            {
                "attached": True,
                # miLLM's AttachmentStatus object (millm/api/schemas/weights.py).
                "attachment": {
                    "state": "stale",
                    "role": "trainer",
                    "label": "nightly-forge",
                    "attached_at": "2026-10-17T00:00:00+00:00",
                    "last_heartbeat_at": None,
                    "stale_since": "2026-10-17T00:00:30+00:00",
                },
                "weights_state": "torn",
                "resident_model": {"model_id": 7, "name": "LFM2-1.2B", "manifest_hash": None},
            },
            holds_attachment=False,
        )
        with session_for(engine) as session:
            run_state.record_status(session, ATTACHED, observed)
        block = client.get(f"/api/v1/runs/{ATTACHED}").json()["attachment"]
        assert block["weights_state"] == "torn"
        assert block["resident_model"] == "LFM2-1.2B"
        assert block["holder"] == "trainer:nightly-forge"
        assert block["stale"] is True
        assert block["attached"] is True
        assert block["observed_at"] == observed["observed_at"]

    def test_this_runs_own_attachment_names_no_holder(self) -> None:
        from training.hotupdate import run_state

        record = run_state.status_record(
            {
                "attached": True,
                "attachment": {"state": "active", "role": "trainer", "label": "miforge"},
                "weights_state": "clean",
            },
            holds_attachment=True,
        )
        assert record["holder"] is None

    def test_the_attachment_id_never_reaches_the_response(self, client: TestClient) -> None:
        response = client.get(f"/api/v1/runs/{ATTACHED}")
        assert response.status_code == 200
        assert SECRET_ID not in response.text
        assert "attachment_id" not in response.text

    def test_any_other_run_says_nothing_is_attached(self, client: TestClient) -> None:
        body = client.get(f"/api/v1/runs/{PLAIN}").json()
        assert "attachment" in body, "present and null: an absent key is an omission"
        assert body["attachment"] is None


class TestTheStepSeriesAreServed:
    def test_the_two_timings_reach_the_step_payload(self, client: TestClient, engine) -> None:  # type: ignore[no-untyped-def]
        """Seeded through the real writer with a bracketed outcome, read through the API."""
        import tempfile

        from training.run_config import RunConfig
        from training.step import StepOutcome
        from training.step_record import StepWriter

        cfg = RunConfig(learning_rate=1e-5, warmup_steps=1, batch_size=1, gradient_accumulation=1)
        writer = StepWriter(run_id=ATTACHED, log_root=Path(tempfile.mkdtemp()))
        with session_for(engine) as session:
            for step, (hold, commit) in enumerate([(231.5, 12.6), (229.0, 13.4)], start=1):
                writer.write(
                    session=session,
                    step=step,
                    outcome=StepOutcome(
                        loss=0.5,
                        objective="sft",
                        grad_norm=1.0,
                        wall_time_ms=10.0,
                        batch_sha256="e" * 64,
                        quiesce_hold_ms=hold,
                        commit_duration_ms=commit,
                    ),
                    cfg=cfg,
                )
        steps = client.get(f"/api/v1/runs/{ATTACHED}/steps").json()["steps"]
        assert [(s["step"], s["quiesce_hold_ms"], s["commit_duration_ms"]) for s in steps] == [
            (2, 229.0, 13.4),
            (1, 231.5, 12.6),
        ]
