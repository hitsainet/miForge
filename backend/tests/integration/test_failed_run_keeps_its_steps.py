"""A failed run's step index is rebuilt from its run log, not rolled back to nothing.

Found live 2026-10-18 (F11 window 2): run f11w2-r3 trained 650 steps, failed on a save
timeout, and kept none of them in ``mf_steps``. The index rows sat in the loop's
uncommitted transaction (they commit with the heartbeat), and the failure path's rollback
took them all. The log is written first and holds every step (ADR-004), so
``_finalize_failed`` rebuilds the index from it.

Driven through ``drive_recorded_run`` with the loop replaced only by one that records
steps the way the real one does (through ``StepWriter``, uncommitted) and then raises the
refusal a real run raised.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa

from core.config import reset_settings
from core.errors import Refusal, refuse
from models.run import Step
from tests.support.jobs import insert_run, memory_engine, read_run, session_for
from training import run_driver, tasks
from training.run_config import RunConfig
from training.step import StepOutcome
from training.step_record import StepWriter

from .test_run_parts_reach_the_source import RESOLVED, SUBMITTED

RUN = SUBMITTED["run_id"]
CFG = RunConfig(learning_rate=2e-5, warmup_steps=5, batch_size=4, gradient_accumulation=2)


class _Source:
    def __init__(self, context) -> None:  # type: ignore[no-untyped-def]
        self.context = context

    def preflight(self, spec) -> None:  # type: ignore[no-untyped-def]
        return None


@pytest.fixture
def failed_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from training import signals

    monkeypatch.setenv("TRAINER_STATE_ROOT", str(tmp_path / "trainer-state"))
    reset_settings()
    signals.ensure_registered()
    monkeypatch.setattr(signals.registry, "get", lambda name: _Source)

    def steps_then_boom(**kwargs):  # type: ignore[no-untyped-def]
        writer = StepWriter(run_id=RUN, log_root=tasks.run_log_root(kwargs.get("log_root")))
        for step in (1, 2, 3):
            writer.write(
                session=kwargs["session"],  # uncommitted, as in the loop
                step=step,
                outcome=StepOutcome(
                    loss=0.5,
                    objective="sft",
                    grad_norm=1.0,
                    wall_time_ms=1.0,
                    batch_sha256="f" * 64,
                ),
                cfg=CFG,
            )
        raise refuse("weights_surface_unreachable", "save timed out after 10 s")

    monkeypatch.setattr(tasks, "drive_run", steps_then_boom)
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
    yield engine
    reset_settings()


def test_the_failed_runs_steps_are_in_the_index(failed_engine) -> None:  # type: ignore[no-untyped-def]
    with session_for(failed_engine) as session:
        row = read_run(session, RUN)
        assert row["status"] == "failed"
        # The run row too: GET /api/v1/runs (the list) reads only this column.
        assert row["last_step"] == 3
        steps = (
            session.execute(sa.select(Step.__table__.c.step).where(Step.__table__.c.run_id == RUN))
            .scalars()
            .all()
        )
    assert sorted(steps) == [1, 2, 3]
