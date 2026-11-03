"""A completed run's row carries its ``last_step`` - the column ``GET /api/v1/runs`` reads.

Found live 2026-10-18: the run dropdown's list showed ``last_step: null`` for every run,
including f11w2-r7 at step 1,000, because ``mf_runs.last_step`` had no production writer.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests.support.jobs import insert_run, memory_engine, read_run, session_for
from training import run_driver

RUN = "run-completed-last-step"


def test_the_terminal_transition_writes_last_step() -> None:
    engine = memory_engine()
    with session_for(engine) as session:
        insert_run(session, RUN, status="running")
    result = SimpleNamespace(
        stopped_because="step_count",
        stop_detail="total reached",
        all_fired=["step_count"],
        steps_completed=12,
        last_step=12,
        rounds=1,
        checkpoints=[],
        trainer_states=[],
    )
    with session_for(engine) as session:
        run_driver._finalize(session, RUN, result, emit=None)  # type: ignore[arg-type]
    with session_for(engine) as session:
        row = read_run(session, RUN)
    assert (row["status"], row["last_step"]) == ("completed", 12)
