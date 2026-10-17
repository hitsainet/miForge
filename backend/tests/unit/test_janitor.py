"""The janitor  (foundation 6.5; ADR-007).

The thing it is judged on is the **distinction between lapsed and slow**, and both
mistakes are silent in opposite directions: reaping a slow run kills live work and reports
it as a crash (miStudio's 5.8-hour job), while never reaping anything leaves immortal rows
holding the node's only GPU.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from core.config import reset_settings
from tests.support.jobs import insert_run, memory_engine, read_run, session_for
from workers.janitor import sweep
from workers.run_record import (
    STATE_CANCELLED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_PENDING,
    STATE_RUNNING,
    utcnow,
)


@pytest.fixture
def engine():
    reset_settings()
    return memory_engine()


class TestLapsedVersusSlow:
    def test_a_lapsed_run_is_reaped_as_worker_lost(self, engine):
        with session_for(engine) as session:
            insert_run(
                session, "dead", status=STATE_RUNNING, heartbeat_at=utcnow() - timedelta(hours=2)
            )
            report = sweep(session, lapse_seconds=900)
            assert report.reaped == ["dead"]
            row = read_run(session, "dead")
            assert row["status"] == STATE_FAILED
            assert row["failure_code"] == "worker_lost"

    def test_a_slow_run_is_not_reaped(self, engine):
        """**Step staleness is not an input to this decision and must not become one.**

        This row's heartbeat is twenty seconds old and it has recorded no step for an hour:
        a generation-bound test pass, or a 2.4 GB save. Reaping it would reproduce
        miStudio's failure with the sign flipped.
        """
        with session_for(engine) as session:
            insert_run(
                session,
                "slow",
                status=STATE_RUNNING,
                heartbeat_at=utcnow() - timedelta(seconds=20),
            )
            report = sweep(session, lapse_seconds=900)
        assert report.reaped == []
        assert report.alive == ["slow"]

    def test_a_run_exactly_at_the_threshold_is_alive(self, engine):
        """Strictly greater. A boundary that reaps at exactly the threshold reaps a run
        whose next write is in flight."""
        with session_for(engine) as session:
            insert_run(
                session,
                "edge",
                status=STATE_RUNNING,
                heartbeat_at=utcnow() - timedelta(seconds=899),
            )
            assert sweep(session, lapse_seconds=900).reaped == []


class TestWhatItRefusesToTouch:
    def test_a_queued_run_is_never_reaped_however_old(self, engine):
        """A queued row has never heartbeated *by definition* - it is waiting for a worker.
        Reaping on heartbeat age would reap every run that waited out the threshold in the
        queue, which is the opposite of this module's job."""
        with session_for(engine) as session:
            insert_run(session, "waiting", status=STATE_PENDING, heartbeat_at=None)
            report = sweep(session, lapse_seconds=1)
        assert report.reaped == []
        assert report.running_without_heartbeat == []

    @pytest.mark.parametrize("terminal", [STATE_COMPLETED, STATE_FAILED, STATE_CANCELLED])
    def test_a_terminal_run_is_never_touched(self, engine, terminal):
        with session_for(engine) as session:
            insert_run(session, "done", status=terminal, heartbeat_at=utcnow() - timedelta(days=7))
            report = sweep(session, lapse_seconds=1)
            assert report.reaped == []
            assert read_run(session, "done")["status"] == terminal
            assert read_run(session, "done")["failure_code"] is None

    def test_a_running_row_with_no_heartbeat_is_reported_not_reaped(self, engine):
        """**Reported, never reaped, and the refusal is named.**

        A worker that died a millisecond after claiming the run looks exactly like a worker
        two seconds into its first model load. Guessing either way is wrong, so the sweep
        reports it: reaping on a guess kills live runs at startup, and ignoring it silently
        hides an immortal row. `Heartbeat.start()` at loop entry is what keeps this empty.
        """
        with session_for(engine) as session:
            insert_run(session, "ambiguous", status=STATE_RUNNING, heartbeat_at=None)
            report = sweep(session, lapse_seconds=1)
        assert report.reaped == []
        assert report.running_without_heartbeat == ["ambiguous"]
        assert report.alive == []


class TestTheReport:
    def test_it_reports_the_threshold_it_used(self, engine):
        """The question asked after an incident is 'why did it reap that' - unanswerable
        from a list of kills alone."""
        with session_for(engine) as session:
            report = sweep(session, lapse_seconds=123)
        assert report.lapse_threshold_seconds == 123
        assert report.swept_at is not None

    def test_a_mixed_population_is_partitioned_completely(self, engine):
        with session_for(engine) as session:
            insert_run(
                session, "dead", status=STATE_RUNNING, heartbeat_at=utcnow() - timedelta(hours=1)
            )
            insert_run(session, "live", status=STATE_RUNNING, heartbeat_at=utcnow())
            insert_run(session, "new", status=STATE_RUNNING, heartbeat_at=None)
            insert_run(session, "queued", status=STATE_PENDING)
            insert_run(session, "done", status=STATE_COMPLETED)
            report = sweep(session, lapse_seconds=900)

        assert report.reaped == ["dead"]
        assert report.alive == ["live"]
        assert report.running_without_heartbeat == ["new"]
        # Every `running` row is accounted for in exactly one bucket.
        assert len(report.reaped) + len(report.alive) + len(report.running_without_heartbeat) == 3

    def test_the_threshold_defaults_to_settings(self, engine):
        from core.config import get_settings

        with session_for(engine) as session:
            report = sweep(session)
        assert report.lapse_threshold_seconds == get_settings().janitor_lapse_seconds


class TestTheReapRace:
    """**A surviving mutation, and the regression test for it.**

    The mutation sweep for this module found that removing the terminal predicate from the
    reap ``UPDATE`` - ``.where(RUNS.c.status.notin_(TERMINAL_STATES))`` - left the suite
    **green**. Every other terminal-row test passes because the candidate ``SELECT``
    already filters to ``running``, so the predicate inside the ``UPDATE`` is never
    exercised by them: they prove the janitor does not *choose* a terminal row, not that it
    cannot *write* one.

    What the predicate actually protects is the window between the ``SELECT`` and the
    ``UPDATE``. A run that completes in that window is a completed run the janitor is about
    to mark ``failed`` / ``worker_lost`` - **a lie that survives in the record**, on a run
    that finished correctly, and one that cannot be distinguished afterwards from a real
    crash. The window is small and the event is not rare: it is what every run does.

    So the race is planted rather than reasoned about. A ``before_cursor_execute`` hook
    commits the completion on the same connection the instant the reap statement is about
    to run, which is exactly what a concurrent transaction would have done.
    """

    def test_a_run_that_completes_between_the_select_and_the_update_is_not_overwritten(
        self, engine
    ):
        from sqlalchemy import event

        with session_for(engine) as session:
            insert_run(
                session,
                "racer",
                status=STATE_RUNNING,
                heartbeat_at=utcnow() - timedelta(hours=1),
            )

        fired: list[str] = []

        @event.listens_for(engine, "before_cursor_execute")
        def _complete_it_first(conn, cursor, statement, parameters, context, executemany):
            if fired or "UPDATE mf_runs" not in statement:
                return
            fired.append(statement)
            # Another transaction commits the run's own completion, right here.
            cursor.execute(
                "UPDATE mf_runs SET status = 'completed', finished_at = CURRENT_TIMESTAMP "
                "WHERE id = 'racer'"
            )

        try:
            with session_for(engine) as session:
                report = sweep(session, lapse_seconds=900)
        finally:
            event.remove(engine, "before_cursor_execute", _complete_it_first)

        assert fired, "the race was not planted; the hook never saw the reap statement"
        assert report.reaped == [], "a run that completed mid-sweep must not be reaped"

        with session_for(engine) as session:
            row = read_run(session, "racer")
        assert row["status"] == STATE_COMPLETED, "the completion must survive the sweep"
        assert row["failure_code"] is None, (
            "a completed run marked worker_lost is a lie that survives in the record, and "
            "is indistinguishable afterwards from a real crash"
        )
