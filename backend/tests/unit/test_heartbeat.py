"""The database heartbeat  (foundation 6.4, 6.4a, 6.4b; ADR-007).

Every test here protects a failure that is **silent**. A heartbeat that does not advance
raises nothing, logs nothing unusual and shows a perfectly normal loss curve - and the run
dies looking like a crash one janitor interval later. A heartbeat that moves a terminal row
raises nothing either; the run list simply shows a finished run as alive.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
import sqlalchemy as sa

from core.config import get_settings, reset_settings
from core.errors import Refusal
from tests.support.jobs import RUNS, insert_run, memory_engine, read_run, session_for
from workers.heartbeat import Heartbeat, liveness_of
from workers.janitor import sweep
from workers.run_record import (
    STATE_CANCELLED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_RUNNING,
    as_aware,
    utcnow,
)


@pytest.fixture
def engine():
    reset_settings()
    return memory_engine()


class TestTheThrottle:
    def test_first_touch_writes(self, engine):
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)
            assert Heartbeat("r1").touch(session) is True
            assert read_run(session, "r1")["heartbeat_at"] is not None

    def test_a_second_touch_inside_the_interval_is_suppressed(self, engine):
        """60 s is a MINIMUM INTERVAL BETWEEN WRITES. One UPDATE per step on a loop doing
        hundreds of steps a minute is write amplification for no information."""
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)
            beat = Heartbeat("r1")
            assert beat.touch(session) is True
            assert beat.touch(session) is False
            assert beat.writes == 1

    def test_force_bypasses_the_throttle(self, engine):
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)
            beat = Heartbeat("r1")
            beat.touch(session)
            assert beat.touch(session, force=True) is True
            assert beat.writes == 2

    def test_the_interval_comes_from_settings_and_defaults_to_sixty(self, engine):
        assert get_settings().heartbeat_min_interval_seconds == 60
        assert Heartbeat("r1").min_interval_seconds == 60


class TestTheTerminalRowGuard:
    """ADR-007: the heartbeat *refuses to move a terminal row*. F6's mutation M-4."""

    @pytest.mark.parametrize("terminal", [STATE_COMPLETED, STATE_FAILED, STATE_CANCELLED])
    def test_it_refuses_to_move_a_terminal_row(self, engine, terminal):
        with session_for(engine) as session:
            insert_run(session, "r1", status=terminal)
            assert Heartbeat("r1").touch(session) is False
            assert read_run(session, "r1")["heartbeat_at"] is None
            assert read_run(session, "r1")["longest_heartbeat_gap_seconds"] is None

    def test_a_run_that_completes_mid_loop_stops_moving(self, engine):
        """The realistic shape: a worker that has not yet noticed its run finished."""
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)
            beat = Heartbeat("r1", min_interval_seconds=0)
            assert beat.touch(session) is True
            first = read_run(session, "r1")["heartbeat_at"]

            # Through the real table, so the `terminal_iff_finished` constraint applies.
            session.execute(
                sa.update(RUNS)
                .where(RUNS.c.id == "r1")
                .values(status=STATE_COMPLETED, finished_at=utcnow())
            )
            assert beat.touch(session) is False
            assert read_run(session, "r1")["heartbeat_at"] == first

    def test_an_absent_row_refuses_by_name_rather_than_reporting_success(self, engine):
        """An absent row is a wrong identifier; a terminal row is a finished run. Two
        different facts, two different answers - collapsing them lets a typo'd run_id look
        like a completed run."""
        with session_for(engine) as session:
            with pytest.raises(Refusal) as caught:
                Heartbeat("nope").touch(session)
            assert caught.value.code == "run_not_found"


class TestTheLongestGap:
    """Foundation 6.4b. Nothing else in the product can produce this number, and until it
    exists the janitor's lapse threshold has no defensible value."""

    def test_the_longest_gap_is_recorded_in_milliseconds(self, engine):
        with session_for(engine) as session:
            start = utcnow() - timedelta(seconds=300)
            insert_run(session, "r1", status=STATE_RUNNING, heartbeat_at=start)
            assert Heartbeat("r1").touch(session) is True
            gap = read_run(session, "r1")["longest_heartbeat_gap_seconds"]
            assert 299 <= gap <= 301

    def test_a_shorter_later_gap_does_not_lower_the_record(self, engine):
        """The record is a maximum. A CASE in the UPDATE keeps it in one statement; a
        read-compute-write would lose a concurrent writer's larger value."""
        with session_for(engine) as session:
            insert_run(
                session,
                "r1",
                status=STATE_RUNNING,
                heartbeat_at=utcnow() - timedelta(seconds=1),
                longest_heartbeat_gap_seconds=500,
            )
            Heartbeat("r1").touch(session)
            assert read_run(session, "r1")["longest_heartbeat_gap_seconds"] == 500

    def test_the_measurement_survives_a_restarted_heartbeat_object(self, engine):
        """Derived from the persisted column, not from in-process memory - so a worker that
        restarted extends the same record instead of starting a new one."""
        with session_for(engine) as session:
            insert_run(
                session,
                "r1",
                status=STATE_RUNNING,
                heartbeat_at=utcnow() - timedelta(seconds=120),
            )
            Heartbeat("r1").touch(session)  # a brand new object, no memory of the last write
            assert read_run(session, "r1")["longest_heartbeat_gap_seconds"] >= 119


class TestRunIdAttribution:
    def test_touching_for_another_run_is_refused_not_honoured(self, engine):
        """F6 spells the call `heartbeat.touch(run_id)`. It is checked, not honoured -
        writing the other row would attribute one run's liveness to a different one."""
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)
            insert_run(session, "r2", status=STATE_RUNNING)
            with pytest.raises(Refusal) as caught:
                Heartbeat("r1").touch(session, "r2")
            assert caught.value.code == "heartbeat_run_id_mismatch"
            assert read_run(session, "r2")["heartbeat_at"] is None

    def test_touching_for_its_own_run_positionally_is_accepted(self, engine):
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)
            assert Heartbeat("r1").touch(session, "r1") is True


class TestTheMiStudioFailure:
    """**The named precedent, reproduced structurally.**

    ADR-007: *"progress reported only over the socket is invisible to the janitor,"* which
    is how miStudio reaped a live 5.8-hour job as "worker lost." A run can emit a thousand
    socket events; if it does not write the column, the janitor reaps it - and is right to,
    because it cannot see the socket at all.
    """

    def test_a_run_reporting_only_over_the_socket_is_reaped(self, engine):
        emitted: list[dict] = []

        with session_for(engine) as session:
            insert_run(
                session,
                "socket-only",
                status=STATE_RUNNING,
                heartbeat_at=utcnow() - timedelta(hours=5, minutes=48),
            )
            # The run is alive and busy - it has been "reporting progress" the whole time.
            for step in range(1000):
                emitted.append({"run_id": "socket-only", "step": step})

            report = sweep(session, lapse_seconds=900)

        assert len(emitted) == 1000, "the run was emitting the whole time"
        assert report.reaped == ["socket-only"]
        with session_for(engine) as session:
            assert read_run(session, "socket-only")["status"] == STATE_FAILED
            assert read_run(session, "socket-only")["failure_code"] == "worker_lost"
            # The check constraint `terminal_iff_finished` makes this not optional.
            assert read_run(session, "socket-only")["finished_at"] is not None

    def test_the_same_run_writing_the_column_is_not_reaped(self, engine):
        """The fix is on the writer's side, not the janitor's: the loop writes the column."""
        with session_for(engine) as session:
            insert_run(
                session,
                "heartbeating",
                status=STATE_RUNNING,
                heartbeat_at=utcnow() - timedelta(hours=5, minutes=48),
            )
            Heartbeat("heartbeating").touch(session)
            report = sweep(session, lapse_seconds=900)

        assert report.reaped == []
        assert report.alive == ["heartbeating"]


class TestLivenessPayload:
    """Foundation 6.4a's publish half. F9's finding F-15: both alternatives to this block
    were written and withdrawn."""

    def test_every_key_is_present_even_before_the_first_beat(self, engine):
        """A missing key is indistinguishable from a healthy one, which is the failure this
        block exists to prevent."""
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)
            block = liveness_of(read_run(session, "r1"))

        assert set(block) == {
            "heartbeat_at",
            "age_seconds",
            "longest_gap_seconds",
            "lapse_threshold_seconds",
            "lapsed",
        }
        assert block["heartbeat_at"] is None
        assert block["age_seconds"] is None
        # None, not 0. A run that has never beaten is not a fresh run, and a run with no
        # observed gap is not a run whose longest gap was nothing.
        assert block["longest_gap_seconds"] is None
        assert block["lapsed"] is False

    def test_it_publishes_the_janitors_threshold_not_the_heartbeat_interval(self, engine):
        """**The defect this project caught once already.** 60 s is a minimum interval
        between writes, not a maximum gap: a threshold derived from it reports a healthy
        run as stalled through every test pass and every save. The client must read the
        janitor's threshold, so the server publishes that one."""
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING, heartbeat_at=utcnow())
            block = liveness_of(read_run(session, "r1"))

        settings = get_settings()
        assert block["lapse_threshold_seconds"] == settings.janitor_lapse_seconds
        assert block["lapse_threshold_seconds"] != settings.heartbeat_min_interval_seconds

    def test_a_healthy_long_save_is_not_lapsed(self, engine):
        """Three minutes without a write is a 2.4 GB save, not a dead worker. Under a
        threshold derived from 60 s this run would read as stalled."""
        with session_for(engine) as session:
            insert_run(
                session,
                "r1",
                status=STATE_RUNNING,
                heartbeat_at=utcnow() - timedelta(seconds=180),
            )
            block = liveness_of(read_run(session, "r1"))
        assert block["age_seconds"] > 60
        assert block["lapsed"] is False

    def test_a_gap_beyond_the_janitors_threshold_is_lapsed(self, engine):
        with session_for(engine) as session:
            insert_run(
                session,
                "r1",
                status=STATE_RUNNING,
                heartbeat_at=utcnow() - timedelta(seconds=5000),
            )
            block = liveness_of(read_run(session, "r1"))
        assert block["lapsed"] is True

    def test_a_terminal_run_is_finished_not_stalled(self, engine):
        """A completed run's heartbeat is *expected* to be old. Reporting it as lapsed is
        the same category of false positive as the 60 s threshold."""
        with session_for(engine) as session:
            insert_run(
                session,
                "r1",
                status=STATE_COMPLETED,
                heartbeat_at=utcnow() - timedelta(days=3),
            )
            block = liveness_of(read_run(session, "r1"))
        assert block["lapsed"] is False
        assert block["age_seconds"] > 0

    def test_the_block_round_trips_the_real_writer(self, engine):
        """ADR-016: prefer a round trip through the real writer and reader to a hand-built
        dict. The writer here is `Heartbeat.touch`, not an inserted value."""
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)
            beat = Heartbeat("r1", min_interval_seconds=0)
            beat.touch(session)
            beat.touch(session)
            block = liveness_of(read_run(session, "r1"))

        assert beat.writes == 2
        assert block["age_seconds"] is not None
        # Two writes under a second apart: a gap of 0 is now a FACT, where before the
        # first write it was absent. The two are different and the column keeps them so.
        assert block["longest_gap_seconds"] == 0
        assert as_aware(beat.last_write_at) is not None


class TestStartAtLoopEntry:
    def test_start_writes_unconditionally(self, engine):
        """Not an optimisation. A `running` row with a NULL heartbeat is ambiguous and the
        janitor declines to reap it; calling start() first is what keeps that case empty."""
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)
            beat = Heartbeat("r1")
            assert beat.start(session) is True
            assert beat.start(session) is True
            assert beat.writes == 2


class TestTheWriteRace:
    """**A second surviving mutation, of the same class as the janitor's.**

    The sweep for this module found that *replacing* the ``UPDATE``'s terminal predicate
    with a ``if status in TERMINAL_STATES: return False`` **before** the statement left the
    suite green. Every test in :class:`TestTheTerminalRowGuard` passes against that
    rewrite, because each one makes the row terminal and *then* calls ``touch`` - so the
    pre-check answers them all. They prove the heartbeat does not move a row it *knows* is
    terminal; they say nothing about the window between knowing and writing.

    That window is the whole point of putting the predicate in the ``WHERE`` clause. The
    run it races with is a run finishing, which is not a rare event - it is what every run
    does exactly once - and the loss is the same shape as the janitor's: a finished run
    shown as alive on the run list, with a heartbeat that keeps advancing.

    Planted the same way, with a hook that commits the completion on the same connection
    as the write is about to go out.
    """

    def test_a_run_that_completes_between_the_read_and_the_write_is_not_resurrected(self, engine):
        from sqlalchemy import event

        with session_for(engine) as session:
            insert_run(session, "racer", status=STATE_RUNNING)

        fired: list[str] = []

        @event.listens_for(engine, "before_cursor_execute")
        def _complete_it_first(conn, cursor, statement, parameters, context, executemany):
            if fired or not statement.lstrip().upper().startswith("UPDATE MF_RUNS"):
                return
            fired.append(statement)
            cursor.execute(
                "UPDATE mf_runs SET status = 'completed', finished_at = CURRENT_TIMESTAMP "
                "WHERE id = 'racer'"
            )

        try:
            with session_for(engine) as session:
                wrote = Heartbeat("racer").touch(session)
        finally:
            event.remove(engine, "before_cursor_execute", _complete_it_first)

        assert fired, "the race was not planted; the hook never saw the heartbeat write"
        assert wrote is False, "the write must be refused by the statement, not merely skipped"

        with session_for(engine) as session:
            row = read_run(session, "racer")
        assert row["status"] == STATE_COMPLETED
        assert row["heartbeat_at"] is None, (
            "a finished run whose heartbeat keeps advancing shows as alive on the run list "
            "forever; the predicate must live in the UPDATE, not in a prior read"
        )
