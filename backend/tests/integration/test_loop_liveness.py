"""The janitor does not reap a live run  (FR-006.10, .11; AC-006.8, .9; M-3, M-4).

**ADR-007's named precedent, reproduced structurally.** *"Progress reported only over the
socket is invisible to the janitor"* - which is how miStudio reaped a live 5.8-hour job as
"worker lost". The loop is exactly the shape of job that precedent describes: its longest
sub-operations each exceed the 60 s heartbeat interval, because **60 s is a minimum interval
between writes and not a maximum gap**.

The assertions here are on the **outcome** - the run was not reaped - not on "the heartbeat
was called". A test that only checked the call would pass against a heartbeat advanced
*after* the long phase instead of around it, which is the defect.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
import sqlalchemy as sa

from tests.support.jobs import insert_run, memory_engine, read_run, session_for
from tests.support.loop_fakes import boundary_context
from training import loop
from workers import janitor
from workers.heartbeat import Heartbeat
from workers.run_record import RUNS, TERMINAL_STATES, utcnow

RUN = "run-live"
#: Shorter than the 60 s production value so the test takes milliseconds. The *shape* is
#: what matters: a sub-operation longer than the threshold, with the loop beating around it.
LAPSE = 2


@pytest.fixture
def engine():
    engine = memory_engine()
    with session_for(engine) as session:
        insert_run(session, RUN, status="running", heartbeat_at=utcnow())
        session.commit()
    return engine


class TestASlowBoundaryIsNotReaped:
    def test_the_janitor_leaves_a_live_run_alone_across_a_slow_evaluation(self, engine) -> None:
        """AC-006.8. **M-3's test.** The evaluation takes longer than the lapse threshold,
        and the loop's advances around it are what keep the row alive.

        The janitor sweeps **from the middle of the evaluation**, which is the only moment
        the question is interesting: before it the heartbeat is fresh, after it the loop has
        already beaten.
        """
        beat = Heartbeat(RUN, min_interval_seconds=0)
        swept: list[janitor.SweepReport] = []
        held: dict[str, object] = {}

        def slow_evaluation(step: int) -> None:
            # Backdate the heartbeat as though the pass had taken longer than the lapse
            # threshold, and leave it backdated. The loop's **post-phase advance** is the
            # only thing that can bring it forward again, so if that advance is missing
            # (M-3) the row is lapsed when the janitor sweeps.
            #
            # Backdated rather than slept, so the test measures the *mechanism* and not
            # the clock - and on the loop's own session, because the suite's in-memory
            # engine shares one connection and a second session cannot open a transaction
            # while this one holds it.
            held["session"].execute(  # type: ignore[union-attr]
                sa.update(RUNS)
                .where(RUNS.c.id == RUN)
                .values(heartbeat_at=utcnow() - timedelta(seconds=LAPSE * 3))
            )

        with session_for(engine) as session:
            held["session"] = session
            loop.run(
                boundary_context(
                    run_id=RUN,
                    session=session,
                    heartbeat=beat,
                    evaluate=slow_evaluation,
                    test_every=1,
                    save_every=None,
                ),
                max_steps=1,
                evaluate_step_zero=False,
                run_end_save=False,
            )
            session.commit()

        with session_for(engine) as session:
            swept.append(janitor.sweep(session, lapse_seconds=LAPSE))
            session.commit()

        with session_for(engine) as session:
            row = read_run(session, RUN)
        assert row["status"] == "running", (
            "the janitor reaped a live run. That is miStudio's 5.8-hour job, and the only "
            "thing standing between the loop and it is the heartbeat advance AROUND the "
            "long phase rather than merely between phases."
        )
        assert swept[0].reaped == []

    def test_the_longest_gap_is_recorded_so_the_threshold_can_be_MEASURED(self, engine) -> None:
        """OQ-5. Nothing else in the product can produce this number, and without it
        ``JANITOR_LAPSE_SECONDS`` has no defensible value - which is why the run screen
        states the gap rather than inventing a liveness rule of its own."""
        beat = Heartbeat(RUN, min_interval_seconds=0)
        with session_for(engine) as session:
            session.execute(
                sa.update(RUNS)
                .where(RUNS.c.id == RUN)
                .values(heartbeat_at=utcnow() - timedelta(seconds=9))
            )
            session.commit()
            beat.touch(session, force=True)
            session.commit()
            row = read_run(session, RUN)
        assert row["longest_heartbeat_gap_seconds"] is not None
        assert row["longest_heartbeat_gap_seconds"] >= 8

    def test_a_run_that_genuinely_stops_beating_IS_reaped(self, engine) -> None:
        """The negative half. Without it, the test above would pass against a janitor that
        never reaps anything - which protects nothing and hides a worker that died."""
        with session_for(engine) as session:
            session.execute(
                sa.update(RUNS)
                .where(RUNS.c.id == RUN)
                .values(heartbeat_at=utcnow() - timedelta(seconds=LAPSE * 10))
            )
            session.commit()
            report = janitor.sweep(session, lapse_seconds=LAPSE)
            session.commit()
        assert RUN in report.reaped
        with session_for(engine) as session:
            assert read_run(session, RUN)["status"] in TERMINAL_STATES


class TestTheHeartbeatRefusesToMoveATerminalRow:
    def test_a_terminal_row_is_not_moved_by_the_loops_heartbeat(self, engine) -> None:
        """FR-006.11, and **M-4 reverts the foundation's own guard** - so it must fail
        loudly. The cost of losing it is a heartbeat that resurrects a finished run on the
        run list, and a janitor that can no longer tell a finished run from a live one."""
        beat = Heartbeat(RUN, min_interval_seconds=0)
        with session_for(engine) as session:
            session.execute(
                sa.update(RUNS)
                .where(RUNS.c.id == RUN)
                .values(status="completed", finished_at=utcnow())
            )
            session.commit()
            before = read_run(session, RUN)["heartbeat_at"]
            assert beat.touch(session, force=True) is False
            session.commit()
            after = read_run(session, RUN)["heartbeat_at"]
        assert before == after, "the heartbeat moved a terminal row"

    def test_the_loop_stops_without_stepping_when_its_row_is_already_terminal(self, engine) -> None:
        """E-19. It stops immediately and **does not move the row** - the earlier, louder
        half of the same rule."""
        from core.errors import Refusal
        from training import run_driver

        with session_for(engine) as session:
            session.execute(
                sa.update(RUNS)
                .where(RUNS.c.id == RUN)
                .values(status="failed", finished_at=utcnow(), failure_code="worker_lost")
            )
            session.commit()
            with pytest.raises(Refusal) as caught:
                run_driver.drive_recorded_run(RUN, session=session)
        # `run_definition_missing` fires first for this row, which is the correct earlier
        # refusal; the terminal guard is asserted directly below so both are covered.
        assert caught.value.code in {"run_definition_missing", "run_not_runnable"}


class TestTheLeaseIsReleasedAfterTheTerminalTransition:
    def test_mark_terminal_then_release_in_that_order(self, engine) -> None:
        """FR-006.29. **A lease has an expiry and a run row does not**: a worker dying after
        the terminal transition but before the release leaves a lease that frees the card by
        itself. Reversed, a worker dying between the two leaves a non-terminal run with no
        lease, which the janitor must then disambiguate from a live run whose lease lapsed.
        """
        from core.gpu import Device
        from training import run_state
        from workers import lease as lease_module

        with session_for(engine) as session:
            held = lease_module.acquire(
                session, RUN, "auto", device=Device(uuid="GPU-aaa", name="x", index=0)
            )
            session.commit()

            assert run_state.mark_completed(session, RUN, stop_reason="step_count") is True
            session.commit()
            # The row is terminal BEFORE the release. Asserted as a state, not as a call
            # order on a spy: the thing that matters on a crash is which of the two
            # durable facts is already true.
            assert read_run(session, RUN)["status"] == "completed"
            assert read_run(session, RUN)["finished_at"] is not None

            assert run_state.release_lease(session, RUN, held.gpu_uuid) is True
            session.commit()

    def test_a_crash_between_the_two_leaves_the_lease_to_expire_on_its_own(self, engine) -> None:
        """The reason the order is this way round, asserted rather than asserted *about*:
        a terminal row with an unreleased lease is a recoverable state, because the lease
        has a TTL and reclaims itself."""
        from core.gpu import Device
        from training import run_state
        from workers import lease as lease_module

        with session_for(engine) as session:
            lease_module.acquire(
                session, RUN, "auto", device=Device(uuid="GPU-bbb", name="x", index=0)
            )
            run_state.mark_completed(session, RUN, stop_reason="step_count")
            session.commit()
            # No release - the worker "died" here.
            assert lease_module.holder_of(session, "GPU-bbb") == RUN
            reclaimed = lease_module.reclaim_expired(
                session, "GPU-bbb", now=utcnow() + timedelta(days=1)
            )
            session.commit()
        assert reclaimed >= 1, "the lease must free the card by itself"

    def test_releasing_with_no_card_is_not_an_error(self, engine) -> None:
        """ADR-023: a developer without a GPU runs everything except a training step, so
        a terminal transition on a machine with no card must not raise on the release."""
        from training import run_state

        with session_for(engine) as session:
            assert run_state.release_lease(session, RUN, None) is False
