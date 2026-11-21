"""Cooperative cancellation  (foundation 6.3; ADR-006, ADR-007).

Two properties carry the weight, and both fail **silently**:

* the exception must be a ``BaseException`` subclass, or the training loop's own
  ``except Exception`` swallows an operator stop and the run keeps going;
* the flag must be read in a **fresh transaction**, or the loop sees the snapshot from
  before the stop was written and never stops at all. F6 calls that mutation M-2.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from core.config import reset_settings
from core.errors import Refusal
from tests.support.jobs import (
    insert_run,
    memory_engine,
    read_run,
    session_for,
    session_scope_factory,
)
from workers.cancellation import (
    CancellationToken,
    RunCancelled,
    request_stop,
    stop_requested,
)
from workers.run_record import STATE_COMPLETED, STATE_RUNNING


@pytest.fixture
def engine():
    reset_settings()
    return memory_engine()


class TestTheExceptionType:
    def test_run_cancelled_is_a_baseexception_and_not_an_exception(self):
        """**The property, not the class name.** A training step is wrapped in
        ``except Exception`` in a dozen places - a provider call, a tokenizer, a file
        write. Any one of them would swallow a cancellation derived from ``Exception``,
        turning an operator stop into a warning and a run that keeps training."""
        assert issubclass(RunCancelled, BaseException)
        assert not issubclass(RunCancelled, Exception)

    def test_except_exception_does_not_swallow_it(self):
        """The property asserted the way the loop will actually meet it."""
        with pytest.raises(RunCancelled):
            try:
                raise RunCancelled("r1")
            except Exception:
                pytest.fail("except Exception must not catch a cancellation")

    def test_it_names_the_run_it_cancels(self):
        """Reading the run from ambient state is how the wrong row gets moved."""
        exc = RunCancelled("r1")
        assert exc.run_id == "r1"
        assert exc.reason == "operator_stop"


class TestTheFlag:
    def test_a_fresh_run_is_not_cancelled(self, engine):
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)
            assert stop_requested(session, "r1") is False

    def test_requesting_a_stop_sets_the_flag(self, engine):
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)
            assert request_stop(session, "r1") is True
            assert read_run(session, "r1")["cancel_requested_at"] is not None

    def test_a_stop_on_a_terminal_run_is_refused_by_predicate(self, engine):
        """Reporting it as accepted would leave the operator waiting for a transition that
        will never come."""
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_COMPLETED)
            assert request_stop(session, "r1") is False
            assert read_run(session, "r1")["cancel_requested_at"] is None

    def test_an_absent_run_refuses_by_name(self, engine):
        with session_for(engine) as session:
            with pytest.raises(Refusal) as caught:
                stop_requested(session, "nope")
            assert caught.value.code == "run_not_found"


class TestTheTokenReadsFresh:
    """**Mutation M-2's target.** The read must open its own transaction."""

    def test_the_token_sees_a_stop_written_after_it_was_constructed(self, engine):
        """The realistic sequence: the worker builds its token, starts stepping, and the
        API writes the flag minutes later. A token holding one long transaction would
        never see it - no error, no log line, just a run that does not stop."""
        scope = session_scope_factory(engine)
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)

        token = CancellationToken("r1", session_scope=scope)
        assert token.is_cancelled() is False

        with session_for(engine) as session:  # a different transaction entirely
            request_stop(session, "r1")

        assert token.is_cancelled() is True

    def test_check_raises_once_the_flag_is_set(self, engine):
        scope = session_scope_factory(engine)
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)
            request_stop(session, "r1")

        token = CancellationToken("r1", session_scope=scope)
        with pytest.raises(RunCancelled) as caught:
            token.check()
        assert caught.value.run_id == "r1"

    def test_check_is_silent_while_no_stop_is_requested(self, engine):
        scope = session_scope_factory(engine)
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)
        CancellationToken("r1", session_scope=scope).check()

    def test_every_check_opens_a_transaction(self, engine):
        """The call count, not merely 'it read something'. A token that cached the answer
        would pass the test above on its first call and never see a later stop."""
        opened: list[int] = []
        inner = session_scope_factory(engine)

        def counting_scope():
            opened.append(1)
            return inner()

        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)

        token = CancellationToken("r1", session_scope=counting_scope)
        token.is_cancelled()
        token.is_cancelled()
        token.is_cancelled()
        assert len(opened) == 3
        assert token.checks == 3


class TestTheFirstRequestTimeIsPreserved:
    """**A surviving mutation: a documented property with no test behind it.**

    :func:`workers.cancellation.request_stop` says it *"preserves the first request's
    timestamp"*, and the sweep for this module found that removing the
    ``cancel_requested_at IS NULL`` predicate - so a second press overwrites the first -
    left the suite **green**. The property was asserted in prose and nowhere else, which is
    the same shape as a requirement claiming a field its schema does not have.

    It matters because that timestamp is the **only** record of how long the worker took to
    notice. An operator who presses stop and then presses it again - which is exactly what
    someone does when a run has not stopped yet - would reset the clock, and a worker that
    ignored the request for ten minutes would then be indistinguishable from one that
    stopped promptly. The number destroyed is the number that would have diagnosed the bug.
    """

    def test_a_second_stop_does_not_move_the_recorded_time(self, engine):
        from workers.cancellation import stop_requested_at
        from workers.run_record import utcnow

        first_press = utcnow() - timedelta(minutes=10)
        second_press = utcnow()

        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)
            assert request_stop(session, "r1", now=first_press) is True
            recorded = stop_requested_at(session, "r1")
            assert recorded is not None

            # The operator presses stop again, ten minutes later, because nothing happened.
            assert request_stop(session, "r1", now=second_press) is True
            again = stop_requested_at(session, "r1")

        assert again == recorded, (
            "the second press moved the recorded request time, destroying the only measure "
            "of how long the worker took to notice"
        )
        assert (
            second_press - again
        ).total_seconds() > 500, "the preserved time must still be the FIRST press, not the second"

    def test_a_second_stop_still_reports_a_stop_is_pending(self, engine):
        """Idempotent, not refused. Returning ``False`` on the second press would read as
        'your stop was rejected' to the one caller who most needs reassurance."""
        with session_for(engine) as session:
            insert_run(session, "r1", status=STATE_RUNNING)
            assert request_stop(session, "r1") is True
            assert request_stop(session, "r1") is True
            assert stop_requested(session, "r1") is True
