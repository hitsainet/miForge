"""Stopping a ``pending`` run must actually stop it  (FR-006.29, E-8; ADR-016 #1).

``services.stop_service.stop_queued_run`` - *"Take a ``pending`` run terminal without ever
allocating (E-8)"* - names its own callers in its docstring: *"the janitor sweep, or the
request itself when the row is still ``pending``."* **Neither called it.** It had two test
references and no production caller, found by ``0xcc/scripts/check_reachability.py``.

**What that cost an operator.** Every worker is ``--pool=solo -c 1`` (ADR-006), so on a
one-card node a submitted run sits ``pending`` behind whatever is training. Cancelling it
recorded ``cancel_requested_at`` and returned ``202`` - and cooperative cancellation happens
**at a step boundary** (ADR-007), of which a run with no worker has none. So the row stayed
``pending`` with a stop outstanding until the janitor reaped it, and the operator's own
queued run finally surfaced as *reaped / worker lost* rather than *stopped* - the wrong
cause of death on the record, and a GPU lease held until its expiry rather than released.

**The existing coverage pinned the half-fix rather than preventing it.**
``test_stop_service_refusals.py::test_a_pending_run_is_stoppable_because_a_queue_must_be
_cancellable`` asserted ``stop_pending is True`` - that the *flag* was written - which is
true of the broken behaviour too. Asserting the flag is not asserting the transition.

**This is the control.** Remove the ``stop_queued_run`` call from ``request_stop`` and both
tests below go red.
"""

from __future__ import annotations

import pytest

from services import stop_service
from tests.support.jobs import insert_run, memory_engine, read_run, session_for
from workers.run_record import TERMINAL_STATES


@pytest.fixture
def engine():  # type: ignore[no-untyped-def]
    return memory_engine()


class TestAPendingRunIsTakenTerminalByTheRequestItself:
    def test_the_row_moves_terminal_and_does_not_merely_flag(self, engine) -> None:  # type: ignore[no-untyped-def]
        """The transition, not the flag. There is no step boundary to honour it later."""
        with session_for(engine) as session:
            insert_run(session, "run-queued", status="pending")
            session.commit()
            outcome = stop_service.request_stop(session, "run-queued")
            session.commit()

        assert outcome["stopped_now"] is True, (
            "a pending run has no worker and no step boundary, so the request must perform "
            "the stop rather than record an intention nothing will ever act on"
        )
        with session_for(engine) as session:
            row = read_run(session, "run-queued")
        assert row["status"] in TERMINAL_STATES, (
            f"run is {row['status']!r} after an operator stop. It has no worker, so nothing "
            f"will move it; it will be reaped as 'worker lost' and the record will name the "
            f"wrong cause of death."
        )
        assert (
            row["finished_at"] is not None
        ), "a terminal row with no finished_at cannot say when it stopped"

    def test_a_running_run_is_left_to_its_step_boundary(self, engine) -> None:  # type: ignore[no-untyped-def]
        """The other direction, and it matters as much.

        A running run holds parameters and an optimizer state. Taking its row terminal from
        under it would strand the worker mid-step writing to a row the API says is finished,
        and lose the checkpoint the boundary was about to take. ``stopped_now`` must be
        ``False`` and the status unchanged: cooperative cancellation at a step boundary is
        the whole mechanism (ADR-007), and a stop that bypassed it would be a second,
        racing, stop path.
        """
        with session_for(engine) as session:
            insert_run(session, "run-live", status="running")
            session.commit()
            outcome = stop_service.request_stop(session, "run-live")
            session.commit()

        assert outcome["stopped_now"] is False
        assert outcome["stop_pending"] is True
        with session_for(engine) as session:
            row = read_run(session, "run-live")
        assert row["status"] == "running", (
            "a running run must be stopped by its own loop at a step boundary, never by the "
            "request reaching past it"
        )
