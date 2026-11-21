"""``request_stop``: the terminal refusal and the idempotent collapse  (FR-006.24, FR-006.26).

**Why this file exists.** `grep -rn run_not_stoppable tests/` was **empty**. E-7 — a stop
against a run that has already finished — is reachable from two production call sites
(`POST /api/v1/runs/{run_id}/cancel` and `services/chains/service.py`), its refusal message
is three sentences of operator guidance, and **nothing asserted any of it**. Only
`stop_queued_run` was covered. Found 2026-10-14 by an agent that grepped the tests rather
than reading that the task list said the work was done.

**What the refusal is for, and why "accept it quietly" is the wrong answer.** A success for
a stop that stopped nothing tells an operator they stopped a run that had already finished —
and on a one-card node the next thing they do is wait for a card that was never taken. So it
refuses by name and says what to do instead: continue from the last checkpoint, or branch.

**The collapse is the opposite case and must NOT refuse.** A second stop against a run that
is already stopping is the same request, and refusing it would make a double-click look like
an error. It collapses, and `collapsed` says which happened — the only way a caller can tell
"I stopped it" from "it was already stopping", which an operator watching a run that will not
die needs to know.

Controls (each run 2026-10-14, each seen red):
* delete the `if status in TERMINAL_STATES` guard → the terminal tests go red;
* make `collapsed` always `False` → the collapse test goes red;
* make `require_run` tolerate an absent row → the not-found test goes red.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from services import stop_service
from tests.support.jobs import insert_run, memory_engine, session_for
from workers.run_record import TERMINAL_STATES


@pytest.fixture
def engine():  # type: ignore[no-untyped-def]
    return memory_engine()


class TestATerminalRunIsRefusedByName:
    @pytest.mark.parametrize("status", sorted(TERMINAL_STATES))
    def test_every_terminal_state_refuses(self, engine, status: str) -> None:  # type: ignore[no-untyped-def]
        """Parametrised over the **live** set, not over a list written here.

        `TERMINAL_STATES` is imported and iterated, so a state added to it without a decision
        about stopping turns this red rather than passing silently — which is the shape of
        the `run_type` and `queued`/`pending` drifts this project has recorded twice.
        """
        with session_for(engine) as session:
            insert_run(session, "run-done", status=status)
            session.commit()
            with pytest.raises(Refusal) as caught:
                stop_service.request_stop(session, "run-done")
        assert caught.value.code == "run_not_stoppable"
        assert caught.value.details["status"] == status

    def test_the_message_says_what_to_do_next(self, engine) -> None:  # type: ignore[no-untyped-def]
        """ADR-021: a user-facing message says what to do next, not how to feel about it."""
        with session_for(engine) as session:
            insert_run(session, "run-done", status="completed")
            session.commit()
            with pytest.raises(Refusal) as caught:
                stop_service.request_stop(session, "run-done")
        message = caught.value.message
        assert "continue" in message.lower() and "branch" in message.lower()
        assert "already completed" in message

    def test_it_refuses_rather_than_recording_a_stop_nobody_can_act_on(
        self, engine
    ) -> None:  # type: ignore[no-untyped-def]
        """**The state is asserted after the refusal**, not just the exception.

        A guard that raised *after* writing would pass an exception-only test while leaving a
        stop request on a finished run — which the janitor would then read as a run that is
        stopping and never does.
        """
        with session_for(engine) as session:
            insert_run(session, "run-done", status="completed")
            session.commit()
            with pytest.raises(Refusal):
                stop_service.request_stop(session, "run-done")
            session.rollback()
            assert stop_service.stop_requested_at(session, "run-done") is None


class TestAnActiveRunIsStoppedAndTheSecondRequestCollapses:
    def test_the_first_request_is_recorded_and_pending(self, engine) -> None:  # type: ignore[no-untyped-def]
        with session_for(engine) as session:
            insert_run(session, "run-live", status="running")
            session.commit()
            result = stop_service.request_stop(session, "run-live", label="operator:sean")
            session.commit()
        assert result["stop_pending"] is True
        assert result["collapsed"] is False
        assert result["cancel_requested_at"] is not None

    def test_the_second_request_COLLAPSES_rather_than_refusing(self, engine) -> None:  # type: ignore[no-untyped-def]
        """A double-click is the same request, not an error."""
        with session_for(engine) as session:
            insert_run(session, "run-live", status="running")
            session.commit()
            first = stop_service.request_stop(session, "run-live")
            session.commit()
            second = stop_service.request_stop(session, "run-live")
            session.commit()
        assert first["collapsed"] is False
        assert second["collapsed"] is True, (
            "a repeated stop must be reported as collapsed, which is the only way a caller "
            "can tell 'I stopped it' from 'it was already stopping'"
        )
        assert second["cancel_requested_at"] == first["cancel_requested_at"], (
            "the collapse must keep the ORIGINAL timestamp - moving it would make a run that "
            "has been refusing to die for an hour look like it was asked a moment ago"
        )

    def test_a_pending_run_is_stoppable_because_a_queue_must_be_cancellable(
        self, engine
    ) -> None:  # type: ignore[no-untyped-def]
        """E-8: honoured, not refused. On a one-card node the queue is the queue for the only
        card there is, and an operator who cannot cancel it cannot use the machine."""
        with session_for(engine) as session:
            insert_run(session, "run-queued", status="pending")
            session.commit()
            result = stop_service.request_stop(session, "run-queued")
            session.commit()
        assert result["stop_pending"] is True


class TestAnAbsentRunIsNotReadAsStoppable:
    def test_it_refuses_run_not_found(self, engine) -> None:  # type: ignore[no-untyped-def]
        """Never ``run_not_stoppable``: the two are different answers and an operator who
        mistyped a run id must not be told the run has already finished."""
        with session_for(engine) as session, pytest.raises(Refusal) as caught:
            stop_service.request_stop(session, "no-such-run")
        assert caught.value.code == "run_not_found"
