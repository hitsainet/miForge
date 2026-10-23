"""The stop reaches the worker - **from a second session**  (FR-006.24, .25; AC-006.18/19).

## Why the second session is the whole test

Every worker is ``--pool=solo -c 1`` (ADR-006). ``revoke(terminate=True)`` signals a pool
child and the solo pool has none, so the call returns successfully and nothing happens - the
operator sees a cancellation that was *accepted*. A busy solo worker never reads the control
queue either. So the stop is a column on the ``mf_runs`` row (ADR-007: *the run record is the
protocol*).

**"Through the database" is necessary and not sufficient.** A worker holding one long-lived
transaction for the run's duration reads a snapshot taken before the operator pressed stop,
and then sees the flag **never** - for hours, with no error on any surface.

So **the flag is set from a different session in every test here**. A fixture that updates
the row through the loop's own session agrees with the loop by construction and **passes
against the broken line** - which is exactly the fixture failure ADR-016 warns about, and
why F6's own task list says *"the control is worthless without this"*.

**Mutation M-2** deletes the new-session / commit-and-expire before the read.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.jobs import (
    insert_run,
    session_for,
    session_scope_factory,
)
from training import loop, stop
from training.stop.base import StopContext

RUN = "run-stop-2s"


@pytest.fixture
def engine(tmp_path):
    """**A file-backed SQLite in WAL mode, with a connection per session.**

    Not the suite's shared in-memory ``StaticPool`` engine, and the reason is the whole
    point of this file. That engine shares **one** connection, so a second session cannot
    even open a transaction while the loop holds one - SQLite answers *"cannot start a
    transaction within a transaction"*. On it the mutation M-2 breaks is **not
    expressible**, so a test written against it would pass against the broken line for an
    infrastructure reason and look like a control.

    WAL mode with a real file gives what is needed: a writer concurrent with a reader, and
    a reader that sees a **consistent snapshot from the start of its read transaction**.
    That snapshot is exactly what the loop's long-lived transaction would be in
    PostgreSQL, and exactly what the fresh read has to defeat.
    """
    import sqlalchemy as sa
    from sqlalchemy import event

    import models  # noqa: F401 - imported for its side effect on Base.metadata
    from db.base import Base
    from db.session import make_engine

    url = f"sqlite:///{tmp_path / 'f6.db'}"

    def build():
        built = make_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(built, "connect")
        def _wal(dbapi_connection, _record):
            dbapi_connection.isolation_level = None
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

        @event.listens_for(built, "begin")
        def _begin(connection):
            connection.exec_driver_sql("BEGIN")

        return built

    writer = build()
    Base.metadata.create_all(writer)
    with session_for(writer) as session:
        insert_run(session, RUN, status="running")
        session.commit()
    # A SECOND engine, so the loop's session and the API's are two connections. One engine
    # would pool them and could hand back the same connection, which is the same defect
    # the in-memory StaticPool has.
    loop_engine = build()
    yield _Engines(loop=loop_engine, other=writer)
    loop_engine.dispose()
    writer.dispose()
    del sa


class _Engines:
    """The loop's engine and the API's, kept apart so a test cannot accidentally share one."""

    def __init__(self, loop, other) -> None:
        self.loop = loop
        self.other = other


def _request_stop_from_elsewhere(engine: _Engines) -> None:
    """Set the flag through a **different connection**, as the API handler does.

    This is the point of the whole file. Done through
    :func:`workers.cancellation.request_stop` rather than a hand-written UPDATE, so the
    column, the timestamp and the collapse rule are the production ones - *a fixture whose
    fields agree with the code by construction is how a suite stays green over a real bug.*
    """
    from workers.cancellation import request_stop

    with session_for(engine.other) as other:
        request_stop(other, RUN)
        other.commit()


class TestTheConditionSeesAFlagSetElsewhere:
    def test_a_fresh_session_scope_sees_the_stop(self, engine) -> None:
        """**M-2's target.** The loop's own session is held open across the write."""
        scope = session_scope_factory(engine.loop)
        with session_for(engine.loop) as held:
            # The loop reads something first, so its transaction is genuinely open and its
            # identity map genuinely populated - which is the state the mutation survives in.
            from workers.cancellation import stop_requested

            assert stop_requested(held, RUN) is False

            _request_stop_from_elsewhere(engine)

            ctx = StopContext(run_id=RUN, step=7, round=0, session=held, session_scope=scope)
            verdict = stop.resolve("operator_stop").evaluate(ctx, {})

        assert verdict.triggered is True, (
            "the operator-stop condition did not see a flag written by another session. "
            "That is the failure where the operator presses stop, the API returns 202, "
            "the row is updated, and the run carries on for hours with no error anywhere."
        )
        assert verdict.values["requested_at"] is not None

    def test_the_scope_branch_is_ACTUALLY_TAKEN_when_a_scope_is_given(self, engine) -> None:
        """**A test finding from F6's own sweep, written before the fix was trusted.**

        Mutation **M-2a** - disabling the ``session_scope`` branch - **SURVIVED** the first
        sweep, and for a benign reason: the fallback path does the same two defeats
        (``commit`` then ``expire_all``), so the condition still saw the flag and every
        test stayed green. The branch was therefore *unprotected*: a later edit could
        delete it, and nothing in the suite would notice that the production path had
        silently changed from "a new connection" to "reuse the loop's, having committed
        it".

        That difference is not cosmetic. Committing the loop's session mid-boundary ends a
        transaction the loop may be relying on, and on PostgreSQL it releases every lock
        the loop holds. The new-session path is the intended one, and this asserts it is
        the one taken - on the **call count**, because "a scope was provided" passes
        against a scope nobody called.
        """
        calls: list[int] = []
        scope = session_scope_factory(engine.loop)

        def counting_scope():
            calls.append(1)
            return scope()

        _request_stop_from_elsewhere(engine)
        with session_for(engine.loop) as held:
            ctx = StopContext(
                run_id=RUN, step=7, round=0, session=held, session_scope=counting_scope
            )
            verdict = stop.resolve("operator_stop").evaluate(ctx, {})

        assert verdict.triggered is True
        assert calls == [1], (
            "the operator-stop condition must read through the supplied session scope "
            "exactly once per boundary; with 0 it fell through to committing the loop's "
            "own session, which is the fallback and not the intended path"
        )

    def test_the_fallback_path_commits_and_expires_the_held_session(self, engine) -> None:
        """With no ``session_scope``, the loop's own session is used - **after** a commit
        and an ``expire_all``.

        Both calls are load-bearing and for different reasons: the commit ends the
        long-lived transaction so the next statement opens a new one, and ``expire_all``
        empties the identity map so the read goes to the database rather than to the object
        already loaded. *"Re-query" is not enough for either.*
        """
        with session_for(engine.loop) as held:
            from workers.cancellation import stop_requested

            assert stop_requested(held, RUN) is False
            _request_stop_from_elsewhere(engine)

            ctx = StopContext(run_id=RUN, step=7, round=0, session=held)
            verdict = stop.resolve("operator_stop").evaluate(ctx, {})
        assert verdict.triggered is True

    def test_with_NEITHER_a_session_nor_a_scope_it_refuses_rather_than_answering_no(
        self,
    ) -> None:
        """**Fail closed.** A condition that cannot read its flag and reports "no stop
        requested" is a stop button that does nothing, and a visible refusal is recoverable
        where a silently wrong answer is not (ADR-013)."""
        ctx = StopContext(run_id=RUN, step=1, round=0)
        with pytest.raises(Refusal) as caught:
            stop.resolve("operator_stop").evaluate(ctx, {})
        assert caught.value.code == "operator_stop_unreadable"


class TestTheLoopStopsAtTheNextBoundary:
    def test_the_current_step_completes_and_is_recorded_never_half(self, engine) -> None:
        """AC-006.18/19, ADR-007: the step **either completes and is recorded, or had not
        started** - never half. The flag is set after step 2, and step 3 never runs."""
        from tests.support.loop_fakes import FakeStep, boundary_context

        scope = session_scope_factory(engine.loop)
        step = FakeStep()
        fired_at: list[int] = []

        def take_step(n: int) -> object:
            result = step(n)
            if n == 2:
                _request_stop_from_elsewhere(engine)
                fired_at.append(n)
            return result

        with session_for(engine.loop) as held:
            result = loop.run(
                boundary_context(
                    run_id=RUN,
                    take_step=take_step,
                    session=held,
                    session_scope=scope,
                    stop_conditions={"operator_stop": {}},
                    save_every=None,
                ),
                max_steps=10,
                run_end_save=False,
            )

        assert fired_at == [2]
        assert step.calls == [1, 2], "step 3 must not have started"
        assert result.stopped_because == "operator_stop"
        assert result.last_step == 2

    def test_a_stop_requested_before_the_run_starts_is_honoured_without_allocating(
        self, engine
    ) -> None:
        """E-8. Refusing it would leave an operator unable to cancel a queue - and on a
        one-card node that is the queue for the only card there is."""
        from services import stop_service
        from workers.run_record import TERMINAL_STATES

        with session_for(engine.other) as session:
            import sqlalchemy as sa

            from workers.run_record import RUNS

            session.execute(sa.update(RUNS).where(RUNS.c.id == RUN).values(status="pending"))
            session.commit()

        _request_stop_from_elsewhere(engine)
        with session_for(engine.other) as session:
            assert stop_service.stop_queued_run(session, RUN) is True
            session.commit()
        with session_for(engine.other) as session:
            from tests.support.jobs import read_run

            row = read_run(session, RUN)
        assert row["status"] in TERMINAL_STATES
        assert row["finished_at"] is not None


class TestNoCeleryControlMessageIsEverSent:
    def test_revoke_appears_nowhere_under_F6(self) -> None:
        """FR-006.24, ADR-006. It signals a pool child and solo has none, so the call
        succeeds and nothing happens - the worst available failure mode, because the
        operator sees a cancellation that was accepted."""
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[3] / "backend" / "src"
        f6 = [
            src / "training" / "loop.py",
            src / "training" / "boundary.py",
            src / "training" / "run_state.py",
            src / "training" / "run_driver.py",
            src / "services" / "stop_service.py",
            src / "services" / "run_launch.py",
            src / "services" / "run_definition.py",
            src / "api" / "v1" / "endpoints" / "runs_loop.py",
            *sorted((src / "training" / "stop").rglob("*.py")),
            *sorted((src / "training" / "refresh").rglob("*.py")),
        ]
        for path in f6:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    assert node.func.attr != "revoke", f"{path.name} sends a control message"
                if isinstance(node, ast.keyword):
                    assert node.arg != "terminate", f"{path.name} names terminate="
