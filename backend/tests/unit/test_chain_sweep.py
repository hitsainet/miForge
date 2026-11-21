"""One advance path, and a double start is impossible  (FR-006.45, E-18; AC-006.32, .33).

## Two sessions, on a FILE-BACKED database, one engine each

The suite's in-memory ``StaticPool`` shares **one connection**, so a second session cannot
open a transaction while the first holds one - the double-start test would pass there *for
an infrastructure reason* and would look exactly like a working control. F6 hit that in its
own loop work. So the two sessions here come from two engines over the database
:func:`tests.support.db.database_url` names, which is a file on SQLite and the real thing
under ``MIFORGE_TEST_DATABASE_URL``.

## The window the guard has to close

Two sweeps race by **acting on a state that was true when they read it**. So the loser here
is given a stage row it read *before* the winner claimed it - which is precisely what a
check-then-insert loses on, and precisely what a conditional ``UPDATE ... WHERE run_id IS
NULL`` does not: the test is in the ``WHERE`` clause, evaluated by the database under the
row lock, not in a ``SELECT`` this process hoped was still true.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from services.chains import service as chain_service
from services.chains.tables import CHAINS, STAGES
from tests.support import checkpoints as ckpt
from tests.support.db import database_url, migrated_engine, session_for
from tests.unit.test_chain_service import _set_terminal, _stage, spec, spies
from workers.run_record import RUNS, utcnow


@pytest.fixture
def engine(tmp_path: Path):
    built = migrated_engine(tmp_path / "db")
    yield built
    built.dispose()


@pytest.fixture
def logs(tmp_path: Path) -> Path:
    return tmp_path / "logs"


def second_engine(tmp_path: Path):
    """A **second** engine over the same database - a second process, in effect."""
    from db.session import make_engine

    return make_engine(database_url(tmp_path / "db"))


def scope_factory(engine: Any) -> Any:
    def scope() -> Any:
        return session_for(engine)

    return scope


class TestTwoSweepsCannotDoubleStartAStage:
    def test_the_loser_is_told_already_started_and_submits_nothing(
        self, engine, logs, tmp_path
    ) -> None:
        """E-18, and the assertion is on the **run count**, not on the return value.

        A second run row for one stage would be two runs training the same curriculum
        position on the same card - both writing checkpoints under different ids, and the
        chain able to name only one of them.
        """
        other = second_engine(tmp_path)
        try:
            with session_for(engine) as session:
                chain_id = str(
                    chain_service.create(session, spec(), log_root=logs, **spies())["chain_id"]
                )

            # The loser's read. Taken BEFORE the winner claims, which is the window.
            with session_for(other) as loser_read:
                stale = _stage(loser_read, chain_id, 1)

            with session_for(engine) as session:
                previous = str(_stage(session, chain_id, 0).run_id)
                ckpt.write_chain(tmp_path / "ck", run_id=previous, steps=(100,))
                _set_terminal(session, previous, stop_reason="step_count")
                winner = chain_service.advance_one(
                    session,
                    chain_id,
                    log_root=logs,
                    checkpoint_root=tmp_path / "ck",
                    **spies(),
                )
            assert winner["action"] == "started"

            with session_for(other) as session:
                claimed = chain_service._claim_stage(
                    session, int(stale.id), "a-second-run-id", utcnow()
                )
            assert claimed is False, (
                "the conditional UPDATE's row count is the guard; a check-then-insert would "
                "have passed here because the stale read said the stage was unstarted"
            )

            with session_for(engine) as session:
                runs = session.execute(
                    sa.select(RUNS).where(RUNS.c.chain_id == chain_id, RUNS.c.stage_index == 1)
                ).all()
                assert len(runs) == 1
                assert runs[0].id == winner["run_id"]
                assert session.execute(sa.select(sa.func.count()).select_from(RUNS)).scalar() == 2
        finally:
            other.dispose()

    def test_two_sweeps_back_to_back_advance_once(self, engine, logs, tmp_path) -> None:
        """The shape beat actually produces: the schedule fires again before anything else
        changed."""
        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(session, spec(), log_root=logs, **spies())["chain_id"]
            )
            previous = str(_stage(session, chain_id, 0).run_id)
            ckpt.write_chain(tmp_path / "ck", run_id=previous, steps=(100,))
            _set_terminal(session, previous, stop_reason="step_count")

        first = chain_service.sweep(
            scope_factory(engine), log_root=logs, checkpoint_root=tmp_path / "ck", **spies()
        )
        second = chain_service.sweep(
            scope_factory(engine), log_root=logs, checkpoint_root=tmp_path / "ck", **spies()
        )
        assert first["advanced"] == 1
        assert second["advanced"] == 0
        assert second["actions"][0]["reason"] == "stage_running"
        with session_for(engine) as session:
            assert session.execute(sa.select(sa.func.count()).select_from(RUNS)).scalar() == 2

    def test_the_unique_constraint_makes_the_stage_plan_single_valued(self, engine, logs) -> None:
        """``uq_mf_chain_stages_chain_stage``, tested for what it **does** do.

        Task 7.7 names this constraint as the double-start guard. It cannot be: every
        stage's row already exists when a sweep arrives, because ``definition`` is
        ``NOT NULL`` and the plan is durable at creation (FR-006.39), so no insert happens
        at start time and nothing can raise. What it does guarantee is that the *plan* is
        single-valued - a second row for one position is impossible - and that is a real
        property worth pinning. The start-time guard is the conditional ``UPDATE`` above.

        *An unreachable ``except IntegrityError`` branch is deliberately not written in the
        service: a guard that cannot fire reads as protection while protecting nothing.*
        """
        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(session, spec(), log_root=logs, **spies())["chain_id"]
            )
        with session_for(engine) as session:
            with pytest.raises(IntegrityError):
                session.execute(
                    sa.insert(STAGES).values(
                        chain_id=chain_id,
                        stage_index=1,
                        definition={"run_type": "change"},
                        advance_on=["step_count"],
                    )
                )
            session.rollback()


class TestAChainSurvivesAWorkerRestart:
    def test_a_fresh_sweeper_resumes_at_the_stage_that_was_running(
        self, engine, logs, tmp_path
    ) -> None:
        """AC-006.32. **Chain progress lives in the database**, nowhere else.

        The sweep below runs with no memory of anything: new sessions, new call, and the
        only input is the rows. It must leave a running stage alone and then advance it when
        it ends - which is the whole of what "survives a restart" means when the record is
        the protocol (ADR-007).
        """
        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(session, spec(), log_root=logs, **spies())["chain_id"]
            )
            previous = str(_stage(session, chain_id, 0).run_id)
            ckpt.write_chain(tmp_path / "ck", run_id=previous, steps=(100,))
            _set_terminal(session, previous, stop_reason="step_count")
        chain_service.sweep(
            scope_factory(engine), log_root=logs, checkpoint_root=tmp_path / "ck", **spies()
        )

        # "The worker dies here." Nothing is carried over but the rows.
        with session_for(engine) as session:
            assert session.execute(sa.select(CHAINS)).one().current_stage == 1
            running = str(_stage(session, chain_id, 1).run_id)

        report = chain_service.sweep(
            scope_factory(engine), log_root=logs, checkpoint_root=tmp_path / "ck", **spies()
        )
        assert report["advanced"] == 0 and report["actions"][0]["run_id"] == running

        with session_for(engine) as session:
            ckpt.write_chain(tmp_path / "ck", run_id=running, steps=(100,))
            _set_terminal(session, running, stop_reason="step_count")
        report = chain_service.sweep(
            scope_factory(engine), log_root=logs, checkpoint_root=tmp_path / "ck", **spies()
        )
        assert report["advanced"] == 1
        with session_for(engine) as session:
            assert session.execute(sa.select(CHAINS)).one().current_stage == 2

    def test_an_unstarted_current_stage_is_started_by_the_sweep(
        self, engine, logs, tmp_path
    ) -> None:
        """The recovery case: the rows say stage 1 is current and it has no run.

        Reachable if a process died between the chain row moving and the stage being
        claimed. The sweep starts what the rows say is current rather than skipping it - a
        chain stuck with a current stage nobody starts is the stall FR-006.45 exists to
        prevent.
        """
        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(session, spec(), log_root=logs, **spies())["chain_id"]
            )
            previous = str(_stage(session, chain_id, 0).run_id)
            ckpt.write_chain(tmp_path / "ck", run_id=previous, steps=(100,))
            session.execute(
                sa.update(CHAINS).where(CHAINS.c.id == chain_id).values(current_stage=1)
            )
        report = chain_service.sweep(
            scope_factory(engine), log_root=logs, checkpoint_root=tmp_path / "ck", **spies()
        )
        assert report["advanced"] == 1
        assert report["actions"][0]["reason"] == "stage_was_unstarted"
        with session_for(engine) as session:
            assert _stage(session, chain_id, 1).run_id is not None


class TestTheSweepIsIsolatedPerChain:
    def test_one_broken_chain_does_not_take_the_sweep_with_it(self, engine, logs, tmp_path) -> None:
        """``services/reindex.py``'s recorded lesson, applied to a sweep.

        A chain whose plan is missing the stage it says it is on refuses. The next chain
        must still advance - *one subsystem must not kill the whole operation* - and the
        broken chain must be left **non-terminal**, because an unexpected error is not
        evidence that a curriculum should be discarded.
        """
        with session_for(engine) as session:
            broken = str(
                chain_service.create(session, spec(), log_root=logs, **spies())["chain_id"]
            )
            healthy = str(
                chain_service.create(session, spec(), log_root=logs, **spies())["chain_id"]
            )
            session.execute(
                sa.delete(STAGES).where(STAGES.c.chain_id == broken, STAGES.c.stage_index == 0)
            )
            session.execute(sa.update(CHAINS).where(CHAINS.c.id == broken).values(current_stage=0))
            run = str(_stage(session, healthy, 0).run_id)
            ckpt.write_chain(tmp_path / "ck", run_id=run, steps=(100,))
            _set_terminal(session, run, stop_reason="step_count")

        report = chain_service.sweep(
            scope_factory(engine), log_root=logs, checkpoint_root=tmp_path / "ck", **spies()
        )
        assert report["errors"] == 1
        assert report["advanced"] == 1
        with session_for(engine) as session:
            rows = {
                str(row.id): str(row.status) for row in session.execute(sa.select(CHAINS)).all()
            }
            assert rows[broken] not in {"completed", "halted"}
            assert rows[healthy] == "running"


class TestTheChainCompletes:
    def test_the_last_stage_advancing_completes_the_chain(self, engine, logs, tmp_path) -> None:
        """``completed``, not ``halted``, and ``finished_at`` set - which
        ``chain_terminal_iff_finished`` requires."""
        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(session, spec(stages=2), log_root=logs, **spies())["chain_id"]
            )
        for index in (0, 1):
            with session_for(engine) as session:
                run = str(_stage(session, chain_id, index).run_id)
                ckpt.write_chain(tmp_path / "ck", run_id=run, steps=(100,))
                _set_terminal(session, run, stop_reason="step_count")
            chain_service.sweep(
                scope_factory(engine), log_root=logs, checkpoint_root=tmp_path / "ck", **spies()
            )
        with session_for(engine) as session:
            chain = session.execute(sa.select(CHAINS)).one()
            assert chain.status == "completed"
            assert chain.finished_at is not None
            assert chain.halt_code is None
            assert _stage(session, chain_id, 1).finished_at is not None
