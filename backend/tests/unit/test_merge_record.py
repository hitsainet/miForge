"""``mf_merges`` is written, and a row that lies cannot be stored  (F8 tasks 11.5, 12.5).

**This file exists because the writer did not have one.** ``services/merge_record.py``,
``mf_merges`` and its three CHECK constraints were all built, and
``grep -rln "merge_record\\|MergeOutcome" backend/tests`` matched **nothing**. So the table
that records an **irreversible** operation, and the constraints whose whole job is to make a
false row unstorable, were protected by reading alone - which this project's own memory says
is how a suite stays green over a real bug.

Two things were found by writing it rather than by reading it:

1. **``interrupted`` had no producer.** It is in the vocabulary, in the ``outcome_known``
   CHECK built from that vocabulary, and in three docstrings that explain what it means.
   ``grep -rn OUTCOME_INTERRUPTED backend/src`` matched the constant's definition, the CHECK
   that lists it, and nothing else. An outcome the schema admits and no path can write is
   the same unreachable capability as a function nothing calls, and it is the outcome an
   operator most needs on the day a merge was killed on the card.
2. **There was no cancellation point on the merge path at all** (task 12.5's second half),
   so the only cancellation available was ``revoke(terminate=True)``, which ADR-006 records
   as doing nothing at all under the solo pool while reporting success.

**The database here is the MIGRATED one, never ``create_all``**, and the constraints are
therefore the ones production applies. That matters more than usual for this table: SQLite
does enforce CHECK constraints but SQLAlchemy does not *reflect* them, so a test built from
the declaration would be testing the declaration against itself.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from core.errors import Refusal, refuse
from services import merge_record
from tests.support import db as support
from training.runtypes.adapters.merge import (
    OUTCOME_INTERRUPTED,
    OUTCOME_REFUSED,
    OUTCOME_WRITTEN,
    MergeOutcome,
)
from training.runtypes.tables import MERGES, OUTCOMES
from workers import merge_tasks
from workers.cancellation import RunCancelled

ADAPTER = "lfm2-1.2b:run-7:adapter-40"
BASE = "lfm2-1.2b:run-base:step-40"
MANIFEST_SHA = "a" * 64

#: What ``runtypes_service.preflight_merge`` hands the worker. Only the four members the
#: record is built from, so a test cannot pass by reading a field production does not.
ADMITTED: dict[str, Any] = {
    "name": ADAPTER,
    "step": 40,
    "base_checkpoint": BASE,
    "base_checkpoint_path": "/nowhere/step-000040",
}


@pytest.fixture(scope="module")
def engine(tmp_path_factory):  # type: ignore[no-untyped-def]
    return support.migrated_engine(tmp_path_factory.mktemp("f8-merges"))


@pytest.fixture
def session(engine):  # type: ignore[no-untyped-def]
    with support.session_for(engine) as handle:
        yield handle
        handle.execute(sa.delete(MERGES))
        handle.commit()


def written(**overrides: Any) -> MergeOutcome:
    fields: dict[str, Any] = {
        "outcome": OUTCOME_WRITTEN,
        "adapter_name": ADAPTER,
        "adapter_step": 40,
        "adapter_manifest_sha256": MANIFEST_SHA,
        "base_checkpoint_name": BASE,
        "result_checkpoint_name": "lfm2-1.2b:run-merge:step-41",
    }
    fields.update(overrides)
    return MergeOutcome(**fields)


def rows(session: Any) -> list[dict[str, Any]]:
    return [dict(row._mapping) for row in session.execute(sa.select(MERGES)).all()]


class TestTheRowIsTheOutcomeItWasGiven:
    """Field for field, through the real table. A spy asserting "a row was inserted" would
    pass against a row naming the wrong adapter, the wrong base or the wrong result - and
    every one of those is a lineage claim about an operation that cannot be undone."""

    def test_a_written_outcome_round_trips_every_field(self, session: Any) -> None:
        row_id = merge_record.record(session, written())
        session.commit()

        stored = rows(session)
        assert len(stored) == 1
        row = stored[0]
        assert row["id"] == row_id
        assert row["adapter_name"] == ADAPTER
        assert row["adapter_step"] == 40
        assert row["adapter_manifest_sha256"] == MANIFEST_SHA
        assert row["base_checkpoint_name"] == BASE
        assert row["result_checkpoint_name"] == "lfm2-1.2b:run-merge:step-41"
        assert row["outcome"] == OUTCOME_WRITTEN
        assert row["refusal_code"] is None
        assert row["created_at"] is not None

    def test_a_refused_outcome_carries_the_raised_code_and_no_checkpoint(
        self, session: Any
    ) -> None:
        merge_record.record(
            session,
            merge_record.refused(
                adapter_name=ADAPTER,
                adapter_step=40,
                adapter_manifest_sha256=MANIFEST_SHA,
                base_checkpoint_name=BASE,
                refusal_code="merge_refused_no_unmerged_base",
            ),
        )
        session.commit()

        row = rows(session)[0]
        assert row["outcome"] == OUTCOME_REFUSED
        assert row["refusal_code"] == "merge_refused_no_unmerged_base"
        assert row["result_checkpoint_name"] is None


class TestTheThirdOutcomeNowExists:
    """``interrupted`` is the half of task 11.5 that was owed. See the module docstring."""

    def test_the_builder_carries_no_code_and_no_checkpoint(self) -> None:
        outcome = merge_record.interrupted(
            adapter_name=ADAPTER,
            adapter_step=40,
            adapter_manifest_sha256=MANIFEST_SHA,
            base_checkpoint_name=BASE,
        )
        assert outcome.outcome == OUTCOME_INTERRUPTED
        assert outcome.refusal_code is None, (
            "`code_only_when_refused` permits a code only on a refused row, and the "
            "distinction is the point: refused means this system decided, interrupted "
            "means something outside it did"
        )
        assert outcome.result_checkpoint_name is None

    def test_it_stores(self, session: Any) -> None:
        merge_record.record(
            session,
            merge_record.interrupted(
                adapter_name=ADAPTER,
                adapter_step=40,
                adapter_manifest_sha256=MANIFEST_SHA,
                base_checkpoint_name=BASE,
            ),
        )
        session.commit()
        assert rows(session)[0]["outcome"] == OUTCOME_INTERRUPTED

    def test_the_vocabulary_has_exactly_the_three_R_02_45_allows(self) -> None:
        assert OUTCOMES == (OUTCOME_WRITTEN, OUTCOME_REFUSED, OUTCOME_INTERRUPTED)


class TestATableThatRefusesARowWhichLies:
    """**``written`` iff a result checkpoint, in BOTH directions**, which is the only place a
    row about an irreversible operation could lie. Each case is asserted against the
    migrated database, so dropping the constraint from the declaration is not enough to make
    these pass - the migration has to carry it too."""

    def test_written_with_no_result_checkpoint_is_refused(self, session: Any) -> None:
        """ "A merge happened" that cannot say to what."""
        with pytest.raises(IntegrityError):
            merge_record.record(session, written(result_checkpoint_name=None))
        session.rollback()

    @pytest.mark.parametrize("outcome", [OUTCOME_REFUSED, OUTCOME_INTERRUPTED])
    def test_a_non_written_row_carrying_a_result_checkpoint_is_refused(
        self, session: Any, outcome: str
    ) -> None:
        """The other direction: a merge that did not happen producing an artifact."""
        candidate = written(
            outcome=outcome,
            refusal_code="merge_refused_no_unmerged_base" if outcome == OUTCOME_REFUSED else None,
        )
        with pytest.raises(IntegrityError):
            merge_record.record(session, candidate)
        session.rollback()

    @pytest.mark.parametrize("outcome", [OUTCOME_WRITTEN, OUTCOME_INTERRUPTED])
    def test_a_refusal_code_on_a_row_that_did_not_refuse_is_refused(
        self, session: Any, outcome: str
    ) -> None:
        candidate = written(
            outcome=outcome,
            result_checkpoint_name=(
                "lfm2-1.2b:run-merge:step-41" if outcome == OUTCOME_WRITTEN else None
            ),
            refusal_code="merge_refused_no_unmerged_base",
        )
        with pytest.raises(IntegrityError):
            merge_record.record(session, candidate)
        session.rollback()

    def test_an_outcome_outside_the_three_is_refused(self, session: Any) -> None:
        with pytest.raises(IntegrityError):
            merge_record.record(session, written(outcome="cancelled"))
        session.rollback()

    def test_record_does_not_SWALLOW_the_integrity_error(self, session: Any) -> None:
        """A caught constraint violation would store nothing and report success, which is
        the silent half of the failure the constraint exists to make loud."""
        with pytest.raises(IntegrityError):
            merge_record.record(session, written(result_checkpoint_name=None))
        session.rollback()
        assert rows(session) == []


class TestTheTwoLookupsAnOperatorPerforms:
    def test_newest_first_and_filtered_by_adapter(self, session: Any) -> None:
        merge_record.record(session, written())
        merge_record.record(session, written(adapter_name="other:run-9:adapter-1"))
        session.commit()

        assert len(merge_record.listing(session)) == 2
        mine = merge_record.listing(session, adapter_name=ADAPTER)
        assert [row["adapter_name"] for row in mine] == [ADAPTER]

    def test_the_limit_is_honoured(self, session: Any) -> None:
        for index in range(3):
            merge_record.record(
                session, written(result_checkpoint_name=f"lfm2-1.2b:run-merge:step-{index}")
            )
        session.commit()
        assert len(merge_record.listing(session, limit=2)) == 2


# ---------------------------------------------------------------------------------------
# The worker: every outcome is recorded, exactly once, with the payload that happened.
# ---------------------------------------------------------------------------------------


@pytest.fixture
def worker(session: Any, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """``merge_adapter`` over the **real table**, with the card's work replaced.

    ``_perform`` is the only thing faked - it is the part that needs a GPU - so everything
    this file asserts about the record is produced by the production arms around it. The
    session is the migrated one, so the three CHECK constraints are live for every row the
    worker writes rather than only for the rows these tests insert by hand.
    """

    @contextmanager
    def scope():  # type: ignore[no-untyped-def]
        yield session

    import db.session as db_session

    monkeypatch.setattr(db_session, "session_scope", scope)
    monkeypatch.setattr("services.runtypes_service.preflight_merge", lambda _s, _n: dict(ADMITTED))
    monkeypatch.setattr(merge_tasks, "_manifest_sha256", lambda _s, _n: MANIFEST_SHA)
    return session


class TestTheWorkerRecordsTheOutcomeThatHappened:
    def test_a_written_merge_records_the_outcome_merge_returned(
        self, worker: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(merge_tasks, "_perform", lambda *a, **k: written())

        answer = merge_tasks.merge_adapter(ADAPTER, "run-merge", 41)

        assert answer["outcome"] == OUTCOME_WRITTEN
        stored = rows(worker)
        assert len(stored) == 1, "exactly one row per merge"
        assert stored[0]["result_checkpoint_name"] == "lfm2-1.2b:run-merge:step-41"

    def test_an_admitted_merge_refused_on_the_card_records_the_RAISED_code(
        self, worker: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: Any, **_k: Any) -> Any:
            raise refuse("ml_runtime_unavailable", "Install the ML extra to merge.")

        monkeypatch.setattr(merge_tasks, "_perform", boom)

        with pytest.raises(Refusal):
            merge_tasks.merge_adapter(ADAPTER, "run-merge", 41)

        stored = rows(worker)
        assert len(stored) == 1
        assert stored[0]["outcome"] == OUTCOME_REFUSED
        assert stored[0]["refusal_code"] == "ml_runtime_unavailable"
        assert stored[0]["result_checkpoint_name"] is None

    def test_a_merge_cancelled_before_the_write_records_INTERRUPTED_and_does_not_raise(
        self, worker: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The operator asked for it, so it is not a failure. A task that raised here
        would report a granted request as an error."""

        def cancelled(*_a: Any, **_k: Any) -> Any:
            raise RunCancelled("run-merge")

        monkeypatch.setattr(merge_tasks, "_perform", cancelled)

        answer = merge_tasks.merge_adapter(ADAPTER, "run-merge", 41)

        assert answer["outcome"] == OUTCOME_INTERRUPTED
        assert answer["result_checkpoint"] is None
        stored = rows(worker)
        assert len(stored) == 1
        assert stored[0]["outcome"] == OUTCOME_INTERRUPTED
        assert stored[0]["refusal_code"] is None

    def test_a_merge_that_CRASHES_records_interrupted_and_re_raises(
        self, worker: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**Without this arm a killed merge leaves no row at all**, which an operator
        cannot tell apart from a merge that was never enqueued - the exact
        indistinguishability the table was built to remove."""

        def crash(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(merge_tasks, "_perform", crash)
        monkeypatch.setattr(merge_tasks, "_result_checkpoint_on_disk", lambda *_a: None)

        with pytest.raises(RuntimeError):
            merge_tasks.merge_adapter(ADAPTER, "run-merge", 41)

        stored = rows(worker)
        assert len(stored) == 1
        assert stored[0]["outcome"] == OUTCOME_INTERRUPTED
        assert stored[0]["result_checkpoint_name"] is None

    def test_a_BaseException_is_recorded_too_not_only_an_Exception(
        self, worker: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A warm shutdown arrives as ``SystemExit``, which ``except Exception`` would miss
        - and the worker dying is the case the row is the only evidence of."""

        def shutdown(*_a: Any, **_k: Any) -> Any:
            raise SystemExit(0)

        monkeypatch.setattr(merge_tasks, "_perform", shutdown)
        monkeypatch.setattr(merge_tasks, "_result_checkpoint_on_disk", lambda *_a: None)

        with pytest.raises(SystemExit):
            merge_tasks.merge_adapter(ADAPTER, "run-merge", 41)

        assert rows(worker)[0]["outcome"] == OUTCOME_INTERRUPTED

    def test_an_interrupt_AFTER_the_atomic_rename_is_recorded_as_WRITTEN(
        self, worker: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The artifact is on disk and authoritative (ADR-004), so an ``interrupted`` row
        would deny a checkpoint that exists - which ``written_iff_result`` exists to make
        unstorable. The name comes from the manifest, never from a rendered guess."""

        def crash(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("killed between the rename and the return")

        monkeypatch.setattr(merge_tasks, "_perform", crash)
        monkeypatch.setattr(
            merge_tasks, "_result_checkpoint_on_disk", lambda *_a: "lfm2-1.2b:run-merge:step-41"
        )

        with pytest.raises(RuntimeError):
            merge_tasks.merge_adapter(ADAPTER, "run-merge", 41)

        stored = rows(worker)
        assert len(stored) == 1
        assert stored[0]["outcome"] == OUTCOME_WRITTEN
        assert stored[0]["result_checkpoint_name"] == "lfm2-1.2b:run-merge:step-41"

    def test_an_unreadable_result_records_NOTHING_rather_than_a_guess(
        self, worker: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both available rows would be false, so neither is written and the reason is
        logged by name. Refuse rather than guess (ADR-013)."""

        def crash(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("killed")

        def unreadable(*_a: Any) -> str:
            raise refuse("checkpoint_manifest_invalid", "It does not conform.")

        monkeypatch.setattr(merge_tasks, "_perform", crash)
        monkeypatch.setattr(merge_tasks, "_result_checkpoint_on_disk", unreadable)

        with pytest.raises(RuntimeError):
            merge_tasks.merge_adapter(ADAPTER, "run-merge", 41)

        assert rows(worker) == [], "no row is better than a row that is false"

    def test_a_PREFLIGHT_refusal_writes_no_row(
        self, worker: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R-02.45's three outcomes describe the merge, not a request for one. Deliberate,
        and asserted so it cannot be "fixed" into existence by a later session."""

        def no(*_a: Any, **_k: Any) -> Any:
            raise refuse("merge_refused_no_unmerged_base", "Save a checkpoint of the base.")

        monkeypatch.setattr("services.runtypes_service.preflight_merge", no)

        with pytest.raises(Refusal):
            merge_tasks.merge_adapter(ADAPTER, "run-merge", 41)

        assert rows(worker) == []


# ---------------------------------------------------------------------------------------
# Task 12.5: the cooperative cancellation point, and it is BEFORE the write.
# ---------------------------------------------------------------------------------------


class TestTheCancellationPointIsBeforeTheWrite:
    def test_perform_checks_cancellation_before_the_load_AND_before_the_merge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**The order is the requirement.** A check only before the load is seen before a
        multi-gigabyte acquisition that takes minutes, so a stop pressed during it would
        arrive after the irreversible step had already run. A check after the merge would
        not be a cancellation at all.

        Removing either ``_check_cancelled(run_id)`` from ``_perform`` turns this red.
        """
        order: list[str] = []
        monkeypatch.setattr(merge_tasks, "_check_cancelled", lambda _r: order.append("check"))
        monkeypatch.setattr(
            merge_tasks,
            "_load_base",
            lambda _d: order.append("load")
            or merge_tasks._LoadedBase(None, None, {"signal_source": "s"}),
        )

        import training.runtypes.adapters.merge as merge_module

        monkeypatch.setattr(merge_module, "merge", lambda **_k: order.append("merge") or written())

        merge_tasks._perform(
            None, adapter_name=ADAPTER, run_id="run-merge", step=41, admitted=dict(ADMITTED)
        )

        assert order == ["check", "load", "check", "merge"]

    def test_the_check_opens_its_OWN_transaction_and_so_sees_a_stop_written_elsewhere(
        self, engine: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F6's mutation M-2 in this file's terms. The task's session is opened before a
        load that takes minutes, so a read through it returns the snapshot from before the
        operator pressed stop - and the merge then runs anyway, irreversibly, with nothing
        in the logs saying why.
        """
        from tests.support import jobs

        with support.session_for(engine) as setup:
            jobs.insert_run(setup, "run-stopme", status="running")
            setup.commit()

        @contextmanager
        def fresh():  # type: ignore[no-untyped-def]
            with support.session_for(engine) as handle:
                yield handle

        import workers.run_record as run_record

        monkeypatch.setattr(run_record, "default_session_scope", fresh)

        # Not cancelled yet.
        merge_tasks._check_cancelled("run-stopme")

        # A SECOND session sets the flag - the only form of this test that can fail
        # against a read taken through the caller's own open transaction.
        with support.session_for(engine) as other:
            from workers.cancellation import request_stop

            request_stop(other, "run-stopme")
            other.commit()

        with pytest.raises(RunCancelled):
            merge_tasks._check_cancelled("run-stopme")

    def test_a_merge_whose_run_has_NO_ROW_is_reported_not_treated_as_not_cancelled(
        self, engine: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A merge is an operation and not a run (FR-006.55), and the route does not require
        the named run to exist. Where there is no row there is no ``cancel_requested_at`` to
        read, so there is **no cancellation channel for this merge at all** - which is a
        different fact from "no stop was requested" and is logged as one.
        """

        @contextmanager
        def fresh():  # type: ignore[no-untyped-def]
            with support.session_for(engine) as handle:
                yield handle

        import workers.run_record as run_record

        monkeypatch.setattr(run_record, "default_session_scope", fresh)

        with caplog.at_level("INFO"):
            merge_tasks._check_cancelled("no-such-run")

        codes = [getattr(record, "code", None) for record in caplog.records]
        assert "merge_cancellation_unavailable" in codes, (
            "the absence of a cancellation channel must be named, not silently read as a "
            "negative answer to a question nothing could have answered"
        )

    def test_a_refusal_that_is_not_run_not_found_is_NOT_swallowed(
        self, engine: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tolerance is for one named code only. Widening it to every ``Refusal`` would
        turn a broken read into "not cancelled" and run the merge anyway, irreversibly.

        **The first version of this test was vacuous and a mutation proved it.** It raised
        the refusal from the *context manager*, which is entered OUTSIDE the guarded block,
        so the exception propagated whatever the ``exc.code`` comparison said: widening the
        tolerance to every ``Refusal`` left the suite 28/28 green. The refusal has to come
        from the call the ``try`` actually wraps.
        """

        @contextmanager
        def fresh():  # type: ignore[no-untyped-def]
            with support.session_for(engine) as handle:
                yield handle

        import workers.cancellation as cancellation_module
        import workers.run_record as run_record

        monkeypatch.setattr(run_record, "default_session_scope", fresh)

        def unreadable(_session: Any, _run_id: str) -> bool:
            raise refuse("run_row_unreadable", "Check the database.")

        monkeypatch.setattr(cancellation_module, "stop_requested", unreadable)

        with pytest.raises(Refusal) as caught:
            merge_tasks._check_cancelled("run-merge")
        assert caught.value.code == "run_row_unreadable"


class TestNoRevokeOnTheMergePath:
    """ADR-006: ``revoke(terminate=True)`` signals a pool child, the solo pool has none, and
    the call returns successfully having done nothing - an operator sees a cancellation that
    was accepted and a merge that ran to completion anyway. FR-006.24.

    **A first draft of this class grepped the two modules for the string ``revoke`` and went
    red on their own docstrings**, which explain why the call is forbidden. That is the
    fifth-checker failure this project has recorded five times: *a citation is not an
    operation.* ``test_celery_solo_pool.py`` already walks the **AST** of every file under
    ``backend/src/`` and has a self-test proving it sees each spelling, so the right
    assertion here is not a second scanner but that the existing one still **covers these
    two files** - which is what silently stops being true if its enumeration narrows.
    """

    def test_the_live_guard_enumerates_both_merge_modules(self) -> None:
        from tests.unit.test_celery_solo_pool import SRC, python_files

        covered = {str(path.relative_to(SRC)) for path in python_files()}
        assert {"workers/merge_tasks.py", "services/merge_record.py"} <= covered

    def test_and_it_finds_no_terminating_revoke_in_either(self) -> None:
        from tests.unit.test_celery_solo_pool import find_terminating_revokes

        for module in (Path(merge_tasks.__file__), Path(merge_record.__file__)):
            assert not find_terminating_revokes(module.read_text(encoding="utf-8"), module.name)
