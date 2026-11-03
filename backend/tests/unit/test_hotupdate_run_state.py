"""The attachment id's durability, and the tripwire for the DDL it needs  (**MUT-12**).

Two halves, and the second is a **tripwire that must be deleted when the migration lands**:

* the round trip is proved against a table that carries the two columns, so the code is
  exercised rather than merely written;
* ``mf_runs`` is asserted **not** to carry them today, so the moment they are added this
  test goes red and says so. *A test that pins the absence of something owed is worth more
  than a note, because a note goes stale silently.*

**Why this is not solved by keeping the id in memory.** That is MUT-12 shipped on purpose:
it passes every test, works in production until the first worker restart, and then produces
a refused commit whose cause nothing connects to the restart. miLLM returns the id once and
no read path returns it again.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from core.errors import Refusal
from training.hotupdate import run_state


@pytest.fixture
def table_with_the_columns() -> tuple[sa.Table, sa.engine.Engine]:
    """A ``mf_runs``-shaped table **with** §5.10's two columns, as the migration will leave
    it. Both nullable, nothing backfilled, nothing dropped (TDD §4.6)."""
    metadata = sa.MetaData()
    runs = sa.Table(
        "mf_runs",
        metadata,
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column(run_state.BASE_URL_COLUMN, sa.String(255), nullable=True),
        sa.Column(run_state.ATTACHMENT_ID_COLUMN, sa.String(128), nullable=True),
    )
    engine = sa.create_engine("sqlite://")
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(sa.insert(runs).values(id="hot-run", status="running"))
    return runs, engine


class TestTheRoundTrip:
    def test_the_id_is_written_and_read_back(
        self, table_with_the_columns: tuple[sa.Table, sa.engine.Engine]
    ) -> None:
        """**Through the real writer and the real reader** (ADR-016): a fixture that set the
        column by hand would prove the assertion rather than the code."""
        runs, engine = table_with_the_columns
        with engine.begin() as connection:
            run_state.remember(
                connection,
                "hot-run",
                attachment_id="attachment-0001",
                base_url="http://millm.test:8000",
                table=runs,
            )
        with engine.begin() as connection:
            assert run_state.recall(connection, "hot-run", table=runs) == (
                "attachment-0001",
                "http://millm.test:8000",
            )

    def test_a_restarted_worker_can_commit_with_what_it_read_back(
        self,
        table_with_the_columns: tuple[sa.Table, sa.engine.Engine],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The whole point: a **fresh client** - a restarted worker - holds nothing in
        memory, reads the id from the row, and the holder-only call succeeds."""
        from tests.support.fake_weights_surface import FakeWeightsSurface
        from tests.support.fixtures_hotupdate import configure
        from training.hotupdate import settings as settings_module
        from training.hotupdate.client import WeightsClient

        configure(monkeypatch)
        runs, engine = table_with_the_columns
        fake = FakeWeightsSurface()
        first = WeightsClient(settings=settings_module.load(), transport=fake.transport())
        attachment_id = str(first.call("attach")["attachment_id"])
        with engine.begin() as connection:
            run_state.remember(
                connection, "hot-run", attachment_id=attachment_id, base_url="http://x", table=runs
            )
        first.close()

        with engine.begin() as connection:
            recalled, _ = run_state.recall(connection, "hot-run", table=runs)
        restarted = WeightsClient(settings=settings_module.load(), transport=fake.transport())
        restarted.remember_attachment(str(recalled))
        assert restarted.call("quiesce")["weights_state"] == "clean"
        restarted.close()

    def test_a_missing_run_row_is_refused_rather_than_silently_dropped(
        self, table_with_the_columns: tuple[sa.Table, sa.engine.Engine]
    ) -> None:
        runs, engine = table_with_the_columns
        with engine.begin() as connection, pytest.raises(Refusal) as caught:
            run_state.remember(
                connection, "no-such-run", attachment_id="a", base_url=None, table=runs
            )
        assert caught.value.code == "hotupdate_run_not_found"

    def test_forget_clears_the_id_after_a_clean_detach(
        self, table_with_the_columns: tuple[sa.Table, sa.engine.Engine]
    ) -> None:
        """A finished run must not hold a live capability."""
        runs, engine = table_with_the_columns
        with engine.begin() as connection:
            run_state.remember(connection, "hot-run", attachment_id="a", base_url="u", table=runs)
            run_state.forget(connection, "hot-run", table=runs)
            assert run_state.recall(connection, "hot-run", table=runs) == (None, "u")


class TestTheColumnsLandedAndTheIdIsDurable:
    """**This class was `TestTheTripwireForTheOwedMigration` and it fired on 2026-10-14.**

    It asserted the columns were **absent** and said *"DELETE THIS TEST when the migration
    lands"*, with the reasoning that *"a note goes stale silently, a test does not."* It did
    not go stale: migration ``0017`` landed and it went red. Replaced by its opposite rather
    than deleted, because the thing worth asserting afterwards is stronger than the thing it
    was waiting for.

    **What the columns buy, stated as the failure they remove.** ``attach`` returns the
    attachment id **exactly once** and no read path returns it (BRD-01 R-01.31), while seven
    of the fifteen calls are holder-only and carry it — including ``commit-begin``,
    ``commit-end``, ``save`` and ``detach``. So holding the id in worker memory is F11's
    MUT-12 shipped on purpose: it passes every test, survives until the first worker restart,
    and then produces a refused commit nothing can explain and a run that dies looking like a
    crash.
    """

    def test_the_columns_are_present_now(self) -> None:
        assert run_state.columns_present(), (
            "mf_runs lost §5.10's two columns. `attach` returns the id once and no read path "
            "returns it, so without somewhere durable to put it a worker restart loses the "
            "ability to commit, save or detach - and the run dies looking like a crash."
        )

    def test_remember_then_read_back_through_a_REAL_session(self, tmp_path) -> None:
        """A round trip through the real table, not a double.

        A double would agree with the writer by construction - the fixture trap this project
        records - and the question here is whether the column actually holds the value across
        a read, which only a real session answers.
        """
        from tests.support.db import migrated_engine, session_for
        from tests.support.jobs import insert_run

        engine = migrated_engine(tmp_path / "db")
        try:
            with session_for(engine) as session:
                insert_run(session, "hot-run", status="running")
                run_state.remember(
                    session,
                    run_id="hot-run",
                    attachment_id="att-9f2c",
                    base_url="https://millm.local:8000",
                )
                session.commit()
            # A SECOND session, so the value is read from the database and not from the
            # first session's identity map - which would pass against a value that was
            # never flushed.
            with session_for(engine) as session:
                # `recall` returns a plain `(attachment_id, base_url)` tuple, not an object -
                # read from its signature rather than guessed at.
                attachment_id, base_url = run_state.recall(session, run_id="hot-run")
                assert attachment_id == "att-9f2c"
                assert base_url == "https://millm.local:8000"
        finally:
            engine.dispose()

    def test_an_absent_run_still_refuses_by_name(self, tmp_path) -> None:
        """The refusal that survives the migration, and it is the important one.

        `hotupdate_run_columns_missing` is gone; `hotupdate_run_not_found` is not. An id held
        by no row is an id that cannot be found later, so the commit it was needed for is
        refused for a reason nothing connects back to this moment.
        """
        from core.errors import Refusal
        from tests.support.db import migrated_engine, session_for

        engine = migrated_engine(tmp_path / "db")
        try:
            with session_for(engine) as session:
                with pytest.raises(Refusal) as caught:
                    run_state.remember(
                        session,
                        run_id="no-such-run",
                        attachment_id="att-1",
                        base_url="https://millm.local:8000",
                    )
                assert caught.value.code == "hotupdate_run_not_found"
        finally:
            engine.dispose()

    def test_the_repr_never_carries_the_attachment_id(self) -> None:
        """`repr=False` is unavailable on this model, so the protection is a `__repr__`.

        F11's handover suggested `mapped_column(..., repr=False)` and SQLAlchemy **refuses**
        it: `Run` is a plain declarative model, not `MappedAsDataclass`. Had the keyword been
        accepted and ignored, the column would have looked protected and not been. A `repr`
        reaches a traceback frame, a debugger watch and any log line that interpolates the
        object, and the attachment id authorises every holder-only call - a credential in
        everything but name.
        """
        from models.run import Run

        text = repr(
            Run(
                id="r1",
                status="running",
                run_type="change",
                attachment_id="SECRET-ATT-12345",
                hotupdate_base_url="https://millm.local:8000",
            )
        )
        assert "SECRET-ATT-12345" not in text, "repr(Run) leaks the attachment id"
        # The base URL is a LOCATION, not a secret, and is exactly what a reader of a
        # traceback needs in order to know which serving process a run attached to.
        assert "millm.local" in text


class TestTheAcquirerActuallyCallsIt:
    """**Declaring a mechanism is not wiring it.** This project has shipped three
    capabilities nobody could reach, and the weaker check - grep for a caller - *"proves a
    call exists, not that anything depends on it."*

    So the assertion is on the behaviour: give ``acquire`` a session and the refusal that
    arrives is ``run_state``'s, which means the call is on the path. When the DDL lands, the
    same call persists the id and nothing here moves.
    """

    def test_acquire_reaches_run_state_when_it_is_given_a_session(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        from core.errors import Refusal
        from tests.support.fake_weights_surface import FakeWeightsSurface
        from tests.support.fixtures_hotupdate import configure
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec, Availability
        from training.hotupdate.attach import AttachAcquirer

        configure(monkeypatch)
        fake = FakeWeightsSurface()
        acquirer = AttachAcquirer(transport=fake.transport())
        # The ML-runtime gate is ADR-023's and is asserted in test_hotupdate_availability.py;
        # here it would refuse before the subject of this test is reached, because CI has no
        # torch. Stubbing the GATE, not the behaviour under test.
        monkeypatch.setattr(type(acquirer), "available", lambda self: Availability(available=True))
        # `session=object()` worked only while `hotupdate_run_columns_missing` fired BEFORE
        # any database call. Migration 0017 landed, so the path now reaches the real UPDATE
        # and a bare `object()` raises AttributeError instead - **a double that stopped being
        # able to express the thing under test.** A real session with no matching row keeps
        # the assertion about reachability while letting the refusal be a real one.
        from tests.support.db import migrated_engine, session_for

        engine = migrated_engine(tmp_path / "db")
        try:
            with session_for(engine) as session, pytest.raises(Refusal) as caught:
                acquirer.acquire(
                    AcquisitionSpec(mode=AcquisitionMode.ATTACH),
                    session=session,
                    run_id="hot-run",
                )
            assert caught.value.code == "hotupdate_run_not_found", (
                "acquire did not reach run_state.remember - the seam is declared and not "
                "wired. (Before 0017 this asserted hotupdate_run_columns_missing; the "
                "columns exist now, so the reachable refusal is the absent row.)"
            )
        finally:
            engine.dispose()
        # And the attach itself happened first, so the id existed to be persisted: the
        # refusal is about storage, not about a failed attach.
        assert fake.call_count("attach") == 1

    def test_without_a_session_the_id_lives_on_the_client_for_the_attachments_lifetime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Honest about what that buys: enough for a step, not enough for a restart."""
        from tests.support.fake_weights_surface import FakeWeightsSurface
        from tests.support.fixtures_hotupdate import configure
        from training.hotupdate.attach import AttachAcquirer

        configure(monkeypatch)
        fake = FakeWeightsSurface()
        acquirer = AttachAcquirer(transport=fake.transport())
        client = acquirer._open_client()
        response = client.call("attach")
        client.remember_attachment(str(response["attachment_id"]))
        assert client.attachment_id == fake.attachment_id
