"""A floor breach stops the run  (R-02.25, R-02.17; AC-006.13; **M-1**).

**The sharpest control in this feature, and it is not optional.**

The PPRD names *"domain gain silently costs general capability"* as a project risk and names
its mitigation precisely: *"the retention set is scored every step and the floor stops the
run."* F4 detects and names the breach and deliberately **imports nothing that can stop or
cancel a run** - its TID says the boundary *"is kept by what this file cannot reach, not by
discipline."* **F6 is the other half of that contract. If the stop does not happen, F4's
entire feature is a log line.**

What makes it silent: the loss curve falls, the test score rises, the suite is green, and the
run carries on past its floor. Nothing raises. Nothing shows on a curve.

## The fixture round-trips F4's real writer

The breach row is produced by **F4's own** ``evaluate_floor`` and ``record``, not by a
hand-built insert. *Fixtures whose fields agree by construction are how a suite stays green
over a real bug*, and the fields this reads - ``breached``, ``floor``, ``floor_layer``,
``n``, ``provider`` - are exactly the ones a hand-built row would get wrong in the same
direction as the code.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from providers.base import ProviderIdentity
from services.scoring import tables as scoring_tables
from services.scoring.retention import ResolvedFloor, evaluate_floor
from services.scoring.tables import SCORES
from tests.support.jobs import insert_run, memory_engine, read_run, session_for
from tests.support.loop_fakes import FakeSave, FakeStep, boundary_context
from training import loop
from training.run_events import EVENT_BOUNDARY
from workers.run_record import utcnow

RUN = "run-breach"
FLOOR = 0.72
BREACHING = 0.58


@pytest.fixture
def engine():
    engine = memory_engine()
    scoring_tables.metadata.create_all(engine)
    with session_for(engine) as session:
        insert_run(session, RUN, status="running")
        session.commit()
    return engine


def record_a_real_breach(session, step: int) -> dict:
    """Produce the breach row **through F4's own comparison and its own columns**.

    ``evaluate_floor`` is the function that owns ``score < floor`` and declares equality not
    a breach; the row carries exactly what its ``BreachSignal`` carries. Using it here means
    a change to F4's comparison shows up in F6's test, which is the point of a cross-seam
    fixture.
    """
    floor = ResolvedFloor(value=FLOOR, layer="pack_slot")
    signal = evaluate_floor(
        BREACHING,
        floor,
        run_id=RUN,
        step=step,
        n=64,
        provider=ProviderIdentity(kind="in_process", model=RUN),
        set_ref="retention/general@1",
    )
    assert signal is not None, "F4's own comparison did not call this a breach"
    payload = signal.payload()
    session.execute(
        sa.insert(SCORES).values(
            run_id=RUN,
            step=step,
            kind="retention",
            set_rendered=payload["set"],
            score=payload["score"],
            n=payload["n"],
            floor=payload["floor"],
            floor_layer=payload["floor_layer"],
            breached=True,
            provider=payload["provider"],
            provider_kind="in_process",
            provider_model=RUN,
            aggregation="mean",
            decode={"temperature": 0.0},
            reproducible=True,
            out_of_cadence=False,
            created_at=utcnow(),
        )
    )
    return payload


class TestTheRunStopsAtTheNextBoundary:
    def test_a_breach_at_step_2_stops_the_run_at_step_2(self, engine, tmp_path) -> None:
        """AC-006.13. **M-1 is this test's negative control.**

        With ``retention_every = 1`` the exposure window is **0** steps: the breach is seen
        at the boundary it occurred on and the loop stops before the next step.
        """
        from training import run_log

        log = tmp_path / "run.jsonl"
        step = FakeStep()
        save = FakeSave()
        holder: dict[str, object] = {}

        def evaluate(n: int) -> None:
            if n == 2:
                record_a_real_breach(holder["session"], n)  # type: ignore[arg-type]

        with session_for(engine) as session:
            holder["session"] = session
            result = loop.run(
                boundary_context(
                    run_id=RUN,
                    session=session,
                    log_path=log,
                    take_step=step,
                    evaluate=evaluate,
                    save=save,
                    test_every=1,
                    retention_every=1,
                    save_every=None,
                    stop_conditions={
                        "retention_floor": {},
                        "step_count": {"total": 1000},
                    },
                ),
                max_steps=1000,
                evaluate_step_zero=False,
                run_end_save=True,
            )
            session.commit()

        assert result.stopped_because == "retention_floor", (
            f"the run did not stop on the floor breach (stopped_because="
            f"{result.stopped_because!r}). This is the failure the whole feature exists "
            f"to prevent: the loss curve falls, the test score rises, the suite is green, "
            f"and general capability is spent invisibly."
        )
        assert step.calls == [1, 2], "step 3 must not have run"
        assert result.last_step == 2

        # ---- the figures are on the run log ------------------------------- FR-006.9
        boundaries = list(run_log.read_blocks(log, block="boundary"))
        final = boundaries[-1]
        breach = next(c for c in final["stop_conditions"] if c["condition"] == "retention_floor")
        assert breach["triggered"] is True
        assert breach["observed_at_step"] == 2
        assert "0.58" in (breach["detail"] or "") and "0.72" in (breach["detail"] or "")
        assert "64 prompts" in (breach["detail"] or ""), (
            "ADR-012 applies hardest to the number that ended a run: the sample must be " "named"
        )

    def test_the_breach_terminated_run_IS_saved_never_suppressed(self, engine) -> None:
        """FR-006.9: *the artifact tells the truth; it is not hidden to make the run look
        better.* Suppressing the save would destroy the only artifact of the step the
        operator most needs to look at, and would contradict R-02.20's run-end save."""
        save = FakeSave()
        holder: dict[str, object] = {}

        def evaluate(n: int) -> None:
            if n == 1:
                record_a_real_breach(holder["session"], n)  # type: ignore[arg-type]

        with session_for(engine) as session:
            holder["session"] = session
            result = loop.run(
                boundary_context(
                    run_id=RUN,
                    session=session,
                    evaluate=evaluate,
                    save=save,
                    test_every=1,
                    save_every=None,
                    stop_conditions={"retention_floor": {}},
                ),
                max_steps=10,
                evaluate_step_zero=False,
                run_end_save=True,
            )
        assert result.stopped_because == "retention_floor"
        assert save.calls == [1], "a breach-terminated run must still get its run-end save"

    def test_a_score_ABOVE_the_floor_does_not_stop_the_run(self, engine) -> None:
        """The negative half. Without it the test above passes against a condition that
        fires unconditionally, which would stop every run at step 1."""
        holder: dict[str, object] = {}

        def evaluate(n: int) -> None:
            session = holder["session"]
            floor = ResolvedFloor(value=FLOOR, layer="pack_slot")
            # F4's own comparison, with a HEALTHY score: it returns None, so no breach row.
            assert (
                evaluate_floor(
                    0.91,
                    floor,
                    run_id=RUN,
                    step=n,
                    n=64,
                    provider=ProviderIdentity(kind="in_process", model=RUN),
                    set_ref="retention/general@1",
                )
                is None
            )
            session.execute(  # type: ignore[union-attr]
                sa.insert(SCORES).values(
                    run_id=RUN,
                    step=n,
                    kind="retention",
                    set_rendered="retention/general@1",
                    score=0.91,
                    n=64,
                    floor=FLOOR,
                    floor_layer="pack_slot",
                    breached=False,
                    provider="in_process",
                    provider_kind="in_process",
                    provider_model=RUN,
                    aggregation="mean",
                    decode={},
                    reproducible=True,
                    out_of_cadence=False,
                    created_at=utcnow(),
                )
            )

        with session_for(engine) as session:
            holder["session"] = session
            result = loop.run(
                boundary_context(
                    run_id=RUN,
                    session=session,
                    evaluate=evaluate,
                    test_every=1,
                    save_every=None,
                    stop_conditions={"retention_floor": {}, "step_count": {"total": 3}},
                ),
                max_steps=10,
                evaluate_step_zero=False,
                run_end_save=False,
            )
        assert result.stopped_because == "step_count"

    def test_the_floor_outranks_a_coincident_test_threshold(self, engine) -> None:
        """FR-006.27. *"The run stopped because general capability fell below the floor"* is
        the finding; *"the run reached its target"* is not - and a run that hit both wants
        the floor named."""
        holder: dict[str, object] = {}

        def evaluate(n: int) -> None:
            session = holder["session"]
            record_a_real_breach(session, n)  # type: ignore[arg-type]
            session.execute(  # type: ignore[union-attr]
                sa.insert(SCORES).values(
                    run_id=RUN,
                    step=n,
                    kind="test",
                    set_rendered="test/legal@1",
                    score=0.99,
                    n=64,
                    floor=None,
                    floor_layer="none",
                    breached=False,
                    provider="in_process",
                    provider_kind="in_process",
                    provider_model=RUN,
                    aggregation="mean",
                    decode={},
                    reproducible=True,
                    out_of_cadence=False,
                    created_at=utcnow(),
                )
            )

        with session_for(engine) as session:
            holder["session"] = session
            result = loop.run(
                boundary_context(
                    run_id=RUN,
                    session=session,
                    evaluate=evaluate,
                    test_every=1,
                    save_every=None,
                    stop_conditions={
                        "retention_floor": {},
                        "test_threshold": {"threshold": 0.9},
                    },
                ),
                max_steps=10,
                evaluate_step_zero=False,
                run_end_save=False,
            )
        assert result.stopped_because == "retention_floor"
        assert result.all_fired == (
            "retention_floor",
            "test_threshold",
        ), "both must be recorded; only the reason is single-valued"

    def test_the_terminal_transition_names_the_floor_and_announces_it_once(self, engine) -> None:
        """FR-006.28: run row and run log first, Socket.IO second - the same ordering F4
        requires for a breach and F5 for a save, for the same reason."""
        from tests.support.loop_fakes import FakeEmit
        from training import run_state

        emit = FakeEmit()
        holder: dict[str, object] = {}

        def evaluate(n: int) -> None:
            record_a_real_breach(holder["session"], n)  # type: ignore[arg-type]

        with session_for(engine) as session:
            holder["session"] = session
            result = loop.run(
                boundary_context(
                    run_id=RUN,
                    session=session,
                    emit=emit,
                    evaluate=evaluate,
                    test_every=1,
                    save_every=None,
                    stop_conditions={"retention_floor": {}},
                ),
                max_steps=10,
                evaluate_step_zero=False,
                run_end_save=False,
            )
            run_state.mark_completed(
                session, RUN, stop_reason=result.stopped_because, detail=result.stop_detail
            )
            session.commit()
            run_state.announce_terminal(
                emit,
                RUN,
                status="completed",
                stop_reason=result.stopped_because,
                detail=result.stop_detail,
            )
            row = read_run(session, RUN)

        assert row["status"] == "completed"
        assert row["finished_at"] is not None
        terminal = emit.named("run_terminal")
        assert len(terminal) == 1
        assert terminal[0]["stop_reason"] == "retention_floor"
        # And the boundary event preceded it.
        assert [name for _r, name, _p in emit.events][-1] == "run_terminal"
        assert EVENT_BOUNDARY in [name for _r, name, _p in emit.events]


class TestTheRunResponseNamesWhatEndedTheRun:
    """``GET /runs/{id}``'s ``stop_reason``, read from the **authoritative run log**.

    ``mf_runs`` has no ``stop_reason`` column - the DDL is owed and is in F6's feature
    report - so the value is derived from the last ``boundary`` block, whose array carries
    every condition's ``triggered`` and whose precedence order is deterministic.

    **The first draft of this feature left the key permanently null**: ``_loop_block`` read
    it from ``definition_resolved``, which is the *definition* and never carries an
    outcome. Nothing wrote it and nothing noticed, on the most-watched screen in the
    product. Found by asking what the key actually contained rather than by reading the
    code that produced it.
    """

    def test_the_reason_is_derived_from_the_logs_last_boundary(self, engine, tmp_path) -> None:
        from services import run_launch
        from tests.support.loop_fakes import boundary_context

        holder: dict[str, object] = {}

        def evaluate(n: int) -> None:
            record_a_real_breach(holder["session"], n)  # type: ignore[arg-type]

        with session_for(engine) as session:
            holder["session"] = session
            loop.run(
                boundary_context(
                    run_id=RUN,
                    session=session,
                    log_path=run_launch._log_path(RUN, tmp_path),
                    evaluate=evaluate,
                    test_every=1,
                    save_every=None,
                    stop_conditions={
                        "retention_floor": {},
                        "step_count": {"total": 1000},
                    },
                ),
                max_steps=10,
                evaluate_step_zero=False,
                run_end_save=False,
            )
            session.commit()

        outcome = run_launch.terminal_outcome(RUN, log_root=tmp_path)
        assert outcome["stop_reason"] == "retention_floor"
        assert outcome["all_fired"] == ["retention_floor"]
        assert "0.58" in (outcome["stop_detail"] or "")
        assert outcome["round"] == 0

    def test_every_key_is_PRESENT_and_null_for_a_run_that_has_not_ended(self, tmp_path) -> None:
        """*A missing key is indistinguishable from a healthy one*, which on a liveness-
        adjacent field is the whole failure."""
        from services import run_launch

        outcome = run_launch.terminal_outcome("never-ran", log_root=tmp_path)
        assert set(outcome) == {"stop_reason", "stop_detail", "all_fired", "round"}
        assert outcome["stop_reason"] is None
        assert outcome["all_fired"] == []

    def test_the_run_response_carries_it_rather_than_a_permanent_null(
        self, engine, tmp_path
    ) -> None:
        """The API half. Deleting the ``terminal_outcome`` call from ``get_run`` must make
        this red - the key would still be present and would be null forever, which reads as
        "this run has not ended" for a run that has."""
        from api.v1.endpoints.runs import run_payload
        from services import run_launch
        from tests.support.loop_fakes import boundary_context

        holder: dict[str, object] = {}

        def evaluate(n: int) -> None:
            record_a_real_breach(holder["session"], n)  # type: ignore[arg-type]

        with session_for(engine) as session:
            holder["session"] = session
            loop.run(
                boundary_context(
                    run_id=RUN,
                    session=session,
                    log_path=run_launch._log_path(RUN, tmp_path),
                    evaluate=evaluate,
                    test_every=1,
                    save_every=None,
                    stop_conditions={"retention_floor": {}},
                ),
                max_steps=10,
                evaluate_step_zero=False,
                run_end_save=False,
            )
            session.commit()
            row = read_run(session, RUN)

        payload = run_payload(
            {**row, "definition_resolved": None},
            run_launch.terminal_outcome(RUN, log_root=tmp_path),
        )
        assert payload["stop_reason"] == "retention_floor"
        assert payload["all_fired"] == ["retention_floor"]
        # And the heartbeat block survives the extension (M-18's standing note).
        assert "heartbeat" in payload

    def test_the_derived_reason_honours_PRECEDENCE_not_the_arrays_order(
        self, engine, tmp_path
    ) -> None:
        """**A test finding from F6's own sweep.** Mutation **M-19b** - deriving the reason
        as "the first triggered entry in the array" instead of through
        :data:`training.stop.precedence.PRECEDENCE` - **SURVIVED**, because no test had a
        terminating boundary with *two* conditions fired whose array order differs from
        their precedence order.

        The array is written in **registry** order, which puts ``test_threshold`` *before*
        ``retention_floor``. So a reader that took the first triggered entry would report
        *"the run reached its target"* for a run that in fact **fell below its floor** -
        the most misleading pair of sentences in the product, and green in every other
        test.
        """
        from services import run_launch
        from tests.support.loop_fakes import boundary_context
        from training import stop

        holder: dict[str, object] = {}

        def evaluate(n: int) -> None:
            session = holder["session"]
            record_a_real_breach(session, n)  # type: ignore[arg-type]
            session.execute(  # type: ignore[union-attr]
                sa.insert(SCORES).values(
                    run_id=RUN,
                    step=n,
                    kind="test",
                    set_rendered="test/legal@1",
                    score=0.99,
                    n=64,
                    floor=None,
                    floor_layer="none",
                    breached=False,
                    provider="in_process",
                    provider_kind="in_process",
                    provider_model=RUN,
                    aggregation="mean",
                    decode={},
                    reproducible=True,
                    out_of_cadence=False,
                    created_at=utcnow(),
                )
            )

        with session_for(engine) as session:
            holder["session"] = session
            loop.run(
                boundary_context(
                    run_id=RUN,
                    session=session,
                    log_path=run_launch._log_path(RUN, tmp_path),
                    evaluate=evaluate,
                    test_every=1,
                    save_every=None,
                    stop_conditions={
                        "retention_floor": {},
                        "test_threshold": {"threshold": 0.9},
                    },
                ),
                max_steps=10,
                evaluate_step_zero=False,
                run_end_save=False,
            )
            session.commit()

        # The array order, asserted so the test states WHY it is a control rather than
        # relying on a coincidence of registry order that a later edit could change.
        from training import run_log

        final = list(run_log.read_blocks(run_launch._log_path(RUN, tmp_path), block="boundary"))[-1]
        fired_in_array_order = [
            entry["condition"] for entry in final["stop_conditions"] if entry["triggered"]
        ]
        assert fired_in_array_order[0] == "test_threshold", (
            "this control depends on the array putting test_threshold first; registry "
            f"order changed and the control is no longer one: {fired_in_array_order}"
        )

        outcome = run_launch.terminal_outcome(RUN, log_root=tmp_path)
        assert outcome["stop_reason"] == "retention_floor", (
            "the derived reason took the array's first triggered entry instead of the "
            "precedence winner, so a run that fell below its floor reports that it "
            "reached its target"
        )
        assert outcome["all_fired"] == ["retention_floor", "test_threshold"]
        assert stop.precedence.rank("retention_floor") < stop.precedence.rank("test_threshold")
