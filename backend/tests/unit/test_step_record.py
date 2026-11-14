"""The step-record writer and its ordering  (FR-001.24 - FR-001.27, AC-001.6).

**The order is the requirement**: run log (authoritative) -> ``mf_steps`` (index) -> socket
(projection) -> heartbeat. A crash between 1 and 2 costs a ``reindex``; a crash between 2
and 1 would cost **the record**, leaving the index as the only evidence of a step the
authoritative log denies ever happened.

Inverting it changes nothing observable until the machine dies, which is exactly why the
test asserts the **observed sequence** rather than the outcome - the same reason
``core/atomic_dir.py``'s own test asserts a syscall order.

Rows go in and come back through the **real table** and the real projection, never a
hand-built dict: a fixture whose fields agree with the assertion by construction is how a
suite stays green over a real bug (ADR-016).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from core.errors import Refusal
from models.run import Step
from tests.support.jobs import insert_run, memory_engine, session_for
from training import run_log
from training.run_config import RunConfig
from training.step import StepOutcome
from training.step_record import STEP_EVENT, StepWriter, rebuild_step_index

RUN = "run-1"
CFG = RunConfig(learning_rate=2e-5, warmup_steps=5, batch_size=4, gradient_accumulation=2)


def outcome(**overrides) -> StepOutcome:
    base = {
        "loss": 0.75,
        "objective": "sft",
        "grad_norm": 1.5,
        "wall_time_ms": 42.25,
        "batch_sha256": "c" * 64,
    }
    base.update(overrides)
    return StepOutcome(**base)  # type: ignore[arg-type]


@pytest.fixture
def engine():
    engine = memory_engine()
    with session_for(engine) as session:
        insert_run(session, RUN, status="running")
    return engine


@pytest.fixture
def writer(tmp_path: Path):
    return StepWriter(run_id=RUN, log_root=tmp_path)


class TestTheWriteOrder:
    def test_the_durable_write_comes_FIRST_then_index_then_socket_then_heartbeat(
        self, engine, tmp_path: Path
    ) -> None:
        emitted: list[tuple[str, str, dict[str, Any]]] = []

        class Heartbeat:
            def touch(self, session: Any) -> bool:
                return True

        writer = StepWriter(
            run_id=RUN,
            log_root=tmp_path,
            heartbeat=Heartbeat(),
            emit=lambda run_id, event, payload: emitted.append((run_id, event, payload)),
        )
        with session_for(engine) as session:
            writer.write(session=session, step=1, outcome=outcome(), cfg=CFG)

        assert writer.effects == ["log", "index", "socket", "heartbeat", "commit"]
        assert emitted and emitted[0][1] == STEP_EVENT

    def test_the_record_exists_on_disk_before_the_index_row(self, engine, tmp_path: Path) -> None:
        """Asserted by observing the state *at the moment the index is written*, which is
        the only way to tell this ordering from its inverse."""
        observed: dict[str, bool] = {}
        writer = StepWriter(run_id=RUN, log_root=tmp_path)
        original = writer._upsert

        def spy(session: Any, record: dict[str, Any]) -> None:
            observed["log_exists_when_indexing"] = writer.log_path.is_file()
            original(session, record)

        writer._upsert = spy  # type: ignore[method-assign]
        with session_for(engine) as session:
            writer.write(session=session, step=1, outcome=outcome(), cfg=CFG)
        assert observed["log_exists_when_indexing"] is True

    def test_the_socket_is_emitted_only_after_the_record_exists(
        self, engine, tmp_path: Path
    ) -> None:
        """ADR-007: *progress reported only over the socket is invisible to the janitor* -
        which is how miStudio reaped a live 5.8-hour job as "worker lost."."""
        seen: list[bool] = []
        writer = StepWriter(
            run_id=RUN,
            log_root=tmp_path,
            emit=lambda *_args: seen.append(writer.log_path.is_file()),
        )
        with session_for(engine) as session:
            writer.write(session=session, step=1, outcome=outcome(), cfg=CFG)
        assert seen == [True]

    def test_an_invalid_record_writes_NOTHING_anywhere(self, engine, tmp_path: Path) -> None:
        """Validation happens before the first side effect, not between two of them, so an
        invalid block leaves no index row with no record behind it.

        A NaN loss is the case that matters: **JSON Schema's ``number`` admits it**, so the
        contract check passes and only the finite-number guard catches it - as a *named*
        refusal rather than as the serialiser's bare ``ValueError``, which would reach a
        client as a 500 with a traceback.
        """
        writer = StepWriter(run_id=RUN, log_root=tmp_path)
        with session_for(engine) as session, pytest.raises(Refusal) as caught:
            writer.write(session=session, step=1, outcome=outcome(loss=float("nan")), cfg=CFG)
        assert caught.value.code == "step_record_not_finite"
        assert caught.value.details["field"] == "loss"
        assert not writer.log_path.exists()
        with session_for(engine) as session:
            assert (
                session.execute(sa.select(sa.func.count()).select_from(Step.__table__)).scalar()
                == 0
            )


class TestTheRecordItself:
    def test_the_block_carries_R_02_14s_fields_populated(self, engine, writer) -> None:
        """AC-001.6: loss, gradient norm, learning rate, batch source hashes and wall time
        populated, and the two §5.10 fields unset."""
        with session_for(engine) as session:
            record = writer.write(session=session, step=3, outcome=outcome(), cfg=CFG)

        assert record["loss"] == 0.75
        assert record["gradient_norm"] == 1.5
        assert record["learning_rate"] == CFG.learning_rate
        assert record["batch_sha256"] == "c" * 64
        assert record["wall_time_ms"] == 42.25
        assert record["objective"] == "sft"
        assert record["quiesce_hold_ms"] is None
        assert record["commit_duration_ms"] is None

    def test_the_learning_rate_comes_from_the_CONFIG_not_the_outcome(self, engine, writer) -> None:
        """The step does not carry it: the rate is a property of the run's configuration and
        the scheduler, and reading it from two places would let the record disagree with the
        optimizer."""
        with session_for(engine) as session:
            record = writer.write(session=session, step=1, outcome=outcome(), cfg=CFG)
        assert record["learning_rate"] == 2e-5

    def test_it_round_trips_through_the_real_reader(self, engine, writer) -> None:
        with session_for(engine) as session:
            record = writer.write(session=session, step=1, outcome=outcome(), cfg=CFG)
        assert list(run_log.read_blocks(writer.log_path, block="step")) == [record]

    def test_the_socket_payload_IS_the_record(self, engine, tmp_path: Path) -> None:
        """So F9 types against ``step-record-v1.json`` rather than against this module's
        Python, and the two cannot drift."""
        emitted: list[dict[str, Any]] = []
        writer = StepWriter(
            run_id=RUN, log_root=tmp_path, emit=lambda r, e, payload: emitted.append(payload)
        )
        with session_for(engine) as session:
            record = writer.write(session=session, step=1, outcome=outcome(), cfg=CFG)
        assert emitted == [record]


class TestTheIndexRow:
    def test_the_row_lands_in_mf_steps_through_the_real_table(self, engine, writer) -> None:
        with session_for(engine) as session:
            writer.write(session=session, step=7, outcome=outcome(), cfg=CFG)
        with session_for(engine) as session:
            row = session.execute(sa.select(Step.__table__)).one()._mapping
        assert row["run_id"] == RUN
        assert row["step"] == 7
        assert row["loss"] == 0.75
        assert row["grad_norm"] == 1.5
        assert row["wall_time_ms"] == 42
        assert row["batch_source_hashes"] == {"batch_sha256": "c" * 64}

    def test_writing_the_same_step_twice_replaces_rather_than_duplicating(
        self, engine, writer
    ) -> None:
        """``reindex`` reruns over the same log, and an insert-only path would make a rebuild
        fail on its second run - making ADR-016's equivalence test pass once and never
        again."""
        with session_for(engine) as session:
            writer.write(session=session, step=1, outcome=outcome(), cfg=CFG)
        with session_for(engine) as session:
            writer.write(session=session, step=1, outcome=outcome(loss=0.1), cfg=CFG)
        with session_for(engine) as session:
            rows = session.execute(sa.select(Step.__table__)).all()
        assert len(rows) == 1
        assert rows[0]._mapping["loss"] == 0.1

    def test_writing_without_a_session_still_writes_the_RECORD(
        self, writer, tmp_path: Path
    ) -> None:
        """The log is the truth; the index is a convenience. A worker with no database
        reachable must still produce the record, or ADR-004's ordering is a fiction."""
        record = writer.write(session=None, step=1, outcome=outcome(), cfg=CFG)
        assert writer.effects == ["log"]
        assert list(run_log.read_blocks(writer.log_path)) == [record]


class TestTheLastRecordedStep:
    def test_it_is_read_from_the_LOG_not_the_index(self, engine, writer) -> None:
        """E-9 asks *"what is the last step that actually happened"* - the question asked
        after a crash, when the index may be behind or absent entirely."""
        with session_for(engine) as session:
            for step in (1, 2, 3):
                writer.write(session=session, step=step, outcome=outcome(), cfg=CFG)
        with session_for(engine) as session:
            session.execute(sa.delete(Step.__table__))
        assert writer.last_recorded_step() == 3

    def test_no_steps_yet_is_None(self, writer) -> None:
        assert writer.last_recorded_step() is None


class TestReindexEquivalenceForSteps:
    """ADR-016 #4, applied to the one table whose authoritative copy is the run log."""

    def test_dropping_the_index_and_rebuilding_restores_identical_rows(
        self, engine, writer
    ) -> None:
        with session_for(engine) as session:
            for step in range(1, 6):
                writer.write(session=session, step=step, outcome=outcome(loss=step / 10), cfg=CFG)
        with session_for(engine) as session:
            before = [
                dict(r._mapping)
                for r in session.execute(
                    sa.select(Step.__table__).order_by(Step.__table__.c.step)
                ).all()
            ]

        with session_for(engine) as session:
            session.execute(sa.delete(Step.__table__))
        with session_for(engine) as session:
            assert rebuild_step_index(session, log_root=writer.log_root, run_id=RUN) == 5
        with session_for(engine) as session:
            after = [
                dict(r._mapping)
                for r in session.execute(
                    sa.select(Step.__table__).order_by(Step.__table__.c.step)
                ).all()
            ]

        def strip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Drop the columns that do not describe the step.

            ``id`` is an autoincrement surrogate and ``created_at`` is a write time; both are
            expected to differ after a rebuild. **Every column that describes the step must
            be identical**, which is what equivalence means here.
            """
            return [{k: v for k, v in row.items() if k not in {"id", "created_at"}} for row in rows]

        assert strip(after) == strip(before)

    def test_a_rebuild_is_idempotent(self, engine, writer) -> None:
        with session_for(engine) as session:
            writer.write(session=session, step=1, outcome=outcome(), cfg=CFG)
        for _ in range(3):
            with session_for(engine) as session:
                rebuild_step_index(session, log_root=writer.log_root, run_id=RUN)
        with session_for(engine) as session:
            assert len(session.execute(sa.select(Step.__table__)).all()) == 1

    def test_a_rebuild_removes_a_row_the_log_no_longer_contains(self, engine, writer) -> None:
        """An upsert-only rebuild would leave a rolled-back step visible forever."""
        with session_for(engine) as session:
            writer.write(session=session, step=1, outcome=outcome(), cfg=CFG)
            writer.write(session=session, step=2, outcome=outcome(), cfg=CFG)
        # Truncate the log to its first line, as a crash mid-append would.
        lines = writer.log_path.read_text().splitlines(keepends=True)
        writer.log_path.write_text(lines[0])
        with session_for(engine) as session:
            rebuild_step_index(session, log_root=writer.log_root, run_id=RUN)
        with session_for(engine) as session:
            steps = [r._mapping["step"] for r in session.execute(sa.select(Step.__table__)).all()]
        assert steps == [1]

    def test_rebuilding_a_run_with_no_log_REFUSES(self, engine, tmp_path: Path) -> None:
        """An empty index would assert that the run took no steps. The log is the record
        (ADR-004): without it the steps are gone, and saying so is the only honest answer."""
        with session_for(engine) as session, pytest.raises(Refusal) as caught:
            rebuild_step_index(session, log_root=tmp_path, run_id=RUN)
        assert caught.value.code == "run_log_absent"
        assert "authoritative record" in caught.value.message
