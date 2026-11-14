"""A breach, end to end, and proved **consumable** by a stand-in stop condition  (FR-004.25).

Publishing a signal nobody can read is the same failure as not detecting it (M-2). A test
that only asserted the emission would prove the signal left the process, not that anything
downstream could act on it - so a stand-in for F6's stop condition reads the published
payload the way R-02.25 will and this file asserts what it decided.

**F4 does not stop the run.** That division is F3's precedent for health signals - *"F3
measures and names; it does not stop a run"* - and here it is visible as: the signal is
published, the run row is untouched, and the stand-in is the thing that says "stop".
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa

from providers.base import GenerationParams
from providers.fakes import FakeProvider
from services.scoring import expectations, retention, service, sets
from services.scoring.retention import BREACH_EVENT
from services.scoring.tables import SCORES
from tests.support import scores as fixtures
from tests.support.db import migrated_engine, session_for

RULE = {
    "kind": "rule",
    "name": "len",
    "config_sha256": None,
    "expression": '{"min_length": 4}',
}


@pytest.fixture
def engine(tmp_path: Path):
    engine = migrated_engine(tmp_path / "db")
    fixtures.create_score_tables(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    with session_for(engine) as session:
        yield session


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "runs" / "run-1" / "run.jsonl"


def spec(floor_value: float, layer: str = "pack_slot") -> service.EvaluationSpec:
    expectation = expectations.resolve_expectation(RULE, source="pack")

    def a_set(kind: str) -> sets.ResolvedSet:
        reference = sets.SetReference(
            rendered=f"{kind}_set/general@1",
            structured={"type": f"{kind}_set", "name": "general", "version": "1"},
        )
        return sets.resolve(
            kind,
            reference,
            [{"prompt": f"p{i}"} for i in range(5)],
            markers=("evaluation_only",),
        )

    return service.EvaluationSpec(
        run_id="run-1",
        provider_name="in_process",
        aggregation="mean",
        decode=GenerationParams(max_tokens=16),
        test_set=a_set("test"),
        retention_set=a_set("retention"),
        test_expectation=expectation,
        retention_expectation=expectation,
        floor=retention.ResolvedFloor(value=floor_value, layer=layer),
        batch_bound=5,
    )


class TestABreachIsDetectedRecordedAndPublished:
    def test_all_three_surfaces_carry_it(self, session, log_path: Path) -> None:
        emitter = fixtures.RecordingEmitter()
        outcome = service.evaluate(
            spec(0.9),
            11,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="no"),  # nothing admitted: score 0.0
            emit=emitter,
        )
        # 1. the signal
        assert outcome.breach is not None
        # 2. the run log, which is the authoritative copy
        # Canonical JSON: sorted keys, no spaces (`core.canonical.canonical_bytes` is the
        # codebase's only serialiser, ADR-005).
        assert '"retention_floor_breached":true' in log_path.read_text()
        # 3. the index
        row = (
            session.execute(sa.select(SCORES).where(SCORES.c.kind == "retention")).mappings().one()
        )
        assert row["breached"] is True
        # 4. the room
        assert len(emitter.events(BREACH_EVENT)) == 1

    def test_the_durable_write_happens_before_the_emission(self, session, log_path: Path) -> None:
        """ADR-007: *progress reported only over the socket is invisible to the janitor.*

        Asserted by an emitter that reads the log at emit time - if the emission came first,
        the file would not exist yet.
        """
        seen: list[bool] = []

        def emit(run_id: str, event: str, payload: dict) -> None:
            seen.append(log_path.is_file())

        service.evaluate(
            spec(0.9),
            11,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="no"),
            emit=emit,
        )
        assert seen and all(seen), "an emission preceded the durable write"

    def test_the_payload_is_what_a_stop_condition_needs(self, session, log_path: Path) -> None:
        emitter = fixtures.RecordingEmitter()
        service.evaluate(
            spec(0.9),
            11,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="no"),
            emit=emitter,
        )
        payload = emitter.events(BREACH_EVENT)[0]
        assert payload["score"] == 0.0
        assert payload["floor"] == 0.9
        assert payload["n"] == 5
        assert payload["step"] == 11
        assert payload["provider"] == "in_process:fake-model"
        assert payload["floor_layer"] == "pack_slot"


class TestTheSignalIsConsumable:
    """The half a publication test alone does not prove."""

    def test_a_stand_in_stop_condition_decides_to_stop(self, session, log_path: Path) -> None:
        emitter = fixtures.RecordingEmitter()
        service.evaluate(
            spec(0.9),
            11,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="no"),
            emit=emitter,
        )
        condition = fixtures.StandInStopCondition()
        assert condition.consume(emitter.events(BREACH_EVENT)) is True

    def test_it_does_not_stop_when_nothing_breached(self, session, log_path: Path) -> None:
        emitter = fixtures.RecordingEmitter()
        service.evaluate(
            spec(0.1),
            11,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="longer"),
            emit=emitter,
        )
        condition = fixtures.StandInStopCondition()
        assert condition.consume(emitter.events(BREACH_EVENT)) is False

    def test_a_breach_is_reconstructible_from_the_log_alone(self, session, log_path: Path) -> None:
        """ADR-004: ``reindex`` rebuilds from the log, so the decision the loop took has to
        be in the line - not recomputed later against a floor that may since have been edited.
        """
        from training import run_log

        service.evaluate(
            spec(0.9),
            11,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="no"),
        )
        blocks = list(run_log.read_blocks(log_path, block="evaluation"))
        assert blocks[0]["retention_floor_breached"] is True

    def test_the_run_row_is_not_transitioned_by_this_feature(self, session, log_path: Path) -> None:
        """R-02.25 is F6's. F4 detects, names and publishes."""
        service.evaluate(
            spec(0.9),
            11,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="no"),
        )
        assert session.execute(sa.text("SELECT COUNT(*) FROM mf_runs")).scalar_one() == 0


class TestTheFloorLayerIsRecordedWhicheverLayerSuppliedIt:
    """FPRD OQ-2 / DEF-7 is unsettled above this document; the record stays honest."""

    @pytest.mark.parametrize("layer", ["pack_slot", "run_override"])
    def test_the_layer_reaches_the_row_and_the_signal(
        self, session, log_path: Path, layer: str
    ) -> None:
        emitter = fixtures.RecordingEmitter()
        service.evaluate(
            spec(0.9, layer),
            11,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="no"),
            emit=emitter,
        )
        row = (
            session.execute(sa.select(SCORES).where(SCORES.c.kind == "retention")).mappings().one()
        )
        assert row["floor_layer"] == layer
        assert emitter.events(BREACH_EVENT)[0]["floor_layer"] == layer

    def test_a_test_row_records_no_floor_and_no_layer(self, session, log_path: Path) -> None:
        service.evaluate(
            spec(0.1),
            11,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="longer"),
        )
        row = session.execute(sa.select(SCORES).where(SCORES.c.kind == "test")).mappings().one()
        assert row["floor"] is None
        assert row["floor_layer"] == "none"
        assert row["breached"] is False
