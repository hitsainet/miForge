"""A full cadence, a save, and a save with nothing to save  (FR-004.14, 24, 29, 30, 32).

A curve over several steps against the fake handle, a save at a cadence boundary whose
manifest is round-tripped **through F5's real writer and reader**, and a save at a
non-evaluation step writing both nulls.

Round-tripped rather than hand-built because that is the one thing this suite most needs: *a
hand-built manifest dict agrees with the test's expectation by construction*, and the defect
it would agree with is a ScoreRecord written without its declared ``provider`` - which
validates, because the field is not required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa

from providers.base import GenerationParams
from providers.fakes import FakeProvider
from services.scoring import expectations, record, retention, service, sets
from services.scoring.settings import ScoringSettings, exposure_window_steps
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


def resolved_set(kind: str) -> sets.ResolvedSet:
    reference = sets.SetReference(
        rendered=f"{kind}_set/general@1",
        structured={"type": f"{kind}_set", "name": "general", "version": "1"},
    )
    return sets.resolve(
        kind,
        reference,
        [{"prompt": f"{kind}-p{i}", "group": f"g{i % 2}"} for i in range(8)],
        markers=("held_out", "evaluation_only"),
    )


def spec() -> service.EvaluationSpec:
    expectation = expectations.resolve_expectation(RULE, source="pack")
    return service.EvaluationSpec(
        run_id="run-1",
        provider_name="in_process",
        aggregation="mean",
        decode=GenerationParams(max_tokens=16, temperature=0.0),
        test_set=resolved_set("test"),
        retention_set=resolved_set("retention"),
        test_expectation=expectation,
        retention_expectation=expectation,
        floor=retention.ResolvedFloor(value=0.4, layer="pack_slot"),
        batch_bound=4,
    )


class TestACadenceProducesTwoCurves:
    def test_four_steps_produce_four_points_per_kind(self, session, log_path: Path) -> None:
        emitter = fixtures.RecordingEmitter()
        # The model "improves": fewer short answers as the step rises, which is the shape an
        # operator reads off the curve.
        for step in (0, 1, 2, 3):

            def answers(prompt, index, step=step):
                return "longer" if index < step * 2 else "no"

            service.evaluate(
                spec(),
                step,
                session=session,
                log_path=log_path,
                provider=FakeProvider(answers=answers),
                emit=emitter,
            )
        rows = list(
            session.execute(sa.select(SCORES).order_by(SCORES.c.step, SCORES.c.kind)).mappings()
        )
        assert len(rows) == 8
        test_curve = [row["score"] for row in rows if row["kind"] == "test"]
        assert test_curve == sorted(test_curve), "the curve should be monotone for this fixture"

    def test_step_zero_is_the_baseline(self, session, log_path: Path) -> None:
        """Criterion 12 names the step-0 test score as a creation run's baseline explicitly."""
        service.evaluate(
            spec(), 0, session=session, log_path=log_path, provider=FakeProvider(answers="no")
        )
        assert session.execute(sa.select(SCORES.c.step).distinct()).scalars().all() == [0]

    def test_every_point_carries_n_the_provider_and_the_markers(
        self, session, log_path: Path
    ) -> None:
        """ADR-012 and FR-004.19 on every row, so nothing downstream can lose them."""
        service.evaluate(
            spec(), 1, session=session, log_path=log_path, provider=FakeProvider(answers="no")
        )
        for row in session.execute(sa.select(SCORES)).mappings():
            assert row["n"] == 8
            assert row["provider"] == "in_process:fake-model"
            assert row["provider_kind"] == "in_process"
            assert sorted(row["split_markers"]) == ["evaluation_only", "held_out"]
            assert row["aggregation"] == "mean"
            assert row["decode"]["temperature"] == 0.0

    def test_one_run_log_line_per_pass(self, session, log_path: Path) -> None:
        for step in (0, 1, 2):
            service.evaluate(
                spec(),
                step,
                session=session,
                log_path=log_path,
                provider=FakeProvider(answers="no"),
            )
        assert len(log_path.read_text().strip().splitlines()) == 3

    def test_a_socket_event_per_score_plus_one_per_breach(self, session, log_path: Path) -> None:
        emitter = fixtures.RecordingEmitter()
        service.evaluate(
            spec(),
            1,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="no"),  # nothing admitted: 0.0, below the 0.4 floor
            emit=emitter,
        )
        assert [event for _r, event, _p in emitter.calls] == [
            "score",
            "score",
            "retention_floor_breached",
        ]


class TestASaveAtACadenceBoundary:
    def test_the_manifest_round_trips_both_scores_with_their_providers(
        self, session, log_path: Path, tmp_path: Path
    ) -> None:
        """Through F5's **real** builder, off the disk, and back."""
        service.evaluate(
            spec(), 5, session=session, log_path=log_path, provider=FakeProvider(answers="longer")
        )
        payload = record.manifest_payload_for_step(session, run_id="run-1", step=5)
        manifest = fixtures.written_and_read_manifest(
            tmp_path / "ckpt", step=5, test=payload["test"], retention=payload["retention"]
        )
        assert manifest["test"]["provider"] == "in_process:fake-model"
        assert manifest["retention"]["provider"] == "in_process:fake-model"
        assert manifest["test"]["n"] == 8
        assert manifest["retention"]["floor"] == 0.4

    def test_the_publication_is_one_call_with_both_arguments(self, session, log_path: Path) -> None:
        service.evaluate(
            spec(), 5, session=session, log_path=log_path, provider=FakeProvider(answers="longer")
        )
        publisher = fixtures.RecordingPublisher()
        record.publish_to_manifest(session, run_id="run-1", step=5, publish=publisher)
        assert publisher.count == 1
        test, retained = publisher.calls[0]
        assert test is not None and retained is not None

    def test_a_save_at_a_non_evaluation_step_writes_both_nulls(
        self, session, log_path: Path, tmp_path: Path
    ) -> None:
        """E-22. Present and null - *null is an assertion that nothing ran.*"""
        service.evaluate(
            spec(), 5, session=session, log_path=log_path, provider=FakeProvider(answers="longer")
        )
        payload = record.manifest_payload_for_step(session, run_id="run-1", step=6)
        assert payload == {"test": None, "retention": None}
        manifest = fixtures.written_and_read_manifest(
            tmp_path / "ckpt", step=6, test=None, retention=None
        )
        assert "test" in manifest and manifest["test"] is None
        assert "retention" in manifest and manifest["retention"] is None

    def test_no_prompt_or_completion_text_reaches_the_manifest(
        self, session, log_path: Path, tmp_path: Path
    ) -> None:
        """ADR-020: the manifest travels with the artifact and is published beside it."""
        service.evaluate(
            spec(), 5, session=session, log_path=log_path, provider=FakeProvider(answers="longer")
        )
        payload = record.manifest_payload_for_step(session, run_id="run-1", step=5)
        manifest = fixtures.written_and_read_manifest(
            tmp_path / "ckpt", step=5, test=payload["test"], retention=payload["retention"]
        )
        rendered = repr(manifest)
        assert "test-p0" not in rendered
        assert "longer" not in rendered


class TestThePairedMeasurementCannotDiverge:
    def test_both_kinds_share_the_step_provider_and_decode(self, session, log_path: Path) -> None:
        service.evaluate(
            spec(), 9, session=session, log_path=log_path, provider=FakeProvider(answers="longer")
        )
        rows = list(session.execute(sa.select(SCORES)).mappings())
        assert len({row["step"] for row in rows}) == 1
        assert len({row["provider"] for row in rows}) == 1
        assert len({str(row["decode"]) for row in rows}) == 1
        assert len({row["created_at"] for row in rows}) == 1

    def test_the_exposure_window_is_reported_for_a_coarse_cadence(self) -> None:
        """FR-004.26, AC-004.15. With a cadence of 5, **four** steps of training can run on a
        model already below its floor."""
        assert exposure_window_steps(5) == 4
        assert ScoringSettings(retention_every=5).retention_every == 5
