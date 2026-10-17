"""Run log, then index, then socket  (FR-004.28, 29, 30; ADR-004, ADR-007; M-2, M-5).

**That order, not another.** The on-disk run log is authoritative, ``mf_scores`` is the index
over it, and the socket emission is a convenience. A crash between them loses a projection,
never a record. The reason is recorded rather than assumed: *"progress reported only over the
socket is invisible to the janitor"*, which is how miStudio reaped a live 5.8-hour job as
"worker lost" - and a generation-bound pass over a large test set is exactly that shape.

A breach is published the same way (M-2: **a detected breach nobody is told about is the same
failure as not detecting it**), and every publication is asserted on **payload and call
count**, because *"was called" passes against a call sending the wrong arguments* - which is
how a breach could be published for the wrong step.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa

from providers.base import ProviderIdentity
from services.scoring import record as record_module
from services.scoring.retention import BREACH_EVENT, BreachSignal
from services.scoring.tables import SCORE_ITEMS, SCORES
from tests.support import scores as fixtures
from tests.support.db import migrated_engine, session_for


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


def a_breach() -> BreachSignal:
    return BreachSignal(
        run_id="run-1",
        step=7,
        kind="retention",
        score=0.60,
        floor=0.72,
        n=180,
        provider="in_process:fake-model",
        set_ref="retention_set/general@1",
        floor_layer="pack_slot",
    )


class TestTheWriteOrderIsDurableFirst:
    def test_the_run_log_line_exists_before_anything_else_could_fail(
        self, session, log_path: Path
    ) -> None:
        emitter = fixtures.RecordingEmitter()
        record_module.write_pass(
            [fixtures.score_record(step=7)], session=session, log_path=log_path, emit=emitter
        )
        assert log_path.is_file()
        assert emitter.count == 1

    def test_a_socket_only_publish_would_leave_no_record(self, session, log_path: Path) -> None:
        """The property, asserted from the other side: with no emitter at all the pass still
        completes and the record still exists.

        That is what makes the socket a convenience rather than the record - and it is also
        what lets a worker with no Redis reachable finish a pass.
        """
        record_module.write_pass(
            [fixtures.score_record(step=3)], session=session, log_path=log_path, emit=None
        )
        assert log_path.read_text().strip()
        assert session.execute(sa.select(sa.func.count()).select_from(SCORES)).scalar_one() == 1

    def test_the_index_row_matches_the_record(self, session, log_path: Path) -> None:
        record_module.write_pass(
            [fixtures.score_record(step=7, score=0.84, n=200)],
            session=session,
            log_path=log_path,
        )
        row = session.execute(sa.select(SCORES)).mappings().one()
        assert (row["run_id"], row["step"], row["kind"]) == ("run-1", 7, "test")
        assert (row["score"], row["n"]) == (0.84, 200)

    def test_the_heartbeat_is_advanced_during_a_pass(self, session, log_path: Path) -> None:
        """*A long in-process pass is exactly the job shape miStudio reaped as "worker
        lost"* when it reported only over the socket."""

        class Beat:
            def __init__(self) -> None:
                self.touches = 0

            def touch(self, session) -> bool:
                self.touches += 1
                return True

        beat = Beat()
        record_module.write_pass(
            [fixtures.score_record()], session=session, log_path=log_path, heartbeat=beat
        )
        assert beat.touches == 1


class TestEveryScoreRecordsItsProvider:
    """M-5: **two requirements demand it**, and a score without it is comparable with nothing."""

    def test_the_row_carries_the_provider_in_both_forms(self, session, log_path: Path) -> None:
        record_module.write_pass([fixtures.score_record()], session=session, log_path=log_path)
        row = session.execute(sa.select(SCORES)).mappings().one()
        assert row["provider"] == "in_process:fake-model"
        assert row["provider_kind"] == "in_process"
        assert row["provider_model"] == "fake-model"
        assert row["parameter_set_digest"] == "fake-parameter-set@step0"

    def test_the_run_log_entry_carries_it_too(self, session, log_path: Path) -> None:
        """``ScoreEntry.provider`` is in the block's ``required`` list - the run log is
        **stricter** than the manifest, deliberately, because it is the authoritative copy."""
        block = record_module.write_pass(
            [fixtures.score_record()], session=session, log_path=log_path
        )
        assert block["scores"][0]["provider"] == "in_process:fake-model"
        assert block["provider"] == "in_process:fake-model"

    def test_the_api_payload_always_carries_n_and_the_provider(self) -> None:
        """ADR-012 enforced by the payload, so **F9 cannot render a bare number**."""
        payload = fixtures.score_record().api_payload()
        assert payload["n"] == 200
        assert payload["provider"] == "in_process:fake-model"

    def test_an_http_providers_rendering_carries_the_endpoint(self) -> None:
        """Two scores from two endpoints serving one model name are not interchangeable."""
        assert fixtures.HTTP.record() == "openai_compatible:judge-7b@http://judge.test:8000"


class TestTheBreachPublication:
    """M-2, asserted on payload **and** call count."""

    def test_the_breach_is_emitted_after_the_durable_write(self, session, log_path: Path) -> None:
        emitter = fixtures.RecordingEmitter()
        record_module.write_pass(
            [fixtures.score_record(kind="retention", floor=0.72, score=0.6, breached=True)],
            session=session,
            log_path=log_path,
            breach=a_breach(),
            emit=emitter,
        )
        assert log_path.is_file()
        assert [event for _r, event, _p in emitter.calls] == ["score", BREACH_EVENT]

    def test_the_payload_field_by_field_and_the_count(self, session, log_path: Path) -> None:
        emitter = fixtures.RecordingEmitter()
        record_module.write_pass(
            [fixtures.score_record(kind="retention", floor=0.72, score=0.6, breached=True)],
            session=session,
            log_path=log_path,
            breach=a_breach(),
            emit=emitter,
        )
        published = emitter.events(BREACH_EVENT)
        assert len(published) == 1, "one publication per breach, counted"
        payload = published[0]
        assert payload["step"] == 7
        assert payload["score"] == 0.60
        assert payload["floor"] == 0.72
        assert payload["n"] == 180
        assert payload["provider"] == "in_process:fake-model"
        assert payload["set"] == "retention_set/general@1"

    def test_the_block_records_the_breach_as_a_fact_on_the_pass(
        self, session, log_path: Path
    ) -> None:
        """The schema's own words: *"recorded as a FACT on the pass rather than derived later
        by comparing two numbers, so the decision the loop took is reconstructible from the
        log alone."*"""
        block = record_module.write_pass(
            [fixtures.score_record(kind="retention", floor=0.72, score=0.6, breached=True)],
            session=session,
            log_path=log_path,
            breach=a_breach(),
        )
        assert block["retention_floor_breached"] is True

    def test_no_breach_records_the_fact_as_false_rather_than_omitting_it(
        self, session, log_path: Path
    ) -> None:
        block = record_module.write_pass(
            [fixtures.score_record()], session=session, log_path=log_path
        )
        assert block["retention_floor_breached"] is False


class TestTheManifestPublicationSeam:
    """FR-004.30. **Removing this call must go red** - that control is run and recorded."""

    def test_one_call_per_save_with_both_arguments(self, session, log_path: Path) -> None:
        record_module.write_pass(
            [
                fixtures.score_record(step=5, kind="test", score=0.84, n=200),
                fixtures.score_record(step=5, kind="retention", score=0.77, n=180, floor=0.72),
            ],
            session=session,
            log_path=log_path,
        )
        publisher = fixtures.RecordingPublisher()
        record_module.publish_to_manifest(session, run_id="run-1", step=5, publish=publisher)
        assert publisher.count == 1, "one call per save, counted - not 'was called'"
        test, retention = publisher.calls[0]
        assert test is not None and retention is not None
        assert test["set"] == "test_set/general@1"
        assert test["score"] == 0.84
        assert test["n"] == 200
        assert test["provider"] == "in_process:fake-model"
        assert retention["floor"] == 0.72

    def test_a_non_evaluation_step_publishes_both_nulls_PRESENT(
        self, session, log_path: Path
    ) -> None:
        """E-22. ``null`` asserts nothing ran; an absent key is an omission."""
        publisher = fixtures.RecordingPublisher()
        record_module.publish_to_manifest(session, run_id="run-1", step=999, publish=publisher)
        assert publisher.calls == [(None, None)]

    def test_an_out_of_cadence_score_never_reaches_a_manifest(
        self, session, log_path: Path
    ) -> None:
        """*An operator's ad-hoc check becoming a data point is how a curve acquires a point
        nobody can explain* - and a manifest is worse, because it travels."""
        record_module.write_pass(
            [fixtures.score_record(step=5, out_of_cadence=True)],
            session=session,
            log_path=log_path,
        )
        publisher = fixtures.RecordingPublisher()
        record_module.publish_to_manifest(session, run_id="run-1", step=5, publish=publisher)
        assert publisher.calls == [(None, None)]

    def test_the_latest_rescore_of_a_step_wins(self, session, log_path: Path) -> None:
        """History appends and **the later line is the later fact**."""
        from datetime import UTC, datetime, timedelta

        base = datetime(2026, 10, 14, tzinfo=UTC)
        record_module.write_pass(
            [fixtures.score_record(step=5, score=0.10, created_at=base)],
            session=session,
            log_path=log_path,
        )
        record_module.write_pass(
            [fixtures.score_record(step=5, score=0.90, created_at=base + timedelta(seconds=1))],
            session=session,
            log_path=log_path,
        )
        publisher = fixtures.RecordingPublisher()
        record_module.publish_to_manifest(session, run_id="run-1", step=5, publish=publisher)
        assert publisher.calls[0][0]["score"] == 0.90


class TestARescoreAppends:
    def test_two_passes_at_one_step_are_two_rows(self, session, log_path: Path) -> None:
        from datetime import UTC, datetime, timedelta

        base = datetime(2026, 10, 14, tzinfo=UTC)
        for offset in (0, 1):
            record_module.write_pass(
                [fixtures.score_record(step=5, created_at=base + timedelta(seconds=offset))],
                session=session,
                log_path=log_path,
            )
        count = session.execute(sa.select(sa.func.count()).select_from(SCORES)).scalar_one()
        assert count == 2, "a mutated history is worse than a duplicated one"

    def test_the_run_log_keeps_both_lines(self, session, log_path: Path) -> None:
        from datetime import UTC, datetime, timedelta

        base = datetime(2026, 10, 14, tzinfo=UTC)
        for offset in (0, 1):
            record_module.write_pass(
                [fixtures.score_record(step=5, created_at=base + timedelta(seconds=offset))],
                session=session,
                log_path=log_path,
            )
        assert len(log_path.read_text().strip().splitlines()) == 2


class TestPerPromptDetail:
    def test_items_are_written_and_joined_to_the_score(self, session, log_path: Path) -> None:
        from services.scoring.expectations import PromptOutcome

        outcomes = (
            PromptOutcome(index=0, outcome=1.0, verdict="accept", scalar=None, reason_code="ok"),
            PromptOutcome(index=1, outcome=0.0, verdict="reject", scalar=None, reason_code="short"),
        )
        record_module.write_pass(
            [fixtures.score_record(outcomes=outcomes, completions=("a" * 900, "b"))],
            session=session,
            log_path=log_path,
        )
        items = list(
            session.execute(sa.select(SCORE_ITEMS).order_by(SCORE_ITEMS.c.prompt_index)).mappings()
        )
        assert [item["verdict"] for item in items] == ["accept", "reject"]
        assert len(items[0]["completion_excerpt"]) == record_module.COMPLETION_EXCERPT_CHARS

    def test_no_prompt_or_completion_text_reaches_the_manifest_payload(self) -> None:
        """ADR-020: a manifest travels with the artifact and is published beside it."""
        payload = fixtures.score_record(completions=("secret completion",)).score_record()
        assert "secret" not in repr(payload)
        assert set(payload) == {"set", "score", "n", "floor", "provider"}


class TestNoCredentialAnywhereInARecord:
    def test_not_in_the_row_the_block_or_the_payload(self, session, log_path: Path) -> None:
        """E-23, ADR-020. ``ProviderIdentity`` has no credential field, so this asserts the
        type's guarantee holds all the way to the three surfaces."""
        identity = ProviderIdentity(
            kind="openai_compatible", model="judge-7b", base_url="http://judge.test:8000"
        )
        block = record_module.write_pass(
            [fixtures.score_record(provider=identity)], session=session, log_path=log_path
        )
        row = session.execute(sa.select(SCORES)).mappings().one()
        for surface in (repr(dict(row)), repr(block), log_path.read_text()):
            assert "api_key" not in surface
            assert "Bearer" not in surface
