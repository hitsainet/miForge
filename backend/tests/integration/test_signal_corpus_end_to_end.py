"""A corpus part, a real batch, F1's real step  (AC-002.2, AC-002.13, AC-002.14, AC-002.18).

**Nothing here is hand-built.** The part goes in through F10's real ingest from a manifest
derived from miDataworks' own document; the batch comes out of F2's real source; the record
is written by F2's real writer; the step is F1's real ``run_one``; and the step record is
F1's real ``StepWriter``. ADR-016: *fixtures whose fields agree by construction are how a
suite stays green over a real bug* - and the fields that must not agree by construction here
are the provenance ones, which is most of what this feature produces.

The one substitution is ``torch`` itself, at the single import seam the three objectives
reach through: it is not installed on a machine without a card and the loss is the only path
that needs it (ADR-023).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from core.errors import Refusal
from tests.support import db as support_db
from tests.support import signal_fixtures as fixtures
from training.signals import batch_index, ensure_registered, registry
from training.signals.base import PartSpec, RunContext, SourceSpec
from training.signals.settings import reset_signal_settings


@pytest.fixture
def engine(tmp_path: Path) -> Any:
    engine = support_db.migrated_engine(tmp_path / "db")
    # F2's table is not in a migration: the Alembic revision graph is shared state and three
    # features are in flight, so the DDL is reported to the coordinator rather than branched
    # off the same parent. Created here, in the test, where the choice is visible.
    return engine


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("SIGNAL_SFT_MASKING_POLICY", "completion_only")
    monkeypatch.setenv("SIGNAL_PENDING_SHORTFALL", "short_batch")
    reset_signal_settings()
    yield
    reset_signal_settings()


def corpus_source(session: Any, tmp_path: Path, *, texts: list[str], **context: Any) -> Any:
    part = fixtures.corpus_part(session, tmp_path, texts=texts)
    ensure_registered()
    source = registry.get("corpus")(
        RunContext(run_id="run-1", session=session, tokenizer=fixtures.FakeTokenizer(), **context)
    )
    source.preflight(
        SourceSpec(
            source="corpus",
            parts=(PartSpec(kind="corpus", part_id=str(part["id"])),),
            batch_size=2,
        )
    )
    return source, part


class TestACorpusRunProducesRealRecordedBatches:
    def test_a_batch_names_the_part_by_contract_reference(
        self, engine: Any, tmp_path: Path
    ) -> None:
        """AC-002.2's F2 share: the batches are real and the record names the part."""
        with support_db.session_for(engine) as session:
            source, _part = corpus_source(session, tmp_path, texts=["one", "two", "three"])
            batch = source.next()
        assert batch.record["parts"] == [
            {
                "kind": "corpus",
                "contract_reference": {"type": "corpus", "name": "support-notes", "version": "1"},
                "external": None,
            }
        ]
        assert batch.record["source"] == "corpus"
        assert batch.objective == "sft"
        assert batch.size == 2
        assert [example.completion for example in batch.examples] == ["one", "two"]

    def test_the_prompt_ids_are_identifiers_resolvable_against_the_part(
        self, engine: Any, tmp_path: Path
    ) -> None:
        secret = "the-operators-private-training-text"
        with support_db.session_for(engine) as session:
            source, part = corpus_source(session, tmp_path, texts=[secret, "second"])
            batch = source.next()
        assert batch.record["prompt_ids"] == [f"{part['id']}:train:0", f"{part['id']}:train:1"]
        # The text itself is nowhere in the record - identifiers and hashes only (ADR-020).
        # A distinctive string, because "one" is a substring of "None" and an assertion that
        # can pass by accident is not an assertion.
        assert secret not in str(batch.record)
        assert batch.examples[0].completion == secret

    def test_the_binding_rides_on_the_batch(self, engine: Any, tmp_path: Path) -> None:
        """A corpus trained from ``messages`` and one trained from ``text`` are not the
        same run, so which semantic filled the role travels with the batch."""
        with support_db.session_for(engine) as session:
            source, _part = corpus_source(session, tmp_path, texts=["one", "two"])
            batch = source.next()
        assert batch.binding == {"text": "text"}
        assert batch.masking_policy == "completion_only"

    def test_the_corpus_reports_zero_candidates_rather_than_a_full_acceptance_rate(
        self, engine: Any, tmp_path: Path
    ) -> None:
        with support_db.session_for(engine) as session:
            source, _part = corpus_source(session, tmp_path, texts=["one", "two"])
            batch = source.next()
        assert batch.record["selection_health"]["candidates_generated"] == 0
        assert batch.record["selection_health"]["acceptance_rate"] == 0.0
        assert batch.health is None

    def test_an_exhausted_corpus_refuses_rather_than_yielding_a_zero_row_batch(
        self, engine: Any, tmp_path: Path
    ) -> None:
        with support_db.session_for(engine) as session:
            source, _part = corpus_source(session, tmp_path, texts=["one", "two"])
            source.next()
            with pytest.raises(Refusal) as caught:
                source.next()
        assert caught.value.code == "corpus_exhausted"


class TestTheRecordIsDurableAndIndexed:
    def test_the_log_line_and_the_index_row_agree(self, engine: Any, tmp_path: Path) -> None:
        with support_db.session_for(engine) as session:
            source, _part = corpus_source(
                session, tmp_path, texts=["one", "two"], log_root=tmp_path / "runs"
            )
            batch = source.next()
            rows = batch_index.list_batches(session, run_id="run-1", limit=10)
        assert len(rows) == 1
        assert rows[0]["batch_sha256"] == batch.record["batch_sha256"]
        assert rows[0]["source"] == "corpus"
        assert (tmp_path / "runs" / "run-1" / "run.jsonl").is_file()

    def test_dropping_the_index_and_rebuilding_from_the_log_reproduces_it_exactly(
        self, engine: Any, tmp_path: Path
    ) -> None:
        """AC-002.18 / ADR-016 #4, over this feature's own index.

        The rebuild reads **the run log alone**, which is what makes ADR-004's claim - *the
        filesystem is the truth and the database is a rebuildable index* - true rather than
        asserted. The comparison is over the API payloads, because that is what a consumer
        sees: two projections that differ only in a field nobody serves would still be a
        divergence.
        """
        from api.v1.endpoints.signal_batches import _row_payload

        with support_db.session_for(engine) as session:
            source, _part = corpus_source(
                session, tmp_path, texts=["one", "two", "three", "four"], log_root=tmp_path / "runs"
            )
            source.next()
            source.next()
            before = [
                _row_payload(row)
                for row in batch_index.list_batches(session, run_id="run-1", limit=10)
            ]

        with support_db.session_for(engine) as session:
            session.execute(sa.delete(batch_index.BATCHES))

        with support_db.session_for(engine) as session:
            assert batch_index.list_batches(session, run_id="run-1", limit=10) == []
            count = batch_index.rebuild_batch_index(
                session, log_root=tmp_path / "runs", run_id="run-1"
            )
            after = [
                _row_payload(row)
                for row in batch_index.list_batches(session, run_id="run-1", limit=10)
            ]

        assert count == 2
        assert after == before

    def test_a_rebuild_is_idempotent(self, engine: Any, tmp_path: Path) -> None:
        """An insert-only rebuild would make the equivalence test pass once and never
        again."""
        with support_db.session_for(engine) as session:
            source, _part = corpus_source(
                session, tmp_path, texts=["one", "two"], log_root=tmp_path / "runs"
            )
            source.next()
        with support_db.session_for(engine) as session:
            batch_index.rebuild_batch_index(session, log_root=tmp_path / "runs", run_id="run-1")
            batch_index.rebuild_batch_index(session, log_root=tmp_path / "runs", run_id="run-1")
            assert len(batch_index.list_batches(session, run_id="run-1", limit=10)) == 1

    def test_a_rebuild_with_no_log_refuses_rather_than_asserting_an_empty_run(
        self, engine: Any, tmp_path: Path
    ) -> None:
        with support_db.session_for(engine) as session, pytest.raises(Refusal) as caught:
            batch_index.rebuild_batch_index(session, log_root=tmp_path / "nothing", run_id="run-9")
        assert caught.value.code == "run_log_absent"


class TestIntoF1sRealStep:
    def test_the_step_record_cites_the_batch_record_hash(
        self, engine: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**AC-002.14, through both real writers.** The join between a step and the
        material it consumed is that hash and nothing else (R-02.14).

        Asserted end to end: F2's source produces the batch, F1's ``run_one`` consumes it
        through the **real** objective registry, and F1's ``StepWriter`` writes the step
        block - so the two records are joined by the writers, not by two hand-built dicts.
        """
        from training.objectives import base as objective_base
        from training.objectives import sft
        from training.run_config import RunConfig
        from training.step import run_one
        from training.step_record import StepWriter

        torch = fixtures.FakeTorch()
        monkeypatch.setattr(sft, "torch_module", lambda: torch)
        monkeypatch.setattr(objective_base, "torch_module", lambda: torch)

        model = fixtures.FakeModel()
        optimizer = fixtures.FakeOptimizer(model.parameter_list)
        with support_db.session_for(engine) as session:
            source, _part = corpus_source(
                session, tmp_path, texts=["one", "two"], log_root=tmp_path / "runs"
            )
            outcome = run_one(model, optimizer, source)

            writer = StepWriter(run_id="run-1", log_root=tmp_path / "runs")
            block = writer.write(
                session=None,
                step=1,
                outcome=outcome,
                cfg=RunConfig(learning_rate=1e-5, warmup_steps=0, batch_size=2),
            )

        assert outcome.objective == "sft"
        assert block["batch_sha256"] == outcome.batch_sha256
        assert len(torch.cross_entropy_calls) == 1
        assert optimizer.steps == 1

        # And the hash in the step block is the one the batch record carries.
        from training import run_log

        batches = list(
            run_log.read_blocks(tmp_path / "runs" / "run-1" / "run.jsonl", block="batch")
        )
        assert batches[0]["batch"]["batch_sha256"] == block["batch_sha256"]

    def test_the_run_log_carries_both_blocks_discriminated(
        self, engine: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F1 and F2 write to one file, interleaved, and the discriminator is in the line -
        because reindex rebuilds from the log alone and classifies a line it is holding."""
        from training import run_log
        from training.objectives import sft
        from training.run_config import RunConfig
        from training.step import run_one
        from training.step_record import StepWriter

        monkeypatch.setattr(sft, "torch_module", lambda: fixtures.FakeTorch())
        model = fixtures.FakeModel()
        with support_db.session_for(engine) as session:
            source, _part = corpus_source(
                session, tmp_path, texts=["one", "two"], log_root=tmp_path / "runs"
            )
            outcome = run_one(model, fixtures.FakeOptimizer(model.parameter_list), source)
            StepWriter(run_id="run-1", log_root=tmp_path / "runs").write(
                session=None,
                step=1,
                outcome=outcome,
                cfg=RunConfig(learning_rate=1e-5, warmup_steps=0, batch_size=2),
            )
        blocks = [
            b["block"] for b in run_log.read_blocks(tmp_path / "runs" / "run-1" / "run.jsonl")
        ]
        assert blocks == ["batch", "step"]


class TestPreflightRefusesBeforeTheRunStarts:
    def test_a_missing_masking_policy_refuses_at_preflight_without_a_gpu(
        self, engine: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """E-5 / FR-002.4: refused at submit, not at step 1 - and with no card present."""
        monkeypatch.delenv("SIGNAL_SFT_MASKING_POLICY", raising=False)
        reset_signal_settings()
        with support_db.session_for(engine) as session:
            part = fixtures.corpus_part(session, tmp_path, texts=["one"])
            ensure_registered()
            source = registry.get("corpus")(RunContext(run_id="run-1", session=session))
            with pytest.raises(Refusal) as caught:
                source.preflight(
                    SourceSpec(
                        source="corpus", parts=(PartSpec(kind="corpus", part_id=str(part["id"])),)
                    )
                )
        assert caught.value.code == "sft_masking_policy_not_configured"

    def test_a_source_asked_for_a_batch_before_preflight_refuses(self) -> None:
        ensure_registered()
        source = registry.get("corpus")(RunContext())
        with pytest.raises(Refusal) as caught:
            source.next()
        assert caught.value.code == "signal_source_not_preflighted"

    def test_a_spec_naming_another_source_is_refused(self) -> None:
        ensure_registered()
        source = registry.get("corpus")(RunContext())
        with pytest.raises(Refusal) as caught:
            source.preflight(SourceSpec(source="reward"))
        assert caught.value.code == "signal_source_spec_mismatch"

    def test_a_run_with_no_part_refuses_naming_the_kind_it_needs(self) -> None:
        ensure_registered()
        source = registry.get("corpus")(RunContext())
        with pytest.raises(Refusal) as caught:
            source.preflight(SourceSpec(source="corpus"))
        assert caught.value.code == "signal_source_no_part"
        assert caught.value.details["required_kind"] == "corpus"
