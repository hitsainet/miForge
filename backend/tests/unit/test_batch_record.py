"""The batch record: the contract, the hash, the order, and **no text**  (FR-002.28-32).

Three tests here are load-bearing:

* :meth:`TestNoTextAnywhere.test_no_text_in_record_log_or_index` - **mutation control M-6**,
  the one data-minimisation guarantee in this feature. Add the first completion's text to
  the summary and it must go red. *The run log leaves this machine in exports and support
  bundles, and user prompt text in it would be a privacy breach no schema check downstream
  could detect. A sibling shipped exactly that bug by flipping one argument and its suite
  stayed green.*
* :meth:`TestTheOrder.test_the_record_is_durable_before_it_is_indexed_or_emitted` - an
  inverted order changes nothing observable until the machine dies, so the **sequence** is
  asserted rather than the outcome.
* :meth:`TestTheSourceVocabulary.test_a_source_the_contract_cannot_record_is_refused` - the
  refusal that keeps a plugin's fifth source from being written down under another source's
  name. *This bullet read "the live gate-contract gap, pinned so that closing it upstream is
  visible here" until 2026-10-16. The gap closed on 2026-10-14, the pin fired into an early
  return, and the test stopped executing its own assertions - see its docstring.*
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.errors import Refusal
from training import run_log
from training.signals import batch_record
from training.signals.base import BatchCounts


def record(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "source": "self-generation",
        "prompt_ids": ["p-1", "p-2"],
        "parts": [
            {
                "kind": "prompt_set",
                "contract_reference": {"type": "prompt-set", "name": "legal", "version": "3"},
                "external": None,
            }
        ],
        "counts": BatchCounts(generated=4, admitted=2, rejected=1, pending=1),
        "resolved_model": "in_process:lfm2-1.2b@12",
    }
    kwargs.update(overrides)
    return batch_record.assemble(**kwargs)


class TestTheContract:
    def test_an_assembled_record_validates_against_the_gate_contract(self) -> None:
        batch_record.validate(record())

    def test_the_three_verdict_counts_are_all_present(self) -> None:
        """THREE counts, never two: a pending candidate miscounted as a rejection makes the
        health record lie about an unattended queue."""
        assert record()["verdicts"] == {"accept": 2, "reject": 1, "undecided": 1}

    def test_the_acceptance_rate_names_the_sample_it_was_taken_over(self) -> None:
        health = record()["selection_health"]
        assert health["candidates_generated"] == 4
        assert health["candidates_admitted"] == 2
        assert health["acceptance_rate"] == 0.5

    def test_a_corpus_batch_reports_zero_candidates_rather_than_a_full_rate(self) -> None:
        """A corpus runs no selection. Reporting ``1.0`` over the rows it read would assert
        that *a selector accepted everything* - one of the three named degenerate signals -
        about a batch where no selector ran at all."""
        built = record(
            source="corpus", counts=BatchCounts(generated=0, admitted=8), resolved_model=None
        )
        assert built["selection_health"] == {
            "candidates_generated": 0,
            "candidates_admitted": 0,
            "acceptance_rate": 0.0,
            "exhausted": False,
        }
        assert built["verdicts"] == {"accept": 8, "reject": 0, "undecided": 0}

    def test_an_invalid_record_is_refused_naming_the_field(self) -> None:
        """E-17: a record a reader cannot trust is worse than a gap."""
        broken = record()
        broken["prompt_ids"] = "p-1"
        with pytest.raises(Refusal) as caught:
            batch_record.validate(broken)
        assert caught.value.code == "batch_record_invalid"
        assert caught.value.details["pointer"] == "prompt_ids"

    def test_an_invented_key_is_a_validation_failure(self) -> None:
        """The contract's objects are closed, which is what makes a smuggled field loud."""
        broken = record()
        broken["objective"] = "sft"
        with pytest.raises(Refusal) as caught:
            batch_record.validate(broken)
        assert caught.value.code == "batch_record_invalid"


class TestTheHash:
    def test_the_hash_covers_the_record_without_itself(self) -> None:
        built = record()
        assert batch_record.verify(built)
        assert built["batch_sha256"] == batch_record.content_hash(built)

    def test_two_records_with_different_content_hash_differently(self) -> None:
        assert record()["batch_sha256"] != record(prompt_ids=["p-9"])["batch_sha256"]

    def test_the_hash_is_stable_across_assemblies(self) -> None:
        """It is the batch's identity and the only join between a step and its material."""
        assert record()["batch_sha256"] == record()["batch_sha256"]

    def test_a_tampered_record_fails_verification(self) -> None:
        built = record()
        built["prompt_ids"] = ["p-9"]
        with pytest.raises(Refusal) as caught:
            batch_record.validate(built)
        assert caught.value.code == "batch_record_hash_mismatch"

    def test_the_hash_goes_through_the_one_canonical_function(self) -> None:
        """ADR-005: two canonicalisers are two answers. Asserted by computing the digest
        through ``core.canonical`` here and requiring the writer's to equal it."""
        from core.canonical import sha256_of

        built = record()
        expected = sha256_of({k: v for k, v in built.items() if k != "batch_sha256"})
        assert built["batch_sha256"] == expected


class TestTheSourceVocabulary:
    @pytest.mark.parametrize(
        "name,value",
        [
            ("corpus", "corpus"),
            ("self-generation", "self-generation"),
            ("preference-pairs", "preference-pairs"),
            ("reward", "reward"),
        ],
    )
    def test_a_registry_name_is_the_contracts_value_unchanged(self, name: str, value: str) -> None:
        """**One vocabulary, so there is nothing to translate.**

        This asserted a *mapping* - hyphens to underscores - because the two enums genuinely
        disagreed, sharing exactly one value out of four. That disagreement was the defect,
        not the mapping: the record's enum has been aligned to the manifest's spelling,
        ``reward`` added and ``mixed`` removed, so a registry name IS the contract's value.

        ``reward`` is in the list now, which is the half that was unbuildable before.
        """
        assert batch_record.record_source(name) == value

    #: A source name outside ``batch-record-v1.json``'s closed ``source`` enum. The same
    #: name the sibling integration test registers as a plugin source, so the two halves
    #: of this refusal - the function and the path that reaches it - describe one case.
    #: **Not imported from there**: a unit test that imports an integration module drags a
    #: database fixture into a pure-function test. The premise is re-proved below instead,
    #: which is the thing that would actually matter if the enum ever carried the name.
    UNRECORDABLE = "handwritten-signal"

    def test_a_source_the_contract_cannot_record_is_refused(self) -> None:
        """**A source outside the closed enum is refused, naming what IS recordable.**

        *This test asserted the opposite until 2026-10-16 and had stopped asserting
        anything at all.* It was written when ``batch-record-v1.json``'s ``source`` enum was
        ``corpus | self_generation | preference_pairs | mixed`` and carried no ``reward``,
        so a reward batch could not be recorded; it opened ``if "reward" in enum: ... return``
        as a tripwire for the day the gap closed. **The gap closed on 2026-10-14 and the
        tripwire fired into the early-return branch**, so from that day the refusal below
        was never executed on any machine - and the dead branch asserted
        ``details["contract_value"]``, **a key production has never set**
        (:func:`training.signals.batch_record.record_source` passes ``source``, ``enum`` and
        ``schema``). Two defects covering for each other: the assertion that was wrong was
        the one nothing ran.

        **The refusal is live code, not a relic.** No *built-in* source can reach it -
        ``registry.names()`` is asserted equal to the enum - but
        ``training.signals.registry.register`` is public and takes any name, so a plugin's
        fifth source is unrecordable and the run that used it unsaveable. This asserts the
        function; ``tests/integration/test_signal_reward_source.py`` asserts that pre-flight
        reaches it before a single part is resolved.
        """
        enum = batch_record.source_enum()
        assert self.UNRECORDABLE not in enum, (
            "this test's premise is that the name is outside the contract's enum; if the "
            "enum ever carries it, choose another name rather than letting the refusal go "
            "unexercised - which is exactly how this test died once already"
        )
        with pytest.raises(Refusal) as caught:
            batch_record.record_source(self.UNRECORDABLE)
        assert caught.value.code == "batch_source_not_in_contract"
        # The real keys, asserted by their real names. `details` carries no
        # `contract_value` and never has.
        assert caught.value.details["source"] == self.UNRECORDABLE
        assert sorted(caught.value.details["enum"]) == sorted(enum)
        assert caught.value.details["schema"] == "batch-record-v1.json"
        assert "contract_value" not in caught.value.details
        # The operator is told what IS possible, not only what failed (ADR-013, ADR-018) -
        # and `reward` being among them is the half that was unbuildable before.
        assert "reward" in caught.value.details["enum"]

    def test_every_registered_source_is_recordable(self) -> None:
        """The other side of the same asymmetry, and the reason no built-in hits the
        refusal above.

        Asserted against the two live registries rather than against a list here: a
        retyped list of four names is a third vocabulary, which is the family that produced
        ``step-`` versus ``000500`` and ``full`` versus ``change``.
        """
        from training.signals import ensure_registered, registry

        ensure_registered()
        assert set(registry.names()) == set(batch_record.source_enum())
        for name in registry.names():
            assert batch_record.record_source(name) == name

    def test_the_enum_is_read_from_the_file_rather_than_retyped(self) -> None:
        schema = json.loads(batch_record.schema_path().read_text("utf-8"))
        declared = schema["$defs"]["BatchRecordV1"]["properties"]["source"]["enum"]
        assert batch_record.source_enum() == frozenset(declared)


class TestNoTextAnywhere:
    def test_no_text_in_record_log_or_index(self, tmp_path: Path) -> None:
        """**Mutation control M-6** - the one data-minimisation guarantee in this feature.

        The completion text appears nowhere: not in the record, not in the run-log line, not
        in the index row and not in the socket summary. It is asserted over the **serialised
        bytes** of each, because a field nested three deep is still a leak.
        """
        secret = "the-users-private-prompt-text"
        built = record()
        emitted: list[tuple[str, str, dict[str, Any]]] = []
        counts = BatchCounts(generated=4, admitted=2, rejected=1, pending=1)
        batch_record.write(
            built,
            run_id="run-1",
            step=3,
            counts=counts,
            objective="sft",
            log_root=tmp_path,
            emit=lambda run_id, event, payload: emitted.append((run_id, event, payload)),
        )
        line = (tmp_path / "run-1" / "run.jsonl").read_text("utf-8")
        summary = json.dumps(emitted[0][2])
        for blob in (json.dumps(built), line, summary):
            assert secret not in blob
            # Nothing that could be a completion: the record's keys are the contract's and
            # none of them is free text.
            assert "completion_text" not in blob

    def test_the_summary_reaches_no_content(self) -> None:
        summary = batch_record.summary_of(
            record(),
            counts=BatchCounts(generated=4, admitted=2, ties=1, dropped=2),
            objective="sft",
        )
        assert set(summary) == {
            "batch_sha256",
            "source",
            "objective",
            "resolved_model",
            "admitted",
            "rejected",
            "pending",
            "dropped",
            "ties",
            "candidates_generated",
            "exhausted",
        }

    def test_the_index_row_has_no_text_column(self) -> None:
        """The absence **is** the guarantee. Read from the table, not from a copy of it."""
        from training.signals.batch_index import text_free_columns

        forbidden = {"prompt", "completion", "sample", "text", "body", "messages"}
        assert not forbidden & set(text_free_columns())


class TestTheOrder:
    def test_the_record_is_durable_before_it_is_indexed_or_emitted(self, tmp_path: Path) -> None:
        """A crash between the log and the index costs a reindex; the other way round costs
        **the record**."""
        effects: list[str] = []
        session = _RecordingSession(effects)
        batch_record.write(
            record(),
            run_id="run-1",
            step=1,
            counts=BatchCounts(generated=1, admitted=1),
            objective="sft",
            log_root=tmp_path,
            session=session,
            emit=lambda *args: effects.append("socket-call"),
            heartbeat=_Heartbeat(effects),
            effects=effects,
        )
        assert [e for e in effects if e in ("log", "index", "socket", "heartbeat")] == [
            "log",
            "index",
            "socket",
            "heartbeat",
        ]

    def test_an_invalid_record_writes_nothing_at_all(self, tmp_path: Path) -> None:
        """E-17 / AC-002.13: not written, not indexed, not emitted, not returned."""
        broken = record()
        broken["source"] = "not-a-source"
        emitted: list[Any] = []
        with pytest.raises(Refusal):
            batch_record.write(
                broken,
                run_id="run-1",
                step=1,
                counts=BatchCounts(),
                objective="sft",
                log_root=tmp_path,
                emit=lambda *args: emitted.append(args),
            )
        assert not (tmp_path / "run-1" / "run.jsonl").exists()
        assert emitted == []

    def test_the_line_is_a_batch_block_of_the_run_log(self, tmp_path: Path) -> None:
        """The fifth block of one log, discriminated **in the line** - because reindex
        classifies a line it is holding."""
        built = record()
        batch_record.write(
            built,
            run_id="run-1",
            step=7,
            counts=BatchCounts(generated=1, admitted=1),
            objective="sft",
            log_root=tmp_path,
        )
        blocks = list(run_log.read_blocks(tmp_path / "run-1" / "run.jsonl"))
        assert [block["block"] for block in blocks] == ["batch"]
        assert blocks[0]["step"] == 7
        assert blocks[0]["batch"]["batch_sha256"] == built["batch_sha256"]

    def test_the_block_validates_against_the_step_record_schema(self, tmp_path: Path) -> None:
        """The payload is a ``$ref`` into the batch contract - one definition, referenced."""
        run_log.validate_block(batch_record.block(record(), run_id="run-1", step=1))


class _RecordingSession:
    def __init__(self, effects: list[str]) -> None:
        self.effects = effects

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        self.effects.append("sql")
        return None


class _Heartbeat:
    def __init__(self, effects: list[str]) -> None:
        self.effects = effects

    def touch(self, session: Any) -> None:
        self.effects.append("heartbeat-call")
