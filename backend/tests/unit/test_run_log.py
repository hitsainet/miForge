"""The authoritative run log  (ADR-004; FR-001.24 - FR-001.27, AC-001.6).

**Every record here round-trips through the real writer and the real reader.** A hand-built
step-record dict whose fields agree with the assertion by construction is how a suite stays
green over a real bug (ADR-016), and this file's whole subject is whether what was written
is what comes back.

Three things it pins:

* **The step block validates against ``docs/schemas/step-record-v1.json``** - the contract,
  which F1 does not define. An invalid block is refused and **nothing is written**.
* **The two §5.10 fields are present and unset, never dropped.** ``commit_duration_ms`` feeds
  another repository's acceptance gate, and the checkpoint manifest's ``domain`` field states
  the principle: *null is a statement, an absent key is an omission.*
* **``commit_duration_ms`` is queryable as a distribution** (FR-001.26), not merely as a last
  value - miLLM Feature 31's watchdog is ten times its p99.9.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.errors import Refusal
from training import run_log

RUN = "run-abc"


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return run_log.run_log_path(tmp_path, RUN)


def block(**overrides) -> dict:
    base = run_log.step_block(
        run_id=RUN,
        step=1,
        loss=0.5,
        objective="sft",
        learning_rate=1e-5,
        wall_time_ms=123.5,
        grad_norm=1.75,
        batch_sha256="a" * 64,
    )
    base.update(overrides)
    return base


class TestTheSchemaIsTheContract:
    def test_a_well_formed_step_block_validates(self) -> None:
        run_log.validate_block(block())

    def test_the_schema_is_found_where_the_contract_lives(self) -> None:
        """``docs/schemas/`` is the one ``docs/`` path published to the public mirror,
        because it is a public contract - so it travels with the repository rather than
        being configured."""
        assert run_log.schema_path().is_file()
        assert json.loads(run_log.schema_path().read_text())["$id"] == "step-record-v1.json"

    def test_an_unknown_block_discriminator_is_refused(self) -> None:
        """The discriminator must be in the line: ``reindex`` rebuilds from this log alone
        and classifies each line it is holding, rather than knowing which file it came from.
        """
        with pytest.raises(Refusal) as caught:
            run_log.validate_block({**block(), "block": "telemetry"})
        assert caught.value.code == "step_record_invalid"
        assert "telemetry" in caught.value.message

    def test_every_discriminator_has_a_subschema_so_refusals_can_name_a_field(self) -> None:
        """The two maps must agree. A block with a writer but no ``$defs`` entry would raise
        a ``KeyError`` inside the validator - turning "your record is malformed" into a 500
        with a traceback, which is the opposite of reporting it."""
        assert set(run_log.BLOCK_WRITERS) == set(run_log.BLOCK_DEFINITIONS)
        document = json.loads(run_log.schema_path().read_text())
        for block, name in run_log.BLOCK_DEFINITIONS.items():
            assert name in document["$defs"], f"{block!r} maps to a missing $defs entry"

    def test_the_five_blocks_and_their_writers_are_enumerated(self) -> None:
        """Four features write five things to one log; before this was settled three of them
        had nowhere defined to put theirs."""
        assert run_log.BLOCK_WRITERS == {
            "step": "F1",
            "evaluation": "F4",
            "boundary": "F6",
            "definition": "F6",
            "batch": "F2",
        }

    def test_a_field_the_schema_does_not_have_is_refused(self) -> None:
        """``additionalProperties: false``. R-02.35's rule that a record's fields are
        *exactly* the schema's is enforced by the schema, not by reviewer attention - before
        2026-10-13 neither vendored schema closed its objects and an invented key validated
        clean."""
        with pytest.raises(Refusal) as caught:
            run_log.validate_block({**block(), "gpu_temperature_c": 71})
        assert caught.value.code == "step_record_invalid"

    @pytest.mark.parametrize("missing", ["loss", "objective", "learning_rate", "wall_time_ms"])
    def test_each_required_field_is_required(self, missing: str) -> None:
        payload = block()
        del payload[missing]
        with pytest.raises(Refusal) as caught:
            run_log.validate_block(payload)
        assert caught.value.code == "step_record_invalid"
        assert missing in caught.value.message

    def test_an_invalid_block_writes_NOTHING(self, log_path: Path) -> None:
        """A record a reader cannot trust is worse than a gap, because the gap is visible."""
        with pytest.raises(Refusal):
            run_log.append(log_path, {**block(), "loss": "not a number"})
        assert not log_path.exists(), "a refused write must leave no trace"


class TestTheTwoSectionFiveTenFields:
    def test_they_are_PRESENT_and_null_on_every_step_F1_writes(self) -> None:
        """FR-001.25. They measure a protocol with a serving process that does not exist on
        the default path, and F11 populates them while defining neither."""
        payload = block()
        assert payload["quiesce_hold_ms"] is None
        assert payload["commit_duration_ms"] is None
        assert "quiesce_hold_ms" in payload and "commit_duration_ms" in payload

    def test_a_block_without_them_still_validates_which_is_why_a_TEST_is_needed(self) -> None:
        """**The schema cannot catch this.** Both fields are nullable *and not required*, so
        a writer that dropped them validates clean and the sync test - which compares schema
        *files* - cannot see it either. This assertion is the only thing standing between
        that and miLLM Feature 31's gate losing its input."""
        stripped = {
            k: v for k, v in block().items() if k not in {"quiesce_hold_ms", "commit_duration_ms"}
        }
        run_log.validate_block(stripped)  # the schema is satisfied...
        assert set(block()) - set(stripped) == {"quiesce_hold_ms", "commit_duration_ms"}

    def test_F11_can_populate_them_through_the_same_schema(self) -> None:
        """F11 defines neither field and validates against this same schema (its FR-011.57),
        so the mode's own values must pass here."""
        run_log.validate_block(block(quiesce_hold_ms=12.0, commit_duration_ms=48.5))


class TestTheRoundTrip:
    def test_what_is_written_is_what_comes_back(self, log_path: Path) -> None:
        payload = block()
        run_log.append(log_path, payload)
        assert list(run_log.read_blocks(log_path)) == [payload]

    def test_lines_come_back_in_written_order(self, log_path: Path) -> None:
        for step in (1, 2, 3):
            run_log.append(log_path, block(step=step))
        assert [record["step"] for record in run_log.read_blocks(log_path)] == [1, 2, 3]

    def test_blocks_can_be_filtered_by_discriminator(self, log_path: Path) -> None:
        """Five writers append **interleaved** to one log; a reader joining them by file
        would be reconstructing an ordering the single log gives for free."""
        run_log.append(log_path, block(step=1))
        run_log.append(
            log_path,
            {
                "block": "definition",
                "run_id": RUN,
                "step": None,
                "ts": "2026-10-14T00:00:00+00:00",
                "submitted": {"mode": "load"},
                "resolved": {"gpu_uuid": "GPU-1"},
                "definition_sha256": "b" * 64,
            },
        )
        run_log.append(log_path, block(step=2))
        assert [r["step"] for r in run_log.read_blocks(log_path, block="step")] == [1, 2]
        assert len(list(run_log.read_blocks(log_path))) == 3

    def test_keys_are_sorted_on_disk_because_one_serialiser_writes_them(
        self, log_path: Path
    ) -> None:
        """ADR-005: ``canonical_bytes`` is the only serialiser. It also forbids NaN, which
        matters here - ``json`` would otherwise emit the literal ``NaN``, which no parser is
        required to accept, so the log would validate going out and fail coming back."""
        run_log.append(log_path, block())
        raw = log_path.read_text().strip()
        assert raw == json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"))

    def test_an_absent_log_yields_nothing_rather_than_raising(self, tmp_path: Path) -> None:
        assert list(run_log.read_blocks(tmp_path / "nope" / "run.jsonl")) == []

    def test_a_corrupt_line_is_REFUSED_not_skipped(self, log_path: Path) -> None:
        """Skipping would make a truncated log read as a shorter run, and an index rebuilt
        from it would disagree with the record silently - the exact divergence ADR-004's
        equivalence test exists to detect."""
        run_log.append(log_path, block())
        with log_path.open("a") as handle:
            handle.write("{truncated\n")
        with pytest.raises(Refusal) as caught:
            list(run_log.read_blocks(log_path))
        assert caught.value.code == "run_log_corrupt"
        assert caught.value.details["line"] == 2

    def test_blank_lines_are_tolerated(self, log_path: Path) -> None:
        run_log.append(log_path, block())
        with log_path.open("a") as handle:
            handle.write("\n")
        assert len(list(run_log.read_blocks(log_path))) == 1

    def test_the_log_requires_a_named_run(self, tmp_path: Path) -> None:
        with pytest.raises(Refusal) as caught:
            run_log.run_log_path(tmp_path, "")
        assert caught.value.code == "run_log_run_id_required"


class TestTheIndexProjection:
    def test_one_projection_serves_the_live_writer_and_the_rebuild(self, log_path: Path) -> None:
        """Two projections would agree until one was changed, and ADR-016 #4's test compares
        **API responses** - so the disagreement would surface as a reindex that quietly
        returns different numbers."""
        payload = block()
        run_log.append(log_path, payload)
        assert run_log.step_rows(log_path) == [run_log.step_rows_from_block(payload)]

    def test_the_row_carries_the_columns_mf_steps_actually_has(self, log_path: Path) -> None:
        """**Widened 2026-10-14, and the test's own name is why it had to be.**

        This asserted a seven-key set and was red the moment the projection gained
        `objective`, `quiesce_hold_ms` and `commit_duration_ms` — which is the test doing its
        job: the key set is pinned so a column cannot quietly appear or vanish.

        But it was asserting the *narrow* set while `mf_steps` had carried all ten columns
        since migration `0011`, so for a day the test's name was false of it: three columns
        existed, the projection skipped them, the partial index
        `ix_mf_steps_commit_duration` was over a column nothing wrote, and **`objective` was
        null on every row while F9 draws loss curves from this table.** Found by F11 reading
        for its own timings work.

        The set is derived from the ORM table rather than restated, so the next column added
        to `mf_steps` makes this red until the projection carries it — which is the direction
        the failure should point. The three excluded names are the ones `mf_steps` holds for
        reasons other than a step record: its surrogate key and its own timestamp.
        """
        from models.run import Step

        run_log.append(log_path, block())
        row = run_log.step_rows(log_path)[0]

        projectable = {c.name for c in Step.__table__.columns} - {"id", "created_at"}
        assert set(row) == projectable, (
            "the projection and mf_steps' columns disagree. A column the projection skips is "
            "null on every row - and if it carries an index, the index is over nothing."
        )
        # Named explicitly as well, so a reader sees the shape without running it and so a
        # column REMOVED from mf_steps cannot make this pass by shrinking both sides.
        assert set(row) == {
            "run_id",
            "step",
            "loss",
            "grad_norm",
            "learning_rate",
            "wall_time_ms",
            "batch_source_hashes",
            "objective",
            "quiesce_hold_ms",
            "commit_duration_ms",
        }

    def test_wall_time_is_rounded_only_at_the_index_boundary(self, log_path: Path) -> None:
        """The authoritative value keeps its precision; the integer column takes the rounded
        one, and the rounding happens in one place."""
        run_log.append(log_path, block(wall_time_ms=123.7))
        assert run_log.step_rows(log_path)[0]["wall_time_ms"] == 124
        assert next(iter(run_log.read_blocks(log_path)))["wall_time_ms"] == 123.7

    def test_the_batch_hash_becomes_a_MAPPING_not_a_bare_string(self, log_path: Path) -> None:
        """The column is ``batch_source_hashes``, plural: F2's batch block carries selector
        and grader config hashes beside the content hash."""
        run_log.append(log_path, block())
        assert run_log.step_rows(log_path)[0]["batch_source_hashes"] == {"batch_sha256": "a" * 64}

    def test_a_step_with_no_batch_hash_indexes_None(self, log_path: Path) -> None:
        run_log.append(log_path, block(batch_sha256=None))
        assert run_log.step_rows(log_path)[0]["batch_source_hashes"] is None


class TestCommitDurationIsQueryableAsADistribution:
    """FR-001.26 - a cross-repo product obligation, not telemetry."""

    def test_a_percentile_over_seeded_values_comes_back(self, log_path: Path) -> None:
        for index, duration in enumerate([10.0, 20.0, 30.0, 40.0, 1000.0], start=1):
            run_log.append(log_path, block(step=index, commit_duration_ms=duration))
        assert run_log.commit_duration_percentile(log_path, 100.0) == 1000.0
        assert run_log.commit_duration_percentile(log_path, 50.0) == 30.0

    def test_a_last_value_column_could_not_serve_this(self, log_path: Path) -> None:
        """The reason the field has to be kept per step: the p99.9 of a distribution is not
        recoverable from the most recent value, and miLLM's watchdog is ten times it."""
        for index in range(1, 11):
            run_log.append(log_path, block(step=index, commit_duration_ms=float(index)))
        assert run_log.commit_duration_percentile(log_path, 90.0) == 9.0
        assert list(run_log.read_blocks(log_path))[-1]["commit_duration_ms"] == 10.0

    def test_nearest_rank_never_interpolates_a_duration_that_was_not_measured(
        self, log_path: Path
    ) -> None:
        """For a watchdog threshold an interpolated value is a commit that never happened,
        and the gate is ten times this number."""
        for index, duration in enumerate([5.0, 7.0], start=1):
            run_log.append(log_path, block(step=index, commit_duration_ms=duration))
        assert run_log.commit_duration_percentile(log_path, 50.0) in {5.0, 7.0}

    def test_the_standalone_case_returns_None_rather_than_zero(self, log_path: Path) -> None:
        """**The normal standalone answer.** The default path has no commit bracket to time,
        and zero would be a measured instantaneous commit - reporting one would hand miLLM a
        watchdog of nothing."""
        run_log.append(log_path, block())
        assert run_log.commit_duration_percentile(log_path, 99.9) is None

    @pytest.mark.parametrize("bad", [0.0, -1.0, 100.1])
    def test_a_percentile_out_of_range_refuses(self, log_path: Path, bad: float) -> None:
        with pytest.raises(Refusal) as caught:
            run_log.commit_duration_percentile(log_path, bad)
        assert caught.value.code == "percentile_out_of_range"


class TestDurability:
    def test_the_append_is_fsynced(self, log_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A step record sitting in the page cache when the machine loses power is a step
        that happened and cannot be proved - and this file is what other records are
        reconciled against."""
        import os

        synced: list[int] = []
        real = os.fsync
        monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
        run_log.append(log_path, block())
        assert synced, "the run log append must fsync"

    def test_appending_never_rewrites_an_earlier_line(self, log_path: Path) -> None:
        run_log.append(log_path, block(step=1))
        first = log_path.read_text()
        run_log.append(log_path, block(step=2))
        assert log_path.read_text().startswith(first)
