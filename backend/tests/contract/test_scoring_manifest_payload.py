"""The ScoreRecord key set, **in both directions**  (FR-004.29; AC-004.6, AC-004.8; M-4).

This is the test ``jsonschema`` cannot be.

Since the 2026-10-13 re-vendor every object in ``checkpoint-manifest-v1.json`` sets
``additionalProperties: false``, so an **extra** key is refused by the validator itself.
What a validator still cannot see is a **declared, non-required** key being **omitted** -
and ``provider`` is exactly that: declared ``anyOf [string, null]``, defaulted ``null``, and
**not in ``required``**. So a payload of the old four keys validates clean while leaving
R-02.16's obligation unmet in the one artifact that travels without its run log:

    "together with which provider produced it - a score is only comparable against another
    taken the same way."

The schema-sync test cannot see it either, because that test compares schema **files**, not
manifests. **A test is the only thing that catches the omission**, and this is it.

``declared`` is read from the schema file at test time rather than retyped, which is the only
reason the builder survived the schema changing underneath it - and why the next change moves
this test rather than breaking it.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from core.errors import Refusal
from services.scoring import manifest_payload

SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "docs" / "schemas" / "checkpoint-manifest-v1.json"
)


def declared_from_file() -> set[str]:
    """The declared property set, read **independently** of the module under test.

    Independently on purpose: if both the builder and the assertion read it through the same
    helper, a bug in that helper would make them agree with each other and with nothing else.
    """
    return set(json.loads(SCHEMA_PATH.read_text("utf-8"))["$defs"]["ScoreRecord"]["properties"])


def a_score() -> dict[str, object]:
    return {
        "set": "test_set/general@1",
        "score": 0.84,
        "n": 200,
        "floor": None,
        "provider": "in_process:lfm2-1.2b",
    }


def validator() -> jsonschema.Draft202012Validator:
    document = json.loads(SCHEMA_PATH.read_text("utf-8"))
    return jsonschema.Draft202012Validator(
        {**document["$defs"]["ScoreRecord"], "$defs": document["$defs"]}
    )


class TestTheKeySetIsExactlyTheSchemas:
    def test_both_directions(self) -> None:
        """No key the schema does not declare, and **no declared key missing**."""
        payload = manifest_payload.score_record_payload(a_score())
        assert payload is not None
        assert set(payload) == declared_from_file()

    def test_the_declared_set_is_what_the_module_reads(self) -> None:
        assert manifest_payload.declared_keys() == declared_from_file()

    def test_provider_is_declared(self) -> None:
        assert "provider" in declared_from_file()

    def test_provider_is_NOT_required_which_is_the_whole_point(self) -> None:
        """The distinction this test rests on, asserted so it cannot change silently."""
        assert "provider" not in manifest_payload.required_keys()
        assert manifest_payload.required_keys() == {"set", "score", "n"}

    def test_the_payload_validates(self) -> None:
        validator().validate(manifest_payload.score_record_payload(a_score()))


class TestM4_THE_OMISSION_A_VALIDATOR_CANNOT_SEE:
    """Mutation control M-4, re-pointed from *adding* the key to *omitting* it.

    The omission is the mutation a well-meaning implementer actually produces - by copying
    the old four-key builder - and it is the one the contract cannot catch.
    """

    def test_a_four_key_payload_VALIDATES_CLEAN(self) -> None:
        """**The premise.** If this ever fails, the schema has changed and M-4 moves."""
        four_key = {k: v for k, v in a_score().items() if k != "provider"}
        validator().validate(four_key)  # no error - that is the finding, not a bug

    def test_but_the_key_set_assertion_catches_it(self) -> None:
        """The missing direction. This is the assertion M-4 breaks."""
        four_key = {k: v for k, v in a_score().items() if k != "provider"}
        assert set(four_key) != declared_from_file()
        assert declared_from_file() - set(four_key) == {"provider"}

    def test_the_builder_refuses_to_produce_one(self) -> None:
        """Belt and braces: the builder cannot emit a short payload even by accident.

        Refused rather than defaulted to ``null``, because a defaulted ``provider`` is M-4 by
        another route - it validates, and R-02.16's obligation is unmet in the artifact.
        """
        four_key = {k: v for k, v in a_score().items() if k != "provider"}
        with pytest.raises(Refusal) as caught:
            manifest_payload.score_record_payload(four_key)
        assert caught.value.code == "manifest_score_payload_invalid"
        assert caught.value.details["key"] == "provider"

    def test_an_undeclared_key_is_caught_by_the_validator_now(self) -> None:
        """The *other* direction, which the schema took over on 2026-10-13.

        Kept as an assertion rather than deleted, because the key-set test covers both and a
        later schema that re-opened its objects would make this the only guard again.
        """
        with pytest.raises(jsonschema.ValidationError):
            validator().validate({**a_score(), "tolerance": 0.01})


class TestANonEvaluationSaveWritesNullPresent:
    """E-22: ``null`` is an assertion that nothing ran; an absent key is an omission."""

    def test_none_becomes_null(self) -> None:
        assert manifest_payload.score_record_payload(None) is None

    def test_both_keys_are_always_present(self) -> None:
        scores = manifest_payload.manifest_scores(test=None, retention=None)
        assert set(scores) == {"test", "retention"}
        assert scores == {"test": None, "retention": None}

    def test_retention_is_written_as_well_as_test(self) -> None:
        """FR-004.31, on the schema's own writer annotation.

        R-02.18's text names only ``test``; the schema has a ``retention`` property annotated
        *"Written by the trainer"*, and criteria 3 and 19 both need a durable retention
        series. **The requirement's text is narrower than its own normative schema and that
        is reported (DEF-9), not amended here.**
        """
        scores = manifest_payload.manifest_scores(
            test=a_score(), retention={**a_score(), "floor": 0.72}
        )
        assert scores["retention"] is not None
        assert scores["retention"]["floor"] == 0.72
        assert set(scores["retention"]) == declared_from_file()

    def test_the_schema_annotates_retention_as_the_trainers(self) -> None:
        """Checked against the file, because this is the whole basis for writing it."""
        document = json.loads(SCHEMA_PATH.read_text("utf-8"))
        # The document is a `$ref` to its own `$defs`, so the properties live there - read
        # rather than assumed, which is the whole habit this file is about.
        root = document["$defs"]["CheckpointManifestV1"]
        description = json.dumps(root["properties"]["retention"])
        assert "Written by the trainer" in description
        assert "R-02.17" in description


class TestItTravelsThroughF5sRealBuilder:
    """*A hand-built dict agrees with the test's expectation by construction.*

    ADR-016's rule, in the one place it matters most: a hand-built fixture omitting
    ``provider`` would agree with a builder that omitted it, and the suite would stay green
    over exactly the defect this file exists to catch. So the round trip goes through F5's
    real manifest builder and back off the disk.
    """

    def test_a_round_tripped_manifest_keeps_the_provider(self, tmp_path: Path) -> None:
        from tests.support.scores import written_and_read_manifest

        scores = manifest_payload.manifest_scores(
            test=a_score(), retention={**a_score(), "floor": 0.72}
        )
        manifest = written_and_read_manifest(
            tmp_path, test=scores["test"], retention=scores["retention"]
        )
        assert manifest["test"]["provider"] == "in_process:lfm2-1.2b"
        assert set(manifest["test"]) == declared_from_file()
        assert set(manifest["retention"]) == declared_from_file()

    def test_a_non_evaluation_save_round_trips_both_nulls(self, tmp_path: Path) -> None:
        from tests.support.scores import written_and_read_manifest

        manifest = written_and_read_manifest(tmp_path, test=None, retention=None)
        assert "test" in manifest and manifest["test"] is None
        assert "retention" in manifest and manifest["retention"] is None


class TestF4ChangesNoSchemaFile:
    def test_no_activation_sae_or_probe_field_exists_in_either_manifest(self) -> None:
        """M-10, as a standing assertion rather than only as a mutation.

        R-02.19's statistics have **no manifest field** in either schema - verified by
        searching both files - and inventing one would be a unilateral widening of a contract
        vendored byte-identical into another repository.
        """
        for name in ("checkpoint-manifest-v1.json", "adapter-manifest-v1.json"):
            text = (SCHEMA_PATH.parent / name).read_text("utf-8").lower()
            for forbidden in ("activation", "sae", "probe"):
                assert forbidden not in text, f"{name} mentions {forbidden!r}"

    def test_the_payload_builder_cannot_emit_an_activation_statistic(self) -> None:
        """M-10's direct form: an extra key never reaches a manifest, because the builder
        only ever emits the schema's declared set."""
        payload = manifest_payload.score_record_payload({**a_score(), "activation_mean": 0.3})
        assert payload is not None
        assert "activation_mean" not in payload
        assert set(payload) == declared_from_file()
