"""A writer-produced pack document against its own schema  (FR-007.7, E9, E10, E14).

**The document under test is produced by the real writer**, never typed out here. A
hand-written document would agree with a buggy reader because both would encode the same
misreading - *fixtures whose fields agree by construction are how a suite stays green over a
real bug.*

``pack-definition-v1.json`` is foundation task 3.4's and is **not** defined by this feature:
these tests assert conformance to it, and the one place a mismatch is allowed to surface is a
refusal naming the schema.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from core.canonical import canonical_bytes, sha256_of
from packs import contracts
from packs.definition import (
    DIGEST_FIELD,
    SCHEMA_VERSION,
    PackVersionDefinition,
    require_floor,
    run_input_projection,
    validate_document,
)
from packs.errors import PACK_DOCUMENT_INVALID, RETENTION_FLOOR_MISSING, PackRefusal
from packs.slots import SlotKind
from tests.support import pack_fixtures as fx


def document() -> dict[str, object]:
    """A document the real writer produced, for every test below."""
    return fx.definition(slots=fx.valid_slots()).document()


class TestTheWriterProducesAConformingDocument:
    def test_it_validates_against_the_vendored_schema(self) -> None:
        # Through `contracts.validator`, which resolves the schema's **cross-file** `$ref`
        # into `batch-record-v1.json` from disk. A plain `Draft202012Validator` would try to
        # fetch that reference over the network and raise `Unresolvable` - this schema is the
        # first vendored contract in the repository to reference another file.
        contracts.validator(contracts.PACK_DEFINITION_SCHEMA).validate(document())

    def test_all_seven_slots_are_present_even_when_empty(self) -> None:
        """Every slot stated, four of them null - an assertion rather than a silence."""
        slots = document()["slots"]
        assert isinstance(slots, dict)
        assert set(slots) == {slot.value for slot in SlotKind}
        assert slots["corpus"] is None
        assert slots["preference_pairs"] is None

    def test_the_domain_key_is_present_when_the_pack_is_not_domain_scoped(self) -> None:
        """Nullable **but required**: null is an assertion, an absent key is an omission.

        This is the distinction the nullable-``domain`` decision exists to protect, and it is
        the one several documents in this project collapsed.
        """
        general = fx.definition(domain=None, slots=fx.valid_slots()).document()
        assert "domain" in general
        assert general["domain"] is None
        validate_document(general)

    def test_the_digest_covers_the_document_minus_itself(self) -> None:
        """One canonical function for hashing **and** writing (ADR-005).

        The bytes an export serves are the bytes the digest was taken over, so an export is
        verifiable by hashing what arrived. Two serialisers would be two answers.
        """
        doc = document()
        body = {key: value for key, value in doc.items() if key != DIGEST_FIELD}
        assert doc[DIGEST_FIELD] == sha256_of(body)
        assert canonical_bytes(doc) == json.dumps(
            doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")


class TestTheReaderRefusesWhatItCannotMean:
    def test_a_round_trip_through_the_real_reader_preserves_the_run_inputs(self) -> None:
        original = fx.definition(slots=fx.valid_slots())
        read_back = PackVersionDefinition.from_document(original.document())
        assert run_input_projection(read_back) == run_input_projection(original)

    def test_a_tampered_document_is_refused_by_its_own_digest(self) -> None:
        doc = document()
        doc["domain"] = "something else"
        with pytest.raises(PackRefusal) as raised:
            PackVersionDefinition.from_document(doc)
        assert raised.value.code == PACK_DOCUMENT_INVALID
        assert "definition_sha256" in raised.value.message

    def test_a_future_schema_version_is_refused_naming_both_versions(self) -> None:
        """E9. Refused **before** schema validation, so the message names the version.

        The schema's ``schema_version`` is a ``const``, so a future document fails anyway -
        as *"'2' was expected to be '1'"* at a JSON pointer, which tells an operator nothing
        about what to do.
        """
        doc = document()
        doc["schema_version"] = "2"
        with pytest.raises(PackRefusal) as raised:
            PackVersionDefinition.from_document(doc)
        assert raised.value.details["declared_version"] == "2"
        assert raised.value.details["supported_version"] == SCHEMA_VERSION

    def test_a_malformed_document_names_every_failing_pointer(self) -> None:
        with pytest.raises(PackRefusal) as raised:
            validate_document({"schema_version": "1", "name": "NOT A SLUG"})
        assert raised.value.details["schema"] == contracts.PACK_DEFINITION_SCHEMA
        assert len(raised.value.details["errors"]) >= 2

    def test_a_slot_with_neither_source_is_refused(self) -> None:
        """An unattributed input makes the run unreproducible: valid shape, still refused."""
        doc = document()
        doc["slots"]["corpus"] = {"contract_reference": None, "external": None, "floor": None}  # type: ignore[index]
        doc[DIGEST_FIELD] = sha256_of({k: v for k, v in doc.items() if k != DIGEST_FIELD})
        with pytest.raises(PackRefusal) as raised:
            PackVersionDefinition.from_document(doc)
        assert raised.value.details["slot"] == "corpus"

    def test_a_slot_with_both_sources_is_refused(self) -> None:
        """Two answers about where one part came from."""
        doc = document()
        doc["slots"]["corpus"] = {  # type: ignore[index]
            "contract_reference": {"type": "corpus", "name": "c", "version": "1"},
            "external": {
                "source": "/x",
                "content_sha256": "f" * 64,
                "retrieved_at": "2026-10-14T00:00:00Z",
            },
            "floor": None,
        }
        doc[DIGEST_FIELD] = sha256_of({k: v for k, v in doc.items() if k != DIGEST_FIELD})
        with pytest.raises(PackRefusal):
            PackVersionDefinition.from_document(doc)

    def test_a_floor_on_a_slot_that_bears_none_is_refused_rather_than_ignored(self) -> None:
        """Conformant to the schema, meaningless in a composition.

        A number that is ignored is a number some later reader will act on, so the floor is
        refused anywhere but the retention slot (R-02.38).
        """
        doc = document()
        doc["slots"]["prompt_set"]["floor"] = 0.5  # type: ignore[index]
        doc[DIGEST_FIELD] = sha256_of({k: v for k, v in doc.items() if k != DIGEST_FIELD})
        with pytest.raises(PackRefusal) as raised:
            PackVersionDefinition.from_document(doc)
        assert raised.value.details["slot"] == "prompt_set"


class TestTheFloorIsPartOfTheRetentionSlot:
    def test_a_filled_retention_slot_without_a_floor_is_its_own_refusal(self) -> None:
        """E14, and **not** reported as a missing retention set: the fixes differ."""
        pack = fx.definition(slots=fx.valid_slots(retention_set=fx.retention_slot(floor=None)))
        with pytest.raises(PackRefusal) as raised:
            require_floor(pack)
        assert raised.value.code == RETENTION_FLOOR_MISSING
        assert raised.value.details["slot"] == SlotKind.RETENTION_SET.value

    def test_a_floor_outside_zero_to_one_is_refused(self) -> None:
        """The same 0..1 range both manifests' ``ScoreRecord.floor`` uses."""
        with pytest.raises(ValidationError):
            fx.definition(slots=fx.valid_slots(retention_set=fx.retention_slot(floor=1.5)))

    def test_an_empty_retention_slot_does_not_demand_a_floor(self) -> None:
        """No set, no threshold: the missing *slot* is what validity reports."""
        slots = fx.valid_slots()
        del slots[SlotKind.RETENTION_SET.value]
        require_floor(fx.definition(slots=slots))


def test_the_created_at_timestamp_is_utc_with_a_z() -> None:
    """Every timestamp miForge writes reads the same way."""
    doc = fx.definition(
        slots=fx.valid_slots(), created_at=datetime(2026, 10, 14, 12, 30, tzinfo=UTC)
    ).document()
    assert doc["created_at"] == "2026-10-14T12:30:00Z"
