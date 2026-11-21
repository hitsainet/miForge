"""The ``parts`` element: one shape, settled by a gate contract  (FR-007.32, FAC-13).

F7's own documents specify the element as ``{slot, pack, source, type, name, version}``.
``batch-record-v1.json``'s ``$defs.PartReference`` - foundation task 3.2's gate contract -
specifies ``{kind, contract_reference, external}``, closed, with all three required. **The
documents' shape does not validate**, and these tests are what establish that rather than an
argument about it.

The other half of the correction is that **a second producer already exists**:
``training/signals/parts_reader.part_reference()`` emits this element for a batch record. F7
is the *pack-side* producer, and the two must agree key for key - which is asserted here,
against both functions' real output.

Mutation control (task 14.3f) for this file: drop ``kind`` from
:meth:`PartReferenceRecord.element`. The conformance assertions below and
``test_pack_parts_element_conformance.py`` must go red. **The control the FTASKS specified -
dropping ``slot`` and expecting the adapter schema to catch it - cannot fire**: that schema
constrains a ``parts`` item to ``{"type": "object"}`` and an object with a missing key is
still an object. Validating against the batch record's closed subschema is what makes the
control bite, and that is why both schemas are used.
"""

from __future__ import annotations

from datetime import UTC, datetime

import jsonschema
import pytest
from pydantic import ValidationError

from packs import contracts
from packs.records import PackVersionIdentity, part_reference_record
from packs.references import ContractPartRef, ExternalPartRef
from packs.rendering import DisplayRef
from packs.slots import SlotKind

PACK = PackVersionIdentity(pack_id="notary-md", version="2")
CONTRACT = ContractPartRef(type="prompt-set", name="support-tickets", version="3")
EXTERNAL = ExternalPartRef(
    origin="/data/operator/prompts.jsonl",
    content_sha256="d" * 64,
    taken_in_at=datetime(2026, 10, 14, tzinfo=UTC),
)


def validator() -> jsonschema.Draft202012Validator:
    """The **real** subschema, read out of the loaded file. Never a copied constraint."""
    return jsonschema.Draft202012Validator(dict(contracts.part_reference_subschema()))


class TestTheElementIsTheContractS:
    def test_a_contract_part_validates_as_a_part_reference(self) -> None:
        element = part_reference_record(SlotKind.PROMPT_SET, CONTRACT, pack=PACK).element()
        validator().validate(element)
        assert element == {
            "kind": "prompt_set",
            "contract_reference": {"type": "prompt-set", "name": "support-tickets", "version": "3"},
            "external": None,
        }

    def test_an_external_part_validates_as_a_part_reference(self) -> None:
        element = part_reference_record(SlotKind.CORPUS, EXTERNAL, pack=PACK).element()
        validator().validate(element)
        assert element["contract_reference"] is None
        assert element["external"] == {
            "source": "/data/operator/prompts.jsonl",
            "content_sha256": "d" * 64,
            "retrieved_at": "2026-10-14T00:00:00Z",
        }

    def test_both_keys_are_always_stated_and_exactly_one_is_null(self) -> None:
        """An absent key is an omission; a null one is an assertion. The contract requires both."""
        for record in (
            part_reference_record(SlotKind.PROMPT_SET, CONTRACT, pack=PACK),
            part_reference_record(SlotKind.CORPUS, EXTERNAL, pack=PACK),
        ):
            element = record.element()
            assert "contract_reference" in element
            assert "external" in element
            assert (element["contract_reference"] is None) != (element["external"] is None)

    def test_the_documents_shape_would_not_validate(self) -> None:
        """The finding, asserted rather than argued.

        ``{slot, pack, source, type, name, version}`` fails three ways: ``slot`` is not
        ``kind``, ``pack`` is an extra key on a closed object, and the contract reference's
        members belong inside ``contract_reference``.
        """
        documents_shape = {
            "slot": "prompt_set",
            "pack": {"pack_id": "notary-md", "version": "2"},
            "source": "midataworks",
            "type": "prompt-set",
            "name": "support-tickets",
            "version": "3",
        }
        with pytest.raises(jsonschema.ValidationError):
            validator().validate(documents_shape)

    def test_the_pack_identity_rides_on_the_record_and_not_in_the_element(self) -> None:
        """R-02.44's reason for the field is real; the artifact has nowhere to put it.

        The element is closed and ``adapter-manifest-v1.json`` has no top-level ``pack``
        field either, so an adapter cannot yet name the pack it trained on. The identity is
        kept **on the record** for lineage and display, and the schema question is the
        operator's cross-repo act (FPRD OQ-4).
        """
        record = part_reference_record(SlotKind.PROMPT_SET, CONTRACT, pack=PACK)
        assert record.pack == PACK
        assert "pack" not in record.element()


class TestOneProducer:
    def test_the_reference_types_offer_no_element_shortcut(self) -> None:
        """No ``to_dict()`` on a reference, no builder on a slot (FAC-13's second half)."""
        for model in (ContractPartRef, ExternalPartRef):
            for name in ("to_dict", "as_element", "as_part_reference", "element"):
                assert not hasattr(model, name), f"{model.__name__}.{name} is a second producer"

    def test_f2s_batch_record_producer_emits_the_same_keys(self) -> None:
        """The seam, asserted against F2's **real** function rather than against its docstring.

        F2 writes this element into a batch record and F7 into an adapter manifest. If the
        two ever disagree, one artifact's ``parts`` means something different from the
        other's - and nothing downstream walking both would know.
        """
        from training.signals.parts_reader import part_reference

        theirs = part_reference(
            {
                "id": "part-1",
                "contract_reference": {"type": "prompt-set", "name": "tickets", "version": "3"},
                "external_provenance": None,
            },
            kind="prompt_set",
        )
        ours = part_reference_record(SlotKind.PROMPT_SET, CONTRACT, pack=PACK).element()
        assert set(theirs) == set(ours)
        validator().validate(theirs)

    def test_no_display_form_can_reach_the_element(self) -> None:
        """FR-007.27.2: the display form never travels into another feature's artifact."""
        record = part_reference_record(SlotKind.PROMPT_SET, CONTRACT, pack=PACK)
        element = record.element()
        assert not any(isinstance(value, DisplayRef) for value in element.values())
        assert all(not isinstance(value, DisplayRef) for value in record.pack.model_dump().values())


class TestTheRecordIsARecord:
    def test_it_is_frozen(self) -> None:
        record = part_reference_record(SlotKind.PROMPT_SET, CONTRACT, pack=PACK)
        with pytest.raises(ValidationError):
            record.slot = SlotKind.TEST_SET  # type: ignore[misc]

    def test_extras_are_refused(self) -> None:
        from packs.records import PartReferenceRecord

        with pytest.raises(ValidationError):
            PartReferenceRecord(
                slot=SlotKind.PROMPT_SET, pack=PACK, ref=CONTRACT, source="midataworks"  # type: ignore[call-arg]
            )

    def test_the_source_discriminator_is_readable_from_the_record(self) -> None:
        assert part_reference_record(SlotKind.PROMPT_SET, CONTRACT, pack=PACK).source == (
            "midataworks"
        )
        assert part_reference_record(SlotKind.CORPUS, EXTERNAL, pack=PACK).source == "external"
