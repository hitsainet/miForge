"""The seven slots, checked against the files rather than against a copied list  (FR-007.3).

**No enum is copied into this file.** Every expectation is read out of a vendored schema,
because a copied list agrees with itself and would keep passing after the schema moved -
which is the construction-green failure this project keeps finding.

Three contracts have an opinion about the slot set and all three are read:

* ``pack-definition-v1.json``'s ``slots`` - the pack artifact's own statement;
* ``batch-record-v1.json``'s ``PartReference.kind`` - the element another artifact carries;
* ``midataworks-dataset-version-v1.json``'s ``Target`` - whose five ``miforge_set_kind``
  values plus two configuration ``kind`` values are the seven.

:mod:`packs.slots` runs these checks **at import** as well. Both are wanted: the import check
cannot be skipped, and this file is what says *which* disagreement a reader is looking at
when it fires.
"""

from __future__ import annotations

import pytest

from packs import contracts
from packs.slots import (
    FLOOR_BEARING_SLOTS,
    REQUIRED_SLOT_ORDER,
    REQUIRED_SLOTS,
    SLOT_ORDER,
    SLOT_TO_CONTRACT,
    SlotKind,
    storage_part_kind,
)


class TestTheSlotSetIsTheContractS:
    def test_the_pack_definition_declares_exactly_these_seven(self) -> None:
        declared = contracts.at(
            contracts.PACK_DEFINITION_SCHEMA,
            "$defs",
            "PackDefinitionV1",
            "properties",
            "slots",
        )
        assert set(declared["properties"]) == {slot.value for slot in SlotKind}

    def test_every_slot_is_stated_even_when_null(self) -> None:
        """The schema **requires all seven keys**, which is the point.

        *"No retention set" must be an assertion rather than a silence* - the same discipline
        as the nullable-but-required ``domain``. An absent key is an omission and a null
        value is a fact, and collapsing the two is the defect that decision exists to prevent.
        """
        declared = contracts.at(
            contracts.PACK_DEFINITION_SCHEMA,
            "$defs",
            "PackDefinitionV1",
            "properties",
            "slots",
        )
        assert set(declared["required"]) == {slot.value for slot in SlotKind}
        assert declared["additionalProperties"] is False

    def test_the_element_shape_can_name_every_slot(self) -> None:
        """An artifact's ``parts`` element carries ``kind``, and it must cover all seven.

        A slot the element cannot name is a part whose role nothing downstream could record -
        which is exactly what the ``slot``/``kind`` disagreement between F7's documents and
        the gate contract would have produced.
        """
        kinds = contracts.enum_at(
            contracts.BATCH_RECORD_SCHEMA, "$defs", "PartReference", "properties", "kind"
        )
        assert set(kinds) == {slot.value for slot in SlotKind}


class TestTheMappingOntoMiDataworks:
    def test_five_set_kinds_plus_two_configuration_kinds_are_the_seven(self) -> None:
        set_kinds = contracts.enum_at(
            contracts.MIDATAWORKS_SCHEMA, "$defs", "Target", "properties", "miforge_set_kind"
        )
        target_kinds = contracts.enum_at(
            contracts.MIDATAWORKS_SCHEMA, "$defs", "Target", "properties", "kind"
        )
        mapped_sets = {
            source.set_kind for source in SLOT_TO_CONTRACT.values() if source.set_kind is not None
        }
        configs = {source.kind for source in SLOT_TO_CONTRACT.values() if source.set_kind is None}
        assert mapped_sets == set(set_kinds)
        assert configs <= set(target_kinds)
        assert len(set_kinds) + len(configs) == len(SlotKind)

    def test_every_slot_has_exactly_one_contract_source(self) -> None:
        assert set(SLOT_TO_CONTRACT) == set(SlotKind)

    @pytest.mark.parametrize("slot", list(SlotKind))
    def test_every_slot_has_a_storage_spelling(self, slot: SlotKind) -> None:
        """``mf_pack_parts`` spells its kinds with hyphens - a **translation**, not an alias.

        That table is another owner's file and its vocabulary is a fourth spelling
        (``prompt-set``, ``selector``, ``grader``). The translation happens in one place so
        the difference is visible at the boundary that crosses it rather than inferred at
        three call sites.
        """
        stored = storage_part_kind(slot)
        assert stored
        assert "_" not in stored

    def test_the_storage_spelling_is_accepted_by_the_table(self) -> None:
        from models.pack import PART_KINDS

        assert {storage_part_kind(slot) for slot in SlotKind} <= PART_KINDS


class TestTheRequiredSetIsRTwoThirtyEightS:
    def test_exactly_three_slots_are_required(self) -> None:
        """R-02.38's sentence, pinned: *a prompt set … a test set, and a retention set*."""
        assert REQUIRED_SLOTS == {
            SlotKind.PROMPT_SET,
            SlotKind.TEST_SET,
            SlotKind.RETENTION_SET,
        }
        assert len(REQUIRED_SLOTS) == 3

    def test_the_required_order_is_stable(self) -> None:
        """A refusal lists missing slots in one order, so two runs read the same way."""
        assert REQUIRED_SLOT_ORDER == (
            SlotKind.PROMPT_SET,
            SlotKind.TEST_SET,
            SlotKind.RETENTION_SET,
        )

    def test_the_selector_and_grader_slots_are_optional(self) -> None:
        """FPRD OQ-1, resolved in favour of R-02.38 and recorded rather than silent.

        R-02.35's prose reads *"a selector and/or grader configuration"*, which sounds like
        at least one is needed; R-02.38 governs validity and requires neither - and that is
        the only reading under which a plain corpus run (R-02.4, no selection at all) can
        start from a pack.
        """
        assert SlotKind.SELECTOR_CONFIG not in REQUIRED_SLOTS
        assert SlotKind.GRADER_CONFIG not in REQUIRED_SLOTS

    def test_only_the_retention_slot_bears_a_floor(self) -> None:
        """The floor is **miForge's own value** (FR-007.5), verified absent from their contract."""
        assert FLOOR_BEARING_SLOTS == {SlotKind.RETENTION_SET}

    def test_the_floor_is_not_in_midataworks_contract(self) -> None:
        import json
        from pathlib import Path

        text = Path(contracts.SCHEMA_DIR / contracts.MIDATAWORKS_SCHEMA).read_text("utf-8")
        document = json.loads(text)
        assert "floor" not in json.dumps(document.get("$defs", {}).get("Target", {}))


def test_the_slot_order_covers_every_slot_once() -> None:
    assert set(SLOT_ORDER) == set(SlotKind)
    assert len(SLOT_ORDER) == len(SlotKind)
