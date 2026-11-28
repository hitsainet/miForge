"""The seven slots  (task 2.3; FR-007.3, R-02.38).

**A pack is not a bag of parts.** One slot per part kind, and the slot is what gives a part
its role in a run: without it the composition cannot say whether a set was the test set or
the retention set, which is the one thing a travelling artifact most needs to know.

## Three vocabularies touch this module and they are NOT aliases

================================  ==================================================
Vocabulary                        Where it is owned
================================  ==================================================
``miforge_set_kind``              **miDataworks'**, in
                                  ``midataworks-dataset-version-v1.json`` - five values,
                                  underscored.
miForge **part/slot kinds**       miForge's, underscored:
                                  ``prompt_set``, ``corpus``, ``preference_pairs``,
                                  ``test_set``, ``retention_set``, ``selector_config``,
                                  ``grader_config``. Settled by **two** gate contracts -
                                  ``pack-definition-v1.json``'s ``slots`` and
                                  ``batch-record-v1.json``'s ``PartReference.kind`` - and
                                  this module maps **onto** them rather than declaring a
                                  vocabulary of its own.
**signal sources**                F2's, hyphenated: ``corpus``, ``self-generation``,
                                  ``preference-pairs``, ``reward``. Identical to the
                                  checkpoint manifest's ``signal_source``.
================================  ==================================================

A rename treating them as one broke 31 tests in this repository once;
``backend/tests/support/signal_fixtures.py`` carries the note. There is a **fourth**
spelling still in the tree - ``models.pack.PART_KINDS`` uses hyphens and
``selector``/``grader`` - and it is reported rather than silently followed, because that
module is single-owner shared state this feature does not edit. :func:`storage_part_kind`
is the one place the translation happens.

## The exhaustiveness checks run at IMPORT, not in a test

A test can be skipped and a feature flag cannot turn a startup failure into a silent
miscategorisation. So when a schema adds a slot kind or a set kind, this module fails when
it loads - loudly - rather than letting the new value fall through a default.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from core.errors import refuse
from packs import contracts


class SlotKind(StrEnum):
    """The seven slots of R-02.38. Checked against both gate contracts at import."""

    PROMPT_SET = "prompt_set"
    CORPUS = "corpus"
    PREFERENCE_PAIRS = "preference_pairs"
    SELECTOR_CONFIG = "selector_config"
    GRADER_CONFIG = "grader_config"
    TEST_SET = "test_set"
    RETENTION_SET = "retention_set"


#: The three R-02.38 requires for validity: *"a prompt set ... a test set, and a retention
#: set with its floor."* Exactly three, pinned by a test, because AC10's whole content is
#: that an incomplete pack **names the slot** - and it cannot name a slot the required set
#: does not contain.
#:
#: Ordered rather than a bare set at the reporting boundary: the refusal lists missing slots
#: in this order so two runs of the same validation read the same way.
REQUIRED_SLOT_ORDER: tuple[SlotKind, ...] = (
    SlotKind.PROMPT_SET,
    SlotKind.TEST_SET,
    SlotKind.RETENTION_SET,
)
REQUIRED_SLOTS: frozenset[SlotKind] = frozenset(REQUIRED_SLOT_ORDER)

#: The order the authoring flow and the pack document present the slots in (R-02.39's seven
#: steps). Stated once so a document, a lineage view and a validity report agree.
SLOT_ORDER: tuple[SlotKind, ...] = (
    SlotKind.PROMPT_SET,
    SlotKind.CORPUS,
    SlotKind.PREFERENCE_PAIRS,
    SlotKind.SELECTOR_CONFIG,
    SlotKind.GRADER_CONFIG,
    SlotKind.TEST_SET,
    SlotKind.RETENTION_SET,
)

#: Only the retention slot carries a floor, and the floor is **miForge's own value**:
#: miDataworks owns the set, this project owns what counts as an unacceptable regression on
#: it. Verified absent from their contract (FR-007.5), so there is nowhere upstream for it
#: to live.
FLOOR_BEARING_SLOTS: frozenset[SlotKind] = frozenset({SlotKind.RETENTION_SET})


@dataclass(frozen=True, slots=True)
class ContractSlotSource:
    """How a slot is addressed in miDataworks' contract.

    ``set_kind`` is ``None`` for the two configuration slots, which are their own
    ``target.kind`` values rather than a ``miforge_set``. That asymmetry is the contract's,
    not a special case here: BRD-03 R-03.48's sixth bullet is "selector and grader
    configurations", two kinds in one bullet, and the schema puts them on the outer axis.
    """

    kind: str
    set_kind: str | None


#: slot -> the contract target that fills it. The ``miforge_set`` mappings are the identity
#: today and are written out anyway: the two vocabularies are owned by different
#: repositories, and an identity mapping that is *stated* survives one of them renaming a
#: value, while an implicit one becomes a ``KeyError`` in production.
SLOT_TO_CONTRACT: dict[SlotKind, ContractSlotSource] = {
    SlotKind.PROMPT_SET: ContractSlotSource(kind="miforge_set", set_kind="prompt_set"),
    SlotKind.CORPUS: ContractSlotSource(kind="miforge_set", set_kind="corpus"),
    SlotKind.PREFERENCE_PAIRS: ContractSlotSource(kind="miforge_set", set_kind="preference_pairs"),
    SlotKind.TEST_SET: ContractSlotSource(kind="miforge_set", set_kind="test_set"),
    SlotKind.RETENTION_SET: ContractSlotSource(kind="miforge_set", set_kind="retention_set"),
    SlotKind.SELECTOR_CONFIG: ContractSlotSource(kind="selector_config", set_kind=None),
    SlotKind.GRADER_CONFIG: ContractSlotSource(kind="grader_config", set_kind=None),
}

#: miForge's slot kind -> the spelling ``mf_pack_parts.part_kind`` stores. **A translation,
#: not an alias**, and it exists because that table is another owner's file: its
#: ``PART_KINDS`` are hyphenated and call the configuration slots ``selector`` and
#: ``grader``. Written out so the difference is visible at the one boundary that crosses it;
#: reported to the coordinator as a vocabulary the index should eventually align with the
#: two gate contracts.
SLOT_TO_STORAGE_KIND: dict[SlotKind, str] = {
    SlotKind.PROMPT_SET: "prompt-set",
    SlotKind.CORPUS: "corpus",
    SlotKind.PREFERENCE_PAIRS: "preference-pairs",
    SlotKind.SELECTOR_CONFIG: "selector",
    SlotKind.GRADER_CONFIG: "grader",
    SlotKind.TEST_SET: "test-set",
    SlotKind.RETENTION_SET: "retention-set",
}


def storage_part_kind(slot: SlotKind) -> str:
    """The ``mf_pack_parts.part_kind`` spelling for ``slot``. The one translation site."""
    return SLOT_TO_STORAGE_KIND[slot]


def _assert_agrees_with_the_contracts() -> None:
    """Every slot is in both gate contracts and maps onto miDataworks'. **At import.**

    Raises:
        Refusal: ``pack_slots_disagree_with_contract`` naming the difference. Four checks,
            each of which has a way of being wrong that a reader would not notice:

            1. ``pack-definition-v1.json``'s ``slots`` properties - the pack artifact's own
               statement of the set;
            2. its ``slots.required`` list - **every slot is stated even when null**,
               because "no retention set" must be an assertion rather than a silence;
            3. ``batch-record-v1.json``'s ``PartReference.kind`` enum - the element shape
               another artifact carries, which has to be able to name the slot;
            4. miDataworks' ``miforge_set_kind`` plus their two configuration target kinds,
               which is what :data:`SLOT_TO_CONTRACT` claims to cover.
    """
    ours = {kind.value for kind in SlotKind}

    definition_slots = contracts.at(
        contracts.PACK_DEFINITION_SCHEMA,
        "$defs",
        "PackDefinitionV1",
        "properties",
        "slots",
    )
    declared = set(definition_slots.get("properties", {}))
    required = set(definition_slots.get("required", []))
    element_kinds = contracts.enum_at(
        contracts.BATCH_RECORD_SCHEMA, "$defs", "PartReference", "properties", "kind"
    )

    for label, theirs in (
        ("pack-definition-v1.json slots.properties", declared),
        ("pack-definition-v1.json slots.required", required),
        ("batch-record-v1.json PartReference.kind", set(element_kinds)),
    ):
        if theirs != ours:
            raise refuse(
                "pack_slots_disagree_with_contract",
                f"{label} is {sorted(theirs)} and packs/slots.py has {sorted(ours)}. The "
                f"contract is the authority; this module maps onto it and declares no "
                f"vocabulary of its own. Failing at import is deliberate - the "
                f"alternative is a slot nothing downstream can name.",
                contract=label,
                contract_values=sorted(theirs),
                ours=sorted(ours),
            )

    set_kinds = contracts.enum_at(
        contracts.MIDATAWORKS_SCHEMA, "$defs", "Target", "properties", "miforge_set_kind"
    )
    target_kinds = contracts.enum_at(
        contracts.MIDATAWORKS_SCHEMA, "$defs", "Target", "properties", "kind"
    )
    mapped_sets = {
        source.set_kind for source in SLOT_TO_CONTRACT.values() if source.set_kind is not None
    }
    if mapped_sets != set(set_kinds):
        raise refuse(
            "pack_slots_disagree_with_contract",
            f"miDataworks defines miforge_set_kind {sorted(set_kinds)} and this map covers "
            f"{sorted(mapped_sets)}. A set kind with no slot is a part a pack cannot hold; "
            f"a slot naming a set kind they do not publish is miForge inventing one.",
            contract="midataworks-dataset-version-v1.json Target.miforge_set_kind",
            contract_values=sorted(set_kinds),
            ours=sorted(mapped_sets),
        )

    config_kinds = {
        source.kind for source in SLOT_TO_CONTRACT.values() if source.set_kind is None
    } | {"miforge_set"}
    unknown = config_kinds - set(target_kinds)
    if unknown:
        raise refuse(
            "pack_slots_disagree_with_contract",
            f"This map addresses target kind(s) {sorted(unknown)} that "
            f"midataworks-dataset-version-v1.json does not define.",
            contract="midataworks-dataset-version-v1.json Target.kind",
            ours=sorted(unknown),
        )

    unmapped = ours - {slot.value for slot in SLOT_TO_CONTRACT} | ours - {
        slot.value for slot in SLOT_TO_STORAGE_KIND
    }
    if unmapped:  # pragma: no cover - dict literals above; kept as a guard, not a claim
        raise refuse(
            "pack_slots_disagree_with_contract",
            f"Slot(s) {sorted(unmapped)} have no contract source or no storage spelling, "
            f"so a pack carrying one could be neither resolved nor indexed.",
            ours=sorted(unmapped),
        )


_assert_agrees_with_the_contracts()
