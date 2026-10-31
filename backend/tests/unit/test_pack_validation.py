"""Validity names every missing slot, and the three failures stay apart  (FR-007.8, FR-007.9).

**AC10's whole content is the wording.** *"A pack missing a required slot says which slot,
never 'validation failed'"* - and FAC-12 makes it plural: three missing slots name all three.
So the tests below assert the **report**, not merely that a pack was rejected.

:func:`packs.validation.validate` is pure, so every combination here is a unit test with no
stub at all: it takes a definition and a resolution map and returns a verdict.

Mutation control (task 14.3e) for this file: make ``validate`` return only the first missing
slot. ``test_three_missing_slots_are_all_named`` must go red.
"""

from __future__ import annotations

import pytest

from packs.errors import (
    MIDATAWORKS_NOT_CONFIGURED,
    PACK_INCOMPLETE,
    PACK_PART_UNRESOLVED,
    RETENTION_FLOOR_MISSING,
    PackRefusal,
)
from packs.references import ContractPartRef
from packs.resolver import Resolution
from packs.slots import REQUIRED_SLOT_ORDER, SlotKind
from packs.validation import validate
from tests.support import pack_fixtures as fx


def resolved(
    slot: SlotKind, ref: ContractPartRef | None = None, rows: int | None = 42
) -> Resolution:
    reference = ref or ContractPartRef(type=slot.value, name="x", version="1")
    return Resolution(ref=reference, state="resolved", rendered="x/x@1", rows=rows)


def all_resolved(definition: object) -> dict[SlotKind, Resolution]:
    return {
        slot: resolved(slot)
        for slot in SlotKind
        if definition.slots.get(slot) is not None  # type: ignore[attr-defined]
    }


class TestAValidPack:
    def test_three_required_slots_filled_and_resolved_is_valid(self) -> None:
        pack = fx.definition(slots=fx.valid_slots())
        verdict = validate(pack, all_resolved(pack))
        assert verdict.state == "valid"
        assert verdict.is_valid
        assert verdict.missing_slots == ()
        assert verdict.payload()["state"] == "valid"

    def test_an_empty_optional_slot_does_not_make_a_pack_invalid(self) -> None:
        """A pack saying it has no corpus is a fact, not a gap (R-02.38)."""
        pack = fx.definition(slots=fx.valid_slots())
        assert pack.slots[SlotKind.CORPUS] is None
        assert validate(pack, all_resolved(pack)).is_valid

    def test_raise_if_invalid_is_silent_for_a_valid_pack(self) -> None:
        pack = fx.definition(slots=fx.valid_slots())
        validate(pack, all_resolved(pack)).raise_if_invalid()


class TestIncompleteNamesTheSlot:
    def test_one_missing_slot_is_named(self) -> None:
        slots = fx.valid_slots()
        del slots[SlotKind.TEST_SET.value]
        pack = fx.definition(slots=slots)
        verdict = validate(pack, all_resolved(pack))
        assert verdict.missing_slots == (SlotKind.TEST_SET,)
        with pytest.raises(PackRefusal) as raised:
            verdict.raise_if_invalid()
        assert raised.value.code == PACK_INCOMPLETE
        assert "no test set" in raised.value.message
        assert raised.value.details["missing_slots"] == ["test_set"]

    def test_three_missing_slots_are_all_named(self) -> None:
        """FAC-12. An operator fixing one at a time learns nothing from a report that stops."""
        pack = fx.definition(slots={})
        verdict = validate(pack, {})
        assert verdict.missing_slots == REQUIRED_SLOT_ORDER
        with pytest.raises(PackRefusal) as raised:
            verdict.raise_if_invalid()
        assert raised.value.details["missing_slots"] == ["prompt_set", "test_set", "retention_set"]
        for name in ("prompt set", "test set", "retention set"):
            assert name in raised.value.message

    def test_the_missing_slots_are_reported_in_a_stable_order(self) -> None:
        pack = fx.definition(slots={SlotKind.TEST_SET.value: fx.contract_slot()})
        verdict = validate(pack, all_resolved(pack))
        assert verdict.missing_slots == (SlotKind.PROMPT_SET, SlotKind.RETENTION_SET)


class TestTheRetentionFloorIsItsOwnReason:
    def test_a_filled_retention_slot_without_a_floor_is_not_a_missing_slot(self) -> None:
        """E14: the operator supplies a number in one case and a dataset in the other."""
        pack = fx.definition(slots=fx.valid_slots(retention_set=fx.retention_slot(floor=None)))
        verdict = validate(pack, all_resolved(pack))
        assert verdict.floor_missing is True
        assert SlotKind.RETENTION_SET not in verdict.missing_slots
        assert verdict.state == "incomplete"
        with pytest.raises(PackRefusal) as raised:
            verdict.raise_if_invalid()
        assert raised.value.code == RETENTION_FLOOR_MISSING


class TestTheThreeFailuresAreKeptApart:
    """The single most important distinction in the feature (FTDD §5, FR-007.9).

    ``midataworks_not_configured`` says *the service is absent*;
    ``pack_part_unresolved`` says *your pack is wrong*. A standalone operator must never be
    told the second when the first is true - that is what R-02.50's convention exists for,
    and the FTDD calls it *"the single most likely thing to be implemented carelessly"*.
    """

    def test_an_absent_part_is_reported_as_unresolved(self) -> None:
        pack = fx.definition(slots=fx.valid_slots())
        resolutions = all_resolved(pack)
        resolutions[SlotKind.TEST_SET] = Resolution(
            ref=ContractPartRef(type="test-set", name="notary-test", version="1"),
            state="not_found",
            rendered="test-set/notary-test@1",
            reason="no part in this installation matches it",
        )
        verdict = validate(pack, resolutions)
        assert [item.slot for item in verdict.unresolved] == [SlotKind.TEST_SET]
        assert verdict.unavailable == ()
        with pytest.raises(PackRefusal) as raised:
            verdict.raise_if_invalid()
        assert raised.value.code == PACK_PART_UNRESOLVED

    def test_an_absent_service_is_reported_separately_and_names_the_service(self) -> None:
        pack = fx.definition(slots=fx.valid_slots())
        resolutions = all_resolved(pack)
        resolutions[SlotKind.PROMPT_SET] = Resolution(
            ref=ContractPartRef(type="prompt-set", name="support-tickets", version="3"),
            state="service_unavailable",
            rendered="prompt-set/support-tickets@3",
            service="midataworks",
            reason="miDataworks is not configured",
        )
        verdict = validate(pack, resolutions)
        assert [item.slot for item in verdict.unavailable] == [SlotKind.PROMPT_SET]
        assert verdict.unresolved == ()
        assert verdict.unavailable_services == ("midataworks",)
        with pytest.raises(PackRefusal) as raised:
            verdict.raise_if_invalid()
        assert raised.value.code == MIDATAWORKS_NOT_CONFIGURED
        assert raised.value.names_an_absent_service
        assert "Nothing is wrong " in raised.value.message

    def test_a_missing_slot_outranks_an_absent_service(self) -> None:
        """Order is a decision, and this one was got wrong once while the module was written.

        With an absent service reported first, a pack genuinely missing its test set came
        back as ``midataworks_not_configured`` - hiding an authoring defect behind an
        infrastructure message. A missing slot is a fact about the pack alone and is
        actionable without any service, so it is named first.
        """
        slots = fx.valid_slots()
        del slots[SlotKind.TEST_SET.value]
        pack = fx.definition(slots=slots)
        resolutions = all_resolved(pack)
        resolutions[SlotKind.PROMPT_SET] = Resolution(
            ref=ContractPartRef(type="prompt-set", name="support-tickets", version="3"),
            state="service_unavailable",
            rendered="prompt-set/support-tickets@3",
            service="midataworks",
        )
        verdict = validate(pack, resolutions)
        with pytest.raises(PackRefusal) as raised:
            verdict.raise_if_invalid()
        assert raised.value.code == PACK_INCOMPLETE
        # And the absent service is still **reported**, in its own field, so nothing is lost
        # by the ordering - the payload carries both facts.
        assert verdict.payload()["service_unavailable"][0]["service"] == "midataworks"

    def test_a_complete_pack_whose_parts_cannot_be_reached_names_the_service(self) -> None:
        """The hazard the convention exists for: a pack that is fine, reported as broken.

        Nothing is missing here - every required slot is filled - and the only reason it
        cannot start is that no service can resolve its parts. **That** is where the 503
        belongs.
        """
        pack = fx.definition(slots=fx.valid_slots())
        resolutions = all_resolved(pack)
        for slot in list(resolutions):
            resolutions[slot] = Resolution(
                ref=ContractPartRef(type=slot.value, name="x", version="1"),
                state="service_unavailable",
                rendered=f"{slot.value}/x@1",
                service="midataworks",
            )
        with pytest.raises(PackRefusal) as raised:
            validate(pack, resolutions).raise_if_invalid()
        assert raised.value.code == MIDATAWORKS_NOT_CONFIGURED

    def test_the_payload_keeps_the_two_in_separate_fields(self) -> None:
        """A reader branches on a field, never on a substring of a message."""
        pack = fx.definition(slots=fx.valid_slots())
        resolutions = all_resolved(pack)
        resolutions[SlotKind.PROMPT_SET] = Resolution(
            ref=ContractPartRef(type="prompt-set", name="p", version="1"),
            state="service_unavailable",
            rendered="prompt-set/p@1",
            service="midataworks",
        )
        payload = validate(pack, resolutions).payload()
        assert payload["unresolved"] == []
        assert payload["service_unavailable"][0]["service"] == "midataworks"

    def test_a_filled_slot_with_no_resolution_attempted_is_not_treated_as_resolved(self) -> None:
        """An absent answer is not a yes - ADR-018's rule applied to a mapping lookup."""
        pack = fx.definition(slots=fx.valid_slots())
        verdict = validate(pack, {})
        assert len(verdict.unresolved) == 3
        assert verdict.state == "incomplete"


class TestValidityDoesNotDependOnADryRun:
    def test_nothing_in_the_verdict_mentions_a_selector_outcome(self) -> None:
        """A selector that accepts everything is advice to the author, not an invalid pack.

        FR-007.20's panel is advisory and task 10.4 requires validity to be unchanged by any
        dry-run outcome; the structural form of that is that a verdict has no field for one.
        """
        pack = fx.definition(slots=fx.valid_slots())
        payload = validate(pack, all_resolved(pack)).payload()
        assert not any("accept" in key or "dry" in key for key in payload)
