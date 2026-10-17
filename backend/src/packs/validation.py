"""Validity, and the three failures that are not the same failure  (task 3.2; FR-007.8, FR-007.9).

**AC10's entire content is the wording.** A pack with a missing required slot is reported
*incomplete, naming the slot* - never "validation failed". So this module **collects** and
never short-circuits: an operator fixing one slot at a time learns nothing from a report
that stops at the first.

:func:`validate` is **pure**. It takes a definition and a resolution map and returns a
verdict, performing no I/O, so every combination is a unit test with no stub at all.
Resolution is :mod:`packs.resolver`'s job and happens before this is called.

## Three failures, reported three ways, because they have three different fixes

=============================  ====================================================
``pack_incomplete``            A required slot is empty. **The operator fills it.**
``pack_part_unresolved``       A reference does not resolve at its pinned version.
                               The part is gone or was never published; **the pack
                               must change.**
``midataworks_not_configured``  The service that could resolve it is absent.
                               **Nothing is wrong with the pack**; configuring the
                               service fixes it.
=============================  ====================================================

Collapsing the third into the second would tell an operator with no miDataworks that their
packs are broken - exactly the "silently degraded into a different behaviour" the standalone
reframe forbids, and the thing R-02.50's convention exists for. So an absent service lands
in its own field of the verdict and **nothing folds it into the missing-slot list.**

A filled retention slot with no floor is its own reason, ``retention_floor_missing``, and not
"missing retention set": the operator supplies a number in one case and a dataset in the
other.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from packs.definition import PackVersionDefinition
from packs.errors import (
    MIDATAWORKS_NOT_CONFIGURED,
    PACK_INCOMPLETE,
    PACK_PART_UNRESOLVED,
    RETENTION_FLOOR_MISSING,
    PackRefusal,
)
from packs.resolver import Resolution
from packs.slots import FLOOR_BEARING_SLOTS, REQUIRED_SLOT_ORDER, SLOT_ORDER, SlotKind


@dataclass(frozen=True, slots=True)
class UnresolvedSlot:
    """One slot whose reference did not resolve, with **which** reference and why.

    ``service`` is set only when the cause is an absent service, and that is how the
    503-vs-422 split survives into the verdict: a reader branches on a field rather than on
    a substring of a message.
    """

    slot: SlotKind
    reference: str
    reason: str
    service: str | None = None


@dataclass(frozen=True, slots=True)
class PackValidity:
    """Whether a pack version can start a run, and if not, precisely what is wrong."""

    state: Literal["valid", "incomplete"]
    #: **Every** missing required slot, in :data:`packs.slots.REQUIRED_SLOT_ORDER` - never
    #: the first one found (FAC-12).
    missing_slots: tuple[SlotKind, ...] = ()
    #: Filled slots whose part could not be resolved because the part is absent.
    unresolved: tuple[UnresolvedSlot, ...] = ()
    #: Filled slots whose part could not be resolved because **a service is absent**. Kept
    #: apart from ``unresolved`` so no caller can iterate past the distinction - the same
    #: reason F10's lookup keeps its three outcomes in three collections.
    unavailable: tuple[UnresolvedSlot, ...] = ()
    #: A filled retention slot with no floor (E14).
    floor_missing: bool = False

    @property
    def unavailable_services(self) -> tuple[str, ...]:
        """The absent services, by name, deduplicated and sorted."""
        return tuple(sorted({slot.service for slot in self.unavailable if slot.service}))

    @property
    def is_valid(self) -> bool:
        return self.state == "valid"

    def payload(self) -> dict[str, object]:
        """The 200 body of ``GET …/validity`` (FTDD §5).

        A **200 describing invalidity**, because asking whether a pack is valid is a
        successful question. The refusals fire where an *action* is attempted.
        """
        return {
            "state": self.state,
            "missing_slots": [slot.value for slot in self.missing_slots],
            "unresolved": [
                {"slot": item.slot.value, "reference": item.reference, "reason": item.reason}
                for item in self.unresolved
            ],
            "service_unavailable": [
                {
                    "slot": item.slot.value,
                    "reference": item.reference,
                    "service": item.service,
                    "reason": item.reason,
                }
                for item in self.unavailable
            ],
            "retention_floor_missing": self.floor_missing,
        }

    def raise_if_invalid(self) -> None:
        """Refuse with the code that fits, in the order that helps most.

        **The order is a decision, not an accident**, and it was got wrong once while this
        module was being written - with an absent service reported first, a pack that was
        genuinely missing its test set came back as ``midataworks_not_configured``, hiding an
        authoring defect behind an infrastructure message.

        So: **a missing slot first.** It is a fact about the pack alone, actionable without
        any service, and naming it can never mislead. Then the floor, for the same reason.
        Only then an absent service - which is what keeps a *complete* pack from being
        reported as incomplete when the real cause is that nothing can resolve its parts,
        which is the hazard R-02.50's convention exists for. An unresolved part comes last
        because chasing a part that has gone is the most expensive fix of the four.

        Raises:
            PackRefusal: ``pack_incomplete`` (422), ``retention_floor_missing`` (422),
                ``midataworks_not_configured`` (503) or ``pack_part_unresolved`` (422).
        """
        if self.missing_slots:
            raise PackRefusal(
                PACK_INCOMPLETE,
                f"This pack is incomplete: no "
                f"{', no '.join(slot.value.replace('_', ' ') for slot in self.missing_slots)}"
                f". A run cannot start from it. Fill "
                f"{'that slot' if len(self.missing_slots) == 1 else 'those slots'} and "
                f"validate again.",
                missing_slots=[slot.value for slot in self.missing_slots],
            )
        if self.floor_missing:
            raise PackRefusal(
                RETENTION_FLOOR_MISSING,
                "This pack's retention set has no floor, so nothing defines an "
                "unacceptable regression and the run has no condition to stop on. Set a "
                "floor between 0 and 1 on the retention slot.",
                slot=SlotKind.RETENTION_SET.value,
            )
        if self.unavailable:
            services = self.unavailable_services
            raise PackRefusal(
                MIDATAWORKS_NOT_CONFIGURED,
                f"This pack's part(s) in slot(s) "
                f"{[item.slot.value for item in self.unavailable]} come from "
                f"{', '.join(services)}, which is not configured here. **Nothing is wrong "
                f"with the pack**: set that service's base URL to resolve its parts, or "
                f"build a pack from external parts, which needs no sibling at all.",
                service=services[0] if services else None,
                services=list(services),
                capability="pack_part_resolution",
                slots=[item.slot.value for item in self.unavailable],
            )
        if self.unresolved:
            raise PackRefusal(
                PACK_PART_UNRESOLVED,
                f"This pack's part(s) in slot(s) "
                f"{[item.slot.value for item in self.unresolved]} do not resolve at their "
                f"pinned versions: {'; '.join(item.reason for item in self.unresolved)}. "
                f"The part is absent or was never published here - import it, or pin a "
                f"different version. The pack must change.",
                slots=[
                    {"slot": item.slot.value, "reference": item.reference}
                    for item in self.unresolved
                ],
            )


def validate(
    definition: PackVersionDefinition, resolutions: Mapping[SlotKind, Resolution]
) -> PackValidity:
    """Whether this pack version can start a run. **Pure; collects every problem.**

    Args:
        definition: The pack version.
        resolutions: One :class:`packs.resolver.Resolution` per **filled** slot, from
            :func:`packs.resolver.resolve_slots`. A filled slot with no entry is treated as
            unresolved rather than as resolved - *an absent answer is not a yes*, which is
            the ADR-018 rule applied to a mapping lookup.

    Returns:
        The verdict. ``valid`` only when every required slot is filled, every filled slot
        resolved, and a filled retention slot carries its floor.
    """
    missing = tuple(slot for slot in REQUIRED_SLOT_ORDER if definition.slots.get(slot) is None)

    unresolved: list[UnresolvedSlot] = []
    unavailable: list[UnresolvedSlot] = []
    for slot in SLOT_ORDER:
        filled = definition.slots.get(slot)
        if filled is None:
            continue
        resolution = resolutions.get(slot)
        if resolution is None:
            unresolved.append(
                UnresolvedSlot(
                    slot=slot,
                    reference=filled.ref.model_dump_json(),
                    reason="no resolution was attempted for this slot",
                )
            )
            continue
        if resolution.state == "resolved":
            continue
        item = UnresolvedSlot(
            slot=slot,
            reference=resolution.rendered,
            reason=resolution.reason or resolution.state,
            service=resolution.service,
        )
        if resolution.state == "service_unavailable":
            unavailable.append(item)
        else:
            unresolved.append(item)

    retention = definition.slots.get(SlotKind.RETENTION_SET)
    floor_missing = (
        retention is not None
        and SlotKind.RETENTION_SET in FLOOR_BEARING_SLOTS
        and retention.floor is None
    )

    state: Literal["valid", "incomplete"] = (
        "valid" if not (missing or unresolved or unavailable or floor_missing) else "incomplete"
    )
    return PackValidity(
        state=state,
        missing_slots=missing,
        unresolved=tuple(unresolved),
        unavailable=tuple(unavailable),
        floor_missing=floor_missing,
    )
