"""The run-start gate  (task 3.4; FR-007.10, R-02.38, FAC-10).

**A run may start only from a valid pack.** One guard, one refusal path, exposed for the
run-creation path (F6) to call: the guard is F7's and the call site is F6's. F6's own submit
order puts it **first** - *"the gate first, because every later step costs something and a
run from an invalid pack must not get as far as a GPU resolution"* - and R-02.53's validation
makes it step 2 of six, before any allocation.

## It re-resolves, every time, and does not read a stored flag

``mf_packs.valid`` exists and this gate ignores it. A part can stop resolving between
authoring and starting, and a cache hit on a deleted part is precisely the failure the gate
exists to catch. The resolution cost is one batched lookup against the local index - cheap
against what starting a run costs.

## Reachability, stated as an obligation rather than assumed

ADR-016 #1: *a capability is not shipped until a test FAILS when its wiring is removed.* So
this function is exported from :mod:`packs` **and** listed on
:class:`packs.service.PackService`, and the reachability test asserts it is present **on the
live service interface** rather than importing this module. The cross-feature half - that
F6's run-create path actually calls it, with the right payload and the right call count - is
F7's task 13.2 and is **owed**, not assumed: a grep proving a call exists is the weaker form
and it is not sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from packs import registry, store
from packs.definition import PackVersionDefinition
from packs.records import PackVersionIdentity, PartReferenceRecord, part_reference_record
from packs.resolver import Resolution, resolve_slots
from packs.slots import SLOT_ORDER, SlotKind
from packs.validation import PackValidity, validate


@dataclass(frozen=True, slots=True)
class ResolvedPack:
    """A valid pack version with its slots resolved. **The view F2, F4 and F6 share.**

    Returned from the gate so the caller gets something useful out of the check rather than a
    bare "yes" - and so three features consume **one** resolution rather than writing three
    resolvers, which is the seam this object exists to be.
    """

    definition: PackVersionDefinition
    resolutions: dict[SlotKind, Resolution]
    validity: PackValidity

    @property
    def identity(self) -> PackVersionIdentity:
        return PackVersionIdentity(pack_id=self.definition.name, version=self.definition.version)

    def slot(self, kind: SlotKind) -> Resolution | None:
        """The resolution for one slot, or ``None`` when the slot is empty.

        ``None`` means *empty*, not *unresolved*: an empty optional slot is a pack saying it
        has no corpus, which is a fact, and a caller that conflated the two would treat a
        valid pack as broken.
        """
        return self.resolutions.get(kind)

    def floor(self) -> float | None:
        """The retention floor this pack pins, if it has a retention slot.

        **The pack carries the canonical floor** (R-02.38); a run may override it and the
        override is recorded **on the run, not the pack** (FPRD OQ-8, still the operator's
        decision). F4's ``resolve_floor`` already models exactly that with its
        ``pack_slot_floor`` and ``run_override_floor`` layers, so this returns the pack's
        layer and nothing else.
        """
        slot = self.definition.slots.get(SlotKind.RETENTION_SET)
        return None if slot is None else slot.floor

    def part_records(self) -> list[PartReferenceRecord]:
        """One :mod:`packs.records` record per filled slot, in slot order.

        **The only route to a ``parts`` element**, for F8's adapter writer and anything else
        that records what a run drew on. Produced from the resolved view rather than from a
        bare reference, so three features cannot disagree about what a part was.
        """
        identity = self.identity
        return [
            part_reference_record(slot, filled.ref, pack=identity)
            for slot in SLOT_ORDER
            if (filled := self.definition.slots.get(slot)) is not None
        ]

    def payload(self) -> dict[str, Any]:
        """The resolved view as a response or a run-definition record carries it."""
        return {
            "pack_id": self.definition.name,
            "version": self.definition.version,
            "domain": self.definition.domain,
            "validity": self.validity.payload(),
            "retention_floor": self.floor(),
            "parts": [record.element() for record in self.part_records()],
        }


def require_valid_pack(session: Session, pack_id: str, version: str) -> ResolvedPack:
    """Refuse unless the pack version is valid; return it resolved. **Re-resolves.**

    Args:
        session: An open session.
        pack_id: The pack's slug.
        version: The pack version.

    Returns:
        The :class:`ResolvedPack` the caller then uses - so the check and the read are one
        call and a caller cannot do the second without the first.

    Raises:
        PackRefusal: ``midataworks_not_configured`` (503) when the service that could
            resolve a part is absent - **nothing is wrong with the pack**;
            ``pack_incomplete`` (422) naming **every** missing slot;
            ``retention_floor_missing`` (422); ``pack_part_unresolved`` (422) naming the
            slot and the reference.
        Refusal: ``pack_not_found`` when there is no such version.
    """
    definition = store.read(pack_id, version)
    resolutions = resolve_slots(session, definition)
    validity = validate(definition, resolutions)
    # Refuse first, return second. A caller that received a ResolvedPack and was expected to
    # check `validity` itself would be a caller that forgets.
    validity.raise_if_invalid()
    return ResolvedPack(definition=definition, resolutions=resolutions, validity=validity)


def resolved_pack(session: Session, pack_id: str, version: str) -> ResolvedPack:
    """The resolved view **without** the gate's refusal, for a read-only consumer.

    F2's source assembly and F4's scoring call this when they already hold a run that
    started - the gate ran at submit, and refusing again mid-run would turn a part that went
    missing into a crash rather than a stop condition. Anything *starting* work calls
    :func:`require_valid_pack` instead.
    """
    definition = store.read(pack_id, version)
    resolutions = resolve_slots(session, definition)
    return ResolvedPack(
        definition=definition,
        resolutions=resolutions,
        validity=validate(definition, resolutions),
    )


def usage(session: Session, pack_id: str, version: str) -> tuple[str, ...]:
    """The runs that used this pack version (FR-007.13). Read-only over F1's and F6's rows."""
    return registry.usage_map(session).get(registry.row_id(pack_id, version), ())
