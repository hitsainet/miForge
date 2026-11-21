"""The ONE producer of a ``parts`` element  (task 2.7; FR-007.32; R-02.35, R-02.40).

Another artifact's ``parts`` array carries one element per part a run drew on. F8's adapter
manifest has such an array and **refuses** (``pack_part_shape_unavailable``) rather than
spelling the element itself, so F7 not publishing this blocks F8's adapter writer. This is
the side that owns it.

## The element shape is the GATE CONTRACT'S, and that corrects this feature's own documents

F7's FPRD (FR-007.32), FTDD (§4.1a) and FTID (§3.2a) each specify the element as
``{slot, pack, source, type, name, version}`` / ``{slot, pack, source, origin,
content_sha256, taken_in_at, local_version}``. **That shape is unbuildable**, and the check
that shows it is the one this project keeps earning: parse the contract.

``batch-record-v1.json`` - foundation task 3.2's gate contract - already settles it, in
``$defs.PartReference``::

    {"kind": <one of the seven slots>, "contract_reference": {type,name,version}|null,
     "external": {source,content_sha256,retrieved_at,licence?}|null}

with all three keys **required** and ``additionalProperties: false``. Its own description
says why it is there: *"This element shape is the seam two features each deferred to the
other ... It is settled here because it is a gate contract."* So:

* the documents' ``slot`` is the contract's **``kind``**;
* the documents' flattened ``type``/``name``/``version`` belong **inside**
  ``contract_reference``, and flattening them makes the element invalid;
* the documents' ``source`` discriminator is expressed by **which of the two is non-null** -
  and both keys are stated on every element, so "no contract reference" is an assertion
  rather than a silence, which is the same discipline as the pack's nullable ``domain``;
* the documents' ``pack`` has **nowhere to go**: the element is closed, and
  ``adapter-manifest-v1.json`` has no top-level ``pack`` field either. Reported to the
  coordinator rather than smuggled in. It stays on :class:`PartReferenceRecord` where
  lineage and display read it, so nothing is lost in-process - only the artifact cannot yet
  carry it, and that is a schema decision for the operator (FPRD OQ-4 becomes live).

**A second producer already exists**, and that is the other half of the correction:
``training/signals/parts_reader.part_reference()`` (F2) emits this element for a batch
record. F7 is therefore not *the only* producer in the tree, as FR-007.32 claims - it is the
**pack-side** producer, and the two must agree. A test asserts they emit the same key set
and that both validate against the subschema read out of the file.

## What goes into an artifact is :meth:`PartReferenceRecord.element`, not ``model_dump``

:meth:`~PartReferenceRecord.element` is the only artifact form. ``model_dump()`` gives the
in-process record, pack identity and all, and would **fail** validation as a
``PartReference``. The names are chosen so a caller cannot reach for the wrong one by
accident, and the conformance test validates what ``element()`` actually returns.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.errors import refuse
from packs import contracts
from packs.references import ContractPartRef, ExternalPartRef, PartRef
from packs.slots import SlotKind


class PackVersionIdentity(BaseModel):
    """Which pack version a part was addressed through.

    R-02.44's reason for an adapter carrying its parts at all is that *"an adapter is small,
    portable and routinely moved between installations, so it has to say what it was trained
    on without its run log travelling alongside"* - and a pack version is the name of that
    answer (R-02.35: *"a pack version names a reproducible run input"*). It rides on the
    record for exactly that purpose; see the module docstring for why it cannot yet ride in
    the artifact.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pack_id: str = Field(min_length=1)
    version: str = Field(min_length=1)

    def render(self) -> str:
        """``<pack_id>@<version>``. The pack's own identity, never a part's reference."""
        return f"{self.pack_id}@{self.version}"


class PartReferenceRecord(BaseModel):
    """One part, with the slot that gave it its role and the pack it came through.

    ``frozen`` because a record that can be mutated after production is not a record of
    anything; ``extra="forbid"`` because an element that can grow a key is an element two
    writers can disagree about.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The slot kind - R-02.38's *"one slot per part kind"*. **A pack is not a bag of
    #: parts**: without this the array cannot say whether a set was the test set or the
    #: retention set, which is the one thing a travelling adapter most needs.
    slot: SlotKind
    pack: PackVersionIdentity
    #: The pinned reference, discriminated on ``source``. Holding the typed reference rather
    #: than copies of its fields is what keeps FR-007.24 true here: there is no flattened
    #: member an external part's origin could be read out of as a contract ``type``.
    ref: PartRef

    @property
    def source(self) -> str:
        """``midataworks`` or ``external`` - the discriminator, for a caller that branches."""
        return self.ref.source

    def element(self) -> dict[str, Any]:
        """The element an artifact's ``parts`` array carries. **The only artifact form.**

        Returns exactly ``batch-record-v1.json#/$defs/PartReference``: ``kind``, plus both
        of ``contract_reference`` and ``external`` with exactly one of them non-null. Both
        keys are always present because the contract requires both - an **absent** key is an
        omission while a **null** one is an assertion, and collapsing the two is the defect
        the nullable-``domain`` decision exists to prevent.

        An external part's ``contract_reference`` is ``null`` - not an empty string, not a
        synthesised reference (R-02.40, FR-007.24). Nothing here can promote one kind to the
        other, because neither reference type offers a conversion.
        """
        return {
            "kind": self.slot.value,
            "contract_reference": (
                self.ref.as_contract_dict() if isinstance(self.ref, ContractPartRef) else None
            ),
            "external": (
                self.ref.as_external_dict() if isinstance(self.ref, ExternalPartRef) else None
            ),
        }


def part_reference_record(
    slot: SlotKind, ref: PartRef, *, pack: PackVersionIdentity
) -> PartReferenceRecord:
    """Build the record for one filled slot. **The pack-side producer, called by F8.**

    Args:
        slot: The slot the part fills.
        ref: The pinned reference, of either source kind.
        pack: The pack version the part is addressed through.

    Returns:
        The record. Its :meth:`PartReferenceRecord.element` is what F8 writes into
        ``adapter_manifest.json``'s ``parts``; nothing constructs such an element inline.

    **Write no second producer and no ``to_dict()`` shortcut** - not on the reference types,
    not on a slot, not in ``transfer.py``, not in a test helper that "builds one quickly".
    F8's mutation control constructs an element inline and must go red; that control is only
    meaningful because there is one producer to bypass.
    """
    return PartReferenceRecord(slot=slot, pack=pack, ref=ref)


def _assert_element_matches_the_contract() -> None:
    """:meth:`PartReferenceRecord.element` emits exactly the contract's keys. **At import.**

    Checked against the file, for both source kinds, by running the real method - not by
    comparing two lists of key names, which would agree with each other by construction.

    Raises:
        Refusal: ``parts_element_drifted`` naming the difference.
    """
    subschema = contracts.part_reference_subschema()
    expected = set(subschema.get("properties", {}))
    required = set(subschema.get("required", []))
    pack = PackVersionIdentity(pack_id="import-time-self-check", version="1")
    samples = (
        part_reference_record(
            SlotKind.PROMPT_SET,
            ContractPartRef(type="prompt_set", name="self-check", version="1"),
            pack=pack,
        ),
        part_reference_record(
            SlotKind.CORPUS,
            ExternalPartRef(
                origin="(import-time self-check)",
                content_sha256="0" * 64,
                taken_in_at="2026-01-01T00:00:00Z",  # type: ignore[arg-type]
            ),
            pack=pack,
        ),
    )
    for record in samples:
        keys = set(record.element())
        if keys != expected or not required <= keys:
            raise refuse(
                "parts_element_drifted",
                f"batch-record-v1.json's PartReference declares {sorted(expected)} "
                f"(required {sorted(required)}) and packs/records.py emits {sorted(keys)}. "
                f"The element shape is a gate contract's and this module produces it; "
                f"failing at import rather than writing an element the batch record would "
                f"reject.",
                contract_properties=sorted(expected),
                contract_required=sorted(required),
                ours=sorted(keys),
            )


_assert_element_matches_the_contract()
