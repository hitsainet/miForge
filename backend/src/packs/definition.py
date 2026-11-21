"""The pack artifact  (task 3.1; FR-007.2, FR-007.7, R-02.38).

``docs/schemas/pack-definition-v1.json`` is **normative** for a pack version and is
foundation task 3.4's, not this feature's. This module builds documents that conform to it,
validates against it on write, export and import, and refuses a non-conforming document
naming what failed. **Nothing here restates the field list in prose** - the schema is loaded
and used, because the only way to agree with a file is to read it.

## Seven slots, each stated even when null

The schema's ``slots`` object requires all seven keys and ``additionalProperties: false``.
That is deliberate and it is the same discipline as the nullable ``domain``: **"no retention
set" must be an assertion, not a silence.** An absent key is an omission, a null value is a
fact, and collapsing the two is the defect that decision exists to prevent. So
:class:`PackVersionDefinition` always emits seven keys.

## What this feature discovered the contract does not carry

Reported rather than worked around, because every one of them is a requirement asserting a
field its own normative schema lacks - this project's signature defect, now at ten
recorded instances:

* **``local_version``** (FPRD OQ-2, FR-007.23). The slot's ``external`` is
  ``batch-record-v1.json#/$defs/ExternalProvenance``, which is closed and carries
  ``{source, content_sha256, retrieved_at, licence}``. So a pack document **cannot** store a
  miForge-local ordinal, and one that lived only in this installation's database would not
  survive the export AC10 requires. **The content hash is the pin.** The ordinal is
  therefore not invented at all - see :class:`packs.references.ExternalPartRef`.
* **``derived_from``** (FR-007.12) and **``is_example``** (FR-007.16). The document's
  top-level object is closed on seven fields and ``mf_packs`` has no column for either, so
  they live in index columns added by migration ``0015`` (written by
  :func:`packs.registry.clone` and :func:`packs.registry.mark_example`).
* **``addressed_through``** (FR-007.25). ``PackSlot`` is closed on
  ``{contract_reference, external, floor}``, so cross-pack addressing is not expressible
  *in the artifact*. It is stored in the index instead (``mf_pack_parts.addressed_through_
  pack_id``, written by :func:`packs.registry.record_addressing`) and rendered from there.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from core.canonical import sha256_of
from packs import contracts
from packs.errors import PACK_DOCUMENT_INVALID, RETENTION_FLOOR_MISSING, PackRefusal
from packs.references import PART_REF_ADAPTER, ContractPartRef, ExternalPartRef, PartRef
from packs.slots import FLOOR_BEARING_SLOTS, SLOT_ORDER, SlotKind

#: The one ``schema_version`` this code writes and accepts. A document of any other version
#: is **refused by name** rather than read optimistically: a future pack may mean something
#: different by the same field, and reading it as though it did not is how a silently wrong
#: run input happens (E9).
SCHEMA_VERSION = "1"

#: The field that carries the document's own digest, excluded from the bytes that are
#: hashed. Named once so the exclusion and the write cannot disagree.
DIGEST_FIELD = "definition_sha256"


class FilledSlot(BaseModel):
    """One filled slot: a reference, and a floor where the slot bears one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref: PartRef
    #: The retention floor, in 0..1 - **miForge's own value** (FR-007.5). Verified absent
    #: from miDataworks' contract: they own the set, this project owns what counts as an
    #: unacceptable regression on it. Same range as both manifests' ``ScoreRecord.floor``.
    floor: float | None = Field(default=None, ge=0.0, le=1.0)


class PackVersionDefinition(BaseModel):
    """A pack version, as the artifact and the API carry it.

    ``frozen``: a committed version's slots are frozen (FR-007.2), and R-02.35's guarantee
    is that *"a pack version names a reproducible run input"*. A version whose contents can
    change afterwards does not name anything.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The pack's slug identity. The schema's pattern, not a copy of it in prose: validated
    #: against the loaded file by :meth:`document`'s conformance check.
    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    version: str = Field(min_length=1)
    #: **NULLABLE BUT REQUIRED.** Null asserts the pack is not domain-scoped - a general
    #: pretraining pack, for which a domain is meaningless - and an absent key is an
    #: omission. Pydantic carries no "absent" here because the model always emits the key;
    #: :meth:`from_document` is where an absent key is refused.
    domain: str | None = None
    slots: Mapping[SlotKind, FilledSlot | None]
    created_at: AwareDatetime

    def filled(self) -> dict[SlotKind, FilledSlot]:
        """The slots that hold a reference, in :data:`packs.slots.SLOT_ORDER`.

        A loop rather than a comprehension with a guard: MyPy strict cannot narrow
        ``self.slots[slot]`` from a ``get(...) is not None`` test in a comprehension, and the
        honest fix is to bind the value once rather than to silence the checker - the
        narrowing is the thing being asserted.
        """
        result: dict[SlotKind, FilledSlot] = {}
        for slot in SLOT_ORDER:
            value = self.slots.get(slot)
            if value is not None:
                result[slot] = value
        return result

    def document(self) -> dict[str, Any]:
        """The conforming document, digest included, **validated before it is returned**.

        The digest covers the document minus :data:`DIGEST_FIELD`, through the one canonical
        function (ADR-005) - the same bytes a hash would be taken over and the same bytes
        :func:`core.canonical.write_canonical` writes, so an export is verifiable and two
        canonicalisers never become two answers.

        Raises:
            PackRefusal: ``pack_document_invalid`` if the result does not conform. A writer
                that produced a non-conforming document would be caught by the reader
                eventually; caught here, it is caught by the thing that made it.
        """
        body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "version": self.version,
            "domain": self.domain,
            "slots": {slot.value: slot_document(self.slots.get(slot)) for slot in SLOT_ORDER},
            "created_at": _iso(self.created_at),
        }
        document = {**body, DIGEST_FIELD: sha256_of(body)}
        validate_document(document)
        return document

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> PackVersionDefinition:
        """Read a document - from an export, an import or the pack store on disk.

        Checks the declared version, validates against the schema, verifies the digest, and
        only then builds. Validation before construction, always: the model below indexes
        into ``slots`` and reads ``created_at``, both of which the schema has guaranteed are
        present, and a Pydantic error about a missing key is a worse diagnosis than the
        schema's JSON pointer.

        Raises:
            PackRefusal: ``pack_document_invalid`` for a non-conforming document, a
                future ``schema_version``, or a digest that does not match the body -
                naming the schema and what failed (E9).
        """
        # THE VERSION CHECK COMES FIRST, before schema validation, and deliberately so.
        # The schema's `schema_version` is a `const`, so a future document fails validation
        # anyway - but it fails as *"'2' was expected to be '1'"* at a JSON pointer, which
        # tells the operator nothing about what to do. Checked here first, the refusal names
        # both versions and says why reading it optimistically would be worse than refusing.
        declared = str(document.get("schema_version"))
        if declared != SCHEMA_VERSION:
            raise PackRefusal(
                PACK_DOCUMENT_INVALID,
                f"This pack document declares schema_version {declared!r} and this "
                f"installation writes and reads {SCHEMA_VERSION!r}. Refused rather than "
                f"read optimistically: a later version may mean something different by the "
                f"same field, and a silently misread pack is a silently wrong run input.",
                schema=contracts.PACK_DEFINITION_SCHEMA,
                declared_version=declared,
                supported_version=SCHEMA_VERSION,
            )

        validate_document(document)

        body = {key: value for key, value in document.items() if key != DIGEST_FIELD}
        recomputed = sha256_of(body)
        if recomputed != document[DIGEST_FIELD]:
            raise PackRefusal(
                PACK_DOCUMENT_INVALID,
                "This pack document's definition_sha256 does not match its own body, so "
                "either it was edited after it was written or it was written by something "
                "that does not use the one canonical function. Re-export it from the "
                "installation that holds it.",
                schema=contracts.PACK_DEFINITION_SCHEMA,
                recorded=str(document[DIGEST_FIELD]),
                recomputed=recomputed,
            )

        slots: dict[SlotKind, FilledSlot | None] = {}
        raw_slots = document["slots"]
        for slot in SLOT_ORDER:
            slots[slot] = slot_from_document(slot, raw_slots.get(slot.value))
        return cls(
            name=str(document["name"]),
            version=str(document["version"]),
            domain=None if document["domain"] is None else str(document["domain"]),
            slots=slots,
            created_at=document["created_at"],
        )


def validate_document(document: Mapping[str, Any]) -> None:
    """Validate against ``pack-definition-v1.json``, refusing with the failing pointers.

    Raises:
        PackRefusal: ``pack_document_invalid``, with every failure listed - not the first.
            An author fixing one at a time learns nothing from a report that stops early,
            which is the same argument AC10 makes about missing slots.
    """
    failures = [
        f"{'/' + '/'.join(str(part) for part in error.absolute_path)}: {error.message}"
        for error in sorted(
            contracts.validator(contracts.PACK_DEFINITION_SCHEMA).iter_errors(dict(document)),
            key=lambda error: list(error.path),
        )
    ]
    if failures:
        raise PackRefusal(
            PACK_DOCUMENT_INVALID,
            f"This document is not a conforming pack definition: "
            f"{'; '.join(failures[:5])}"
            f"{' (and more)' if len(failures) > 5 else ''}. "
            f"The schema is {contracts.PACK_DEFINITION_SCHEMA} and it is the authority.",
            schema=contracts.PACK_DEFINITION_SCHEMA,
            errors=failures,
        )


def slot_document(slot: FilledSlot | None) -> dict[str, Any] | None:
    """One slot as the document carries it, or ``null`` for an empty slot.

    Both ``contract_reference`` and ``external`` are always present and exactly one is
    non-null. An external part's ``contract_reference`` is ``null`` - never an empty object
    and never a synthesised reference (R-02.40, FR-007.24).
    """
    if slot is None:
        return None
    ref = slot.ref
    return {
        "contract_reference": (
            ref.as_contract_dict() if isinstance(ref, ContractPartRef) else None
        ),
        "external": ref.as_external_dict() if isinstance(ref, ExternalPartRef) else None,
        "floor": slot.floor,
    }


def _both_or_neither(reference: Any) -> str:
    """Which of the two shapes a refusal is about, in the operator's words."""
    if reference is not None:
        return "both a contract reference and external provenance"
    return "neither a contract reference nor external provenance"


def slot_from_document(slot: SlotKind, raw: Any) -> FilledSlot | None:
    """Read one slot, refusing anything the composition cannot mean.

    Three refusals, each for a document that **conforms** to the schema and is still not a
    usable composition - structural validity and semantic validity are different layers
    (FTDD §4.5):

    * **neither source present** - an unattributed input makes the run unreproducible and
      nothing downstream could say where a behaviour came from;
    * **both sources present** - two answers about where one part came from;
    * **a floor on a slot that bears none** - the floor is the retention set's threshold
      (R-02.38), and a floor on the prompt set is a number with no meaning that some later
      reader would act on.
    """
    if raw is None:
        return None
    reference = raw.get("contract_reference")
    external = raw.get("external")
    if (reference is None) == (external is None):
        raise PackRefusal(
            PACK_DOCUMENT_INVALID,
            f"The {slot.value!r} slot carries "
            f"{_both_or_neither(reference)}"
            f". Exactly one is required: a part with neither is unattributed, and a part "
            f"with both gives two answers about where it came from.",
            schema=contracts.PACK_DEFINITION_SCHEMA,
            slot=slot.value,
        )

    floor = raw.get("floor")
    if floor is not None and slot not in FLOOR_BEARING_SLOTS:
        raise PackRefusal(
            PACK_DOCUMENT_INVALID,
            f"The {slot.value!r} slot carries a floor of {floor}. A floor is the retention "
            f"set's threshold and means nothing on any other slot; refusing "
            f"rather than ignoring it, because a number that is ignored is a number some "
            f"later reader will act on.",
            schema=contracts.PACK_DEFINITION_SCHEMA,
            slot=slot.value,
            floor=floor,
        )

    ref: PartRef
    if reference is not None:
        ref = PART_REF_ADAPTER.validate_python({"source": "midataworks", **reference})
    else:
        ref = PART_REF_ADAPTER.validate_python(
            {
                "source": "external",
                "origin": external["source"],
                "content_sha256": external["content_sha256"],
                "taken_in_at": external["retrieved_at"],
            }
        )
    return FilledSlot(ref=ref, floor=floor)


def require_floor(definition: PackVersionDefinition) -> None:
    """Refuse a filled retention slot with no floor (E14, R-02.38).

    .. warning::

       **This is not the wiring, and it must not become a second one.** The rule it states
       is already computed by :func:`packs.validation.validate` as ``floor_missing`` and
       raised by :meth:`packs.validation.PackValidity.raise_if_invalid`, which is wired at
       **both** ends that matter: :func:`packs.gate.require_valid_pack` (step 2 of
       R-02.53's validator, so a run cannot start from a floorless pack) and
       :func:`packs.authoring.commit`. *Measured, not read*: a pack committed with a filled
       retention slot and no floor is refused ``retention_floor_missing`` with
       ``details.slot == "retention_set"`` - byte for byte what this function raises.

       So a reachability sweep that reports this function as an unwired capability is
       reporting a **duplicate**, not a gap. Giving it a caller would create two sources of
       truth for one rule, which is the defect this project has already paid for once:
       *"the validator runs first, so it was unreachable for the two cases it existed to
       catch, and an unreachable guard reads as protection while protecting nothing."*
       ``test_pack_floor_is_enforced_live.py`` pins the live path and carries a tripwire
       that goes red if this function ever gains a production caller.

    Raises:
        PackRefusal: ``retention_floor_missing``. **A distinct refusal from a missing
            retention set**, because the operator's fix is different: one supplies a set,
            the other supplies a number.
    """
    slot = definition.slots.get(SlotKind.RETENTION_SET)
    if slot is not None and slot.floor is None:
        raise PackRefusal(
            RETENTION_FLOOR_MISSING,
            "This pack's retention set has no floor. The retention set "
            "and its floor are one artifact: without the floor nothing defines an "
            "unacceptable regression, and the run would have no condition to stop on. "
            "Set a floor between 0 and 1 on the retention slot.",
            slot=SlotKind.RETENTION_SET.value,
        )


def run_input_projection(definition: PackVersionDefinition) -> dict[str, Any]:
    """What *"the same run inputs"* means, operationally (FAC-2, AC10's second half).

    The slot set, each slot's source kind and **pinned identity**, and the floor. Compared
    after an export/import round trip through the real writer and the real reader.

    For an external part the pin is ``content_sha256`` and ``taken_in_at`` is **excluded**:
    the digest identifies the bytes, while the retrieval time is when *this* installation saw
    them. Comparing the time would make the round trip compare two installations' histories
    rather than the document that travelled.
    """
    projection: dict[str, Any] = {"name": definition.name, "version": definition.version}
    projection["domain"] = definition.domain
    for slot in SLOT_ORDER:
        filled = definition.slots.get(slot)
        if filled is None:
            projection[slot.value] = None
            continue
        ref = filled.ref
        pinned: dict[str, Any] = (
            ref.as_contract_dict()
            if isinstance(ref, ContractPartRef)
            else {"origin": ref.origin, "content_sha256": ref.content_sha256}
        )
        projection[slot.value] = {"source": ref.source, "pinned": pinned, "floor": filled.floor}
    return projection


def _iso(value: AwareDatetime) -> str:
    """UTC ISO-8601 with a ``Z``, matching every other timestamp miForge writes."""
    return value.isoformat().replace("+00:00", "Z")
