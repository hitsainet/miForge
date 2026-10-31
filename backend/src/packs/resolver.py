"""Resolution, with the three failures kept apart  (task 8.x; FR-007.9, FR-007.25, E5).

Resolving a slot means confirming the referenced part exists at the pinned version and
retrieving the facts the UI and a run need from it - row count, content hash, licence class
where present. **Three failures, three states**, because they have three different fixes
(see :mod:`packs.validation` for the table).

## Through F10's published entry point, and nowhere else

Contract parts resolve through :func:`services.parts.resolve.resolve_parts` - **one batched
lookup** for all of a pack's references, which is what §9's registry budget assumes and what
keeps a six-slot pack from being six chances for the answer to change half way through. F7
adds **no outbound call site** (ADR-010): F10 owns the one module that talks to miDataworks,
and this module talks to F10.

F10's lookup is deliberately generous about what it answers and deliberately silent about
what to do: it *"never raises for a missing or unpinnable part - the caller decides, because
the refusal that fits depends on what the caller was doing."* F7 is that caller, and these
are its decisions:

* **missing, with miDataworks configured** → ``not_found``. The part is gone or was never
  published; the pack must change.
* **missing, with miDataworks unconfigured** → ``service_unavailable``, naming the service.
  **Nothing is wrong with the pack.** This is the single most important line in the module:
  a standalone operator told their pack is broken will edit a pack that is fine, and
  R-02.50's convention exists to prevent exactly that.
* **indexed with a null ``contract_reference``** → ``part_unpinnable``, raised at
  **binding** rather than at resolution (:func:`bind_indexed_part`). A pack pins
  ``{type, name, version}``, so by the time a slot exists the reference is already there;
  the place an unpinnable part can actually arrive is an operator choosing one from the part
  listing. The refusal's spelling comes from F10 - it builds the refusal so the code is
  spelled once across the seam - and F7 raises it.

## Nothing here synthesises anything

``rows`` and ``content_sha256`` are ``None`` when unknown, **never 0** and never a
``getattr(x, k, 0)`` (ADR-018): a prompt count of 0 because a service is unreachable is a
fabricated value, and a fabricated value is worse than a refusal because it is silent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from clients import availability
from packs import external as external_parts
from packs.definition import PackVersionDefinition
from packs.references import ContractPartRef, ExternalPartRef, PartRef
from packs.rendering import contract_form, external_form
from packs.slots import SLOT_ORDER, SlotKind
from services.parts import resolve as part_lookup

#: The service name used in a ``service_unavailable`` resolution and in the
#: ``midataworks_not_configured`` refusal built from it. ADR-010's spelling, written once.
MIDATAWORKS = "midataworks"

ResolutionState = Literal["resolved", "not_found", "service_unavailable"]


class Resolution(BaseModel):
    """What resolving one slot found. Facts, or a named absence - never a substitute."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref: PartRef
    state: ResolutionState
    #: The inner rendering of the reference, for a refusal's ``details`` and a log line.
    #: **Never the display form** (FR-007.27.2): a display form in a refusal payload is a
    #: display form a client could store.
    rendered: str
    #: miForge's part id, when the part is in the index. ``None`` when nothing matched.
    part_id: str | None = None
    #: The part's directory, as the index records it - what a panel reads rows from. ``None``
    #: when the part did not resolve, which is why a sampler refuses rather than reading an
    #: empty path.
    local_path: str | None = None
    #: Row count. ``None`` when unknown - never 0 (ADR-018).
    rows: int | None = None
    #: The part's content digest: a contract part's ``manifest_sha256``, an external part's
    #: own ``content_sha256``.
    content_sha256: str | None = None
    licence_classes: tuple[str, ...] = ()
    #: F10's caveat objects, **carried verbatim** - not summarised, not reduced to a
    #: boolean, not flattened to a string. Their shape is miDataworks' and normalising it
    #: would be miForge taking an opinion on someone else's vocabulary; FR-010.17 requires
    #: them carried, not understood. *(Typed `str` at first here, which is how the shape was
    #: discovered: the real index returns `{"code", "detail", ...}`.)* A caveat must be
    #: visible at the point of use, and a pack slot is a point of use.
    caveats: tuple[dict[str, Any], ...] = ()
    #: Set **only** when ``state == "service_unavailable"``. Its presence is what a caller
    #: branches on, rather than a substring of ``reason``.
    service: str | None = None
    reason: str | None = None


def resolve_slots(
    session: Session | None, definition: PackVersionDefinition
) -> dict[SlotKind, Resolution]:
    """Resolve every filled slot of ``definition`` in **one** lookup per source kind.

    Args:
        session: An open session, or ``None`` for an installation with no database
            reachable. ``None`` is honest rather than convenient: external parts still
            resolve from the filesystem, and contract parts are reported as
            ``service_unavailable`` naming the index rather than as missing.
        definition: The pack version.

    Returns:
        One :class:`Resolution` per filled slot, keyed by slot. Empty slots are **absent
        from the mapping**, because "this slot has nothing to resolve" and "this slot
        resolved" are different facts and :func:`packs.validation.validate` distinguishes
        them.
    """
    resolutions: dict[SlotKind, Resolution] = {}
    contract_slots: dict[SlotKind, ContractPartRef] = {}

    for slot in SLOT_ORDER:
        filled = definition.slots.get(slot)
        if filled is None:
            continue
        ref = filled.ref
        if isinstance(ref, ExternalPartRef):
            resolutions[slot] = _resolve_external(session, ref)
        else:
            contract_slots[slot] = ref

    if contract_slots:
        resolutions.update(_resolve_contract(session, contract_slots))
    return resolutions


def _resolve_external(session: Session | None, ref: ExternalPartRef) -> Resolution:
    """An external part, resolved **locally and by digest**.

    This path must work with the miDataworks client absent entirely - not merely stubbed -
    because a pack of wholly external parts is the standalone case AC20 pins (FR-007.31).
    Nothing it touches reaches a sibling: the material was taken in locally and the index
    row is local, so a ``not_found`` here is never reported as a service absence. Built and
    tested first for that reason.
    """
    rendered = external_form(ref)
    if session is None:
        return Resolution(
            ref=ref,
            state="not_found",
            rendered=rendered,
            reason="no database session is available, so the part index cannot be read",
        )
    record = external_parts.lookup_by_digest(session, ref.content_sha256)
    if record is None:
        return Resolution(
            ref=ref,
            state="not_found",
            rendered=rendered,
            reason=(
                f"no external part in this installation carries digest "
                f"{ref.content_sha256[:12]}… (taken in from {ref.origin!r}). Import the "
                f"material as a contract-conformant export, or pin a part that is here"
            ),
        )
    return Resolution(
        ref=ref,
        state="resolved",
        rendered=rendered,
        part_id=record.part_id,
        local_path=record.local_path,
        rows=record.rows,
        content_sha256=record.content_sha256,
        licence_classes=record.licence_classes,
        caveats=record.caveats,
    )


def _resolve_contract(
    session: Session | None, slots: Mapping[SlotKind, ContractPartRef]
) -> dict[SlotKind, Resolution]:
    """Contract parts, in **one** batched lookup through F10's entry point."""
    configured = availability.is_configured()

    if session is None:
        return {
            slot: _unavailable(
                ref,
                reason=(
                    "no database session is available, so the part index cannot be read; "
                    "this is an infrastructure absence, not a fault in the pack"
                ),
            )
            for slot, ref in slots.items()
        }

    targets = [part_lookup.reference_from(ref.as_contract_dict()) for ref in slots.values()]
    lookup = part_lookup.resolve_parts(session, targets)

    resolved: dict[SlotKind, Resolution] = {}
    for slot, ref in slots.items():
        key = contract_form(ref)
        found = lookup.pinned.get(key)
        if found is not None:
            resolved[slot] = _resolved_from(ref, found, key)
            continue
        unpinnable = lookup.unpinnable.get(key)
        if unpinnable is not None:  # pragma: no cover - unreachable via a reference lookup
            # Kept, and documented as unreachable rather than presented as a guard: a
            # lookup BY reference cannot return a part whose reference is null. The place an
            # unpinnable part arrives is `bind_indexed_part`, which is where the refusal is
            # raised and tested.
            resolved[slot] = Resolution(
                ref=ref,
                state="not_found",
                rendered=key,
                part_id=unpinnable.part_id,
                reason="the indexed part carries no contract reference",
            )
            continue
        if configured:
            resolved[slot] = Resolution(
                ref=ref,
                state="not_found",
                rendered=key,
                reason=(
                    f"no part in this installation matches {key}; import the version from "
                    f"miDataworks, or pin one that is here"
                ),
            )
        else:
            # THE LINE THAT KEEPS A STANDALONE OPERATOR FROM BEING TOLD THEIR PACK IS
            # BROKEN. With no miDataworks configured there is no way to tell "this version
            # was never published" from "this installation has not imported it yet", and
            # only one of those is the pack's fault. Reporting the service by name sends the
            # operator to the fix that exists.
            resolved[slot] = _unavailable(
                ref,
                reason=(
                    f"{key} is not in this installation's part index and miDataworks is not "
                    f"configured, so there is nowhere to resolve it from"
                ),
            )
    return resolved


def _resolved_from(ref: ContractPartRef, found: Any, key: str) -> Resolution:
    """Build a resolution from F10's :class:`services.parts.resolve.ResolvedPart`.

    Reads the facts off ``payload`` - F10's one rendering of a part - rather than
    re-deriving them, so the registry listing and F10's own part listing cannot disagree
    about a row count. ``row_count`` is read as ``None`` when the key is absent, never
    defaulted to 0.
    """
    payload: Mapping[str, Any] = found.payload
    rows = payload.get("row_count")
    return Resolution(
        ref=ref,
        state="resolved",
        rendered=key,
        part_id=found.part_id,
        local_path=payload.get("local_path"),
        rows=None if rows is None else int(rows),
        content_sha256=payload.get("manifest_sha256"),
        licence_classes=tuple(payload.get("licence_classes") or ()),
        caveats=tuple(payload.get("caveats") or ()),
    )


def _unavailable(ref: ContractPartRef, *, reason: str) -> Resolution:
    return Resolution(
        ref=ref,
        state="service_unavailable",
        rendered=contract_form(ref),
        service=MIDATAWORKS,
        reason=reason,
    )


def bind_indexed_part(session: Session, part_id: str) -> ContractPartRef:
    """Turn an indexed part into a pinnable reference, or refuse by name (E5, OQ-3).

    This is where an operator's choice from the part listing becomes a pack slot, and it is
    the **only** place ``part_unpinnable`` can arise: a published version whose
    ``target.contract_reference`` is ``null`` is **conformant** - the field is
    ``anyOf [ContractReference, null]`` *and* in ``required``, so a null is an assertion
    miDataworks deliberately permits - and F10 records that null faithfully. It names nothing
    reproducible, so it cannot pin a pack slot.

    **Nothing synthesises a reference** from the version id, the dataset name or the
    directory name. That would be miForge inventing contract identity, which R-02.40
    forbids and which is the defect this project has recorded three times.

    Raises:
        Refusal: ``part_unpinnable`` (F10's spelling, raised here) or
            ``dataset_version_unresolved`` when nothing in the index matches.
    """
    resolved = part_lookup.resolve_one(session, part_id)
    reference = resolved.contract_reference
    if reference is None:  # pragma: no cover - `resolve_one` refuses first; see below
        # Defence in depth, and documented as such: `resolve_one` raises
        # `part_unpinnable` itself for this case. The branch stays because "unreachable"
        # is a claim about another module staying as it is, and the cost is three lines.
        raise part_lookup.PartLookup(unpinnable={part_id: resolved}).unpinnable_refusal()
    return ContractPartRef(
        type=str(reference.type), name=str(reference.name), version=str(reference.version)
    )
