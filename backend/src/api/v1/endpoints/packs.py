"""The pack registry endpoints  (FR-007.28, FR-007.11 - FR-007.16; ADR-013).

The registry, one pack version, its validity and lineage, export and import.

## Two distinctions the route layer carries, and both are easy to get wrong

**``GET …/validity`` returns 200 describing invalidity.** Asking whether a pack is valid is a
*successful question*; the refusals fire where an **action** is attempted - starting a run,
committing a draft. A route that returned 422 for "this pack is incomplete" would make the
Domain packs screen's own listing an error page.

**``midataworks_not_configured`` is 503 and ``pack_part_unresolved`` is 422.** One says *the
service is absent*, the other *your pack is wrong*, and a standalone operator must never be
told the second when the first is true. The status mapping is :mod:`api.app`'s table, which
already maps the first by name **and** by the ``_not_configured`` suffix rule, so this module
sets no status of its own and raises no ``HTTPException``: a refusal raised three calls down
arrives in one shape, and a bare ``HTTPException`` here would put a second error shape on the
same API.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from api.agent_origin import requires_approval_when_agent
from api.deps import session_dependency
from core.errors import refuse
from packs.definition import PackVersionDefinition, slot_from_document
from packs.service import pack_service
from packs.slots import SLOT_ORDER, SlotKind
from services import approval_gates

router = APIRouter(prefix="/packs", tags=["packs"])

SessionDep = Annotated[Session, Depends(session_dependency)]


class PackListing(BaseModel):
    """``GET /api/v1/packs``.

    ``prompt_count`` is ``null`` when it cannot be resolved - **never 0** (ADR-018,
    FR-007.11) - and the UI renders null as "unknown". ``parts_filled`` travels with
    ``parts_total`` so every number names its scale and sample (ADR-012).
    """

    model_config = ConfigDict(extra="forbid")

    packs: list[dict[str, Any]]


class CreatePackRequest(BaseModel):
    """``POST /api/v1/packs`` - commit a pack version directly (FR-007.12).

    The seven-step flow (``/pack-drafts``) is the path for a human; this is the path for a
    client that already has the composition - a script, a test, or F9's New run form offering
    *"save these choices as a pack"*. Both end in the same writer, so there is one way a pack
    version comes into existence.

    ``domain`` is **nullable and required by the artifact**: null asserts the pack is not
    domain-scoped and an absent key is an omission. Here the field defaults to ``None``,
    which is the null assertion - a client that means "general pretraining" needs no special
    spelling, and one that forgot a domain gets a pack that says it has none rather than a
    pack with a missing key.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = "1"
    domain: str | None = None
    #: ``{slot kind: slot document | null}``. Slots left out are ``null`` - *stated*, because
    #: the artifact states all seven even when empty.
    slots: dict[str, Any] = Field(default_factory=dict)
    #: ``{slot kind: "<pack>@<version>"}`` - the OTHER pack each slot's part is addressed
    #: through (FR-007.25, cross-pack mixing). Index-only: the artifact cannot carry it.
    addressed_through: dict[str, str] = Field(default_factory=dict)


class CloneRequest(BaseModel):
    """``POST /api/v1/packs/{pack_id}/versions/{version}/clone`` (FR-007.12)."""

    model_config = ConfigDict(extra="forbid")

    into: str = Field(min_length=1)
    version: str = "1"


@router.get("", response_model=PackListing)
def list_packs(session: SessionDep) -> PackListing:
    """Every pack version, with validity, counts, usage and derivation (FR-007.11)."""
    return PackListing(packs=pack_service.listing(session))


@router.post("", status_code=201)
def create_pack(request: CreatePackRequest, session: SessionDep) -> dict[str, Any]:
    """Commit a pack version directly (FR-007.12).

    Writes the file first and indexes it second, so a crash between the two leaves a pack a
    rebuild recovers rather than a row pointing at a pack that does not exist - ADR-004's
    "where they disagree the filesystem wins" only works if the filesystem is written first.
    """
    definition = PackVersionDefinition(
        name=request.name,
        version=request.version,
        domain=request.domain,
        slots={
            slot: slot_from_document(slot, request.slots.get(slot.value)) for slot in SLOT_ORDER
        },
        created_at=datetime.now(tz=UTC),
    )
    try:
        through = {SlotKind(slot): pack for slot, pack in request.addressed_through.items()}
    except ValueError as exc:
        raise refuse(
            "pack_slot_unknown",
            f"addressed_through names a slot that is not one of the seven: {exc}.",
            known=[slot.value for slot in SLOT_ORDER],
        ) from exc
    created = pack_service.create(session, definition, addressed_through=through)
    return pack_service.get(session, created.name, created.version)


@router.get("/{pack_id}/versions/{version}")
def get_pack_version(pack_id: str, version: str, session: SessionDep) -> dict[str, Any]:
    """One pack version's registry line."""
    return pack_service.get(session, pack_id, version)


@router.get("/{pack_id}/versions/{version}/validity")
def get_validity(pack_id: str, version: str, session: SessionDep) -> dict[str, Any]:
    """**200 describing invalidity** (FR-007.8, FR-007.9).

    Names **every** missing slot, not the first: an operator fixing one at a time learns
    nothing from a report that stops early, and that completeness is the whole of AC10.
    Reports an absent service in its own field, so nothing reads an unconfigured sibling as
    a fault in the pack.
    """
    return pack_service.validity(session, pack_id, version).payload()


@router.get("/{pack_id}/versions/{version}/lineage")
def get_lineage(pack_id: str, version: str, session: SessionDep) -> dict[str, Any]:
    """Where every part came from, in full (FR-007.26) - the audit record of US-8."""
    return {"slots": pack_service.lineage(session, pack_id, version)}


@router.get("/{pack_id}/versions/{version}/export")
def export_pack(pack_id: str, version: str, session: SessionDep) -> Response:
    """The pack document, **by value**, as the canonical bytes (FR-007.14).

    The same bytes ``definition_sha256`` was taken over, so an export is verifiable by
    hashing what was received. Served as an attachment rather than as a JSON body because it
    is a document an operator keeps: the durable, portable form of a pack.

    ``session`` is taken for consistency with the rest of the surface and because a future
    export may want the resolution facts; the document itself is read from the store, which
    is the authority (ADR-004).
    """
    del session
    return Response(
        content=pack_service.export_bytes(pack_id, version),
        media_type="application/json",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{pack_service.export_filename(pack_id, version)}"'
            )
        },
    )


@router.post("/import", status_code=201)
def import_pack(
    session: SessionDep,
    document: Annotated[dict[str, Any], Body(description="An exported pack document.")],
) -> dict[str, Any]:
    """Import an exported pack (FR-007.15, AC10's second half).

    **Never rewrites a reference to make it resolve** (E8): a pack whose parts this
    installation cannot resolve is imported, stored and *reported* unresolvable, because
    rewriting would make the imported pack a different pack - and AC10's claim is that the
    imported version yields the **same** run inputs.
    """
    definition = pack_service.import_document(session, document)
    return pack_service.get(session, definition.name, definition.version)


@router.post("/{pack_id}/versions/{version}/clone", status_code=201)
def clone_pack(
    pack_id: str, version: str, request: CloneRequest, session: SessionDep
) -> dict[str, Any]:
    """A new pack pinning the same parts (FR-007.12).

    References, not resolution facts: a clone is a new composition over the same parts, and
    copying a row count into it would be copying a measurement taken at another time.
    """
    clone = pack_service.clone(
        session, pack_id, version, into=request.into, new_version=request.version
    )
    return pack_service.get(session, clone.name, clone.version)


@router.delete("/{pack_id}/versions/{version}", status_code=204)
@requires_approval_when_agent(
    "delete",
    validate=approval_gates.validate_pack_delete,
    summary=approval_gates.summarise_pack_delete,
)
def delete_pack(pack_id: str, version: str, session: SessionDep) -> Response:
    """Delete a pack version, **refusing while a run cites it** (E11).

    Deleting a pack deletes **no part**: miForge does not own them, and an external part's
    files stay where the operator put them.
    """
    pack_service.delete(session, pack_id, version)
    return Response(status_code=204)
