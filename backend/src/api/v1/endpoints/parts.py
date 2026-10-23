"""``/api/v1/parts`` - taking parts in and reading the index  (F10 task 6.x; FR-010.12, .19).

**Thin routes.** Every decision is in a service: the availability guard, the validator, the
kind map, the confinement check. A guard in a route is bypassed by the worker calling the
service directly, which is exactly how a capability comes back unguarded (F10 task 6.6), so
the guard lives in :mod:`clients.availability` and the acquisition module calls it.

**``POST /parts/validate`` is a first-class route, not a debug helper.** Its product *is*
the refusal: an operator fixing an export needs the failing JSON Pointer, the keyword and
what was expected, without creating a part. It shares the **same validator function** as
ingest - not a parallel one, because a validator used only by a diagnostic is a validator
that drifts from the one that matters.

**``POST /parts/import`` returns 202 with a job id.** Import streams digests over split
files that may be gigabytes; a request that blocks on that is a request that times out. The
synchronous path exists too, named ``wait=true``, for a small part and for the tests - and
it runs *the same function* the worker runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from api.deps import session_dependency
from core.config import get_settings
from core.errors import refuse
from services import archive
from services.parts import acquire, index
from services.parts.contract_validator import failures, validate
from services.parts.kind_map import PART_KINDS

router = APIRouter(prefix="/parts", tags=["parts"])


class ImportRequest(BaseModel):
    """What to import, and from where.

    ``extra="forbid"``: a typo'd field name should refuse rather than be ignored. An
    ignored ``source`` would silently take the default path, which is the difference
    between "needs miDataworks" and "does not".
    """

    model_config = ConfigDict(extra="forbid")

    #: ``dir`` or ``http``. ``hub`` is accepted by the model and refused by name at the
    #: service (``acquisition_path_not_built``), because a 422 "not a valid enumeration
    #: member" would not say *why* it is absent or what to do instead.
    source: str = Field(description="dir | http | hub")
    #: For ``source=dir``: the export directory. For ``source=http``: the export id.
    locator: str = Field(min_length=1)
    #: ``midataworks_contract`` (the default) or ``external_files``. An external part's
    #: contract-reference columns stay null (R-02.40) and its own provenance is required.
    recorded_source: str = acquire.SOURCE_CONTRACT
    #: The gate contract's ``ExternalProvenance`` object, for an external part.
    external_provenance: dict[str, Any] | None = None
    #: Run the import now rather than queueing it. For a small part and for tests.
    wait: bool = False


class ValidateRequest(BaseModel):
    """A candidate manifest to validate without importing."""

    model_config = ConfigDict(extra="forbid")

    manifest: dict[str, Any]


@router.post("/import", summary="Take a part in from an export directory or over HTTP")
def import_part(
    body: Annotated[ImportRequest, Body()],
    session: Annotated[Session, Depends(session_dependency)],
    response: Response,
) -> dict[str, Any]:
    """Import a part. ``202`` with a job id, or ``200`` with the summary when ``wait``.

    Raises:
        Refusal: ``midataworks_not_configured`` for ``source=http`` while unset;
            ``acquisition_path_not_built`` for ``source=hub``; any of the contract's
            refusals naming what failed.
    """
    settings = get_settings()
    parts_root = Path(settings.parts_root)

    if body.source not in acquire.PATHS:
        raise refuse(
            "part_source_unknown",
            f"{body.source!r} is not an acquisition path. miForge imports from "
            f"{sorted(acquire.PATHS)}.",
            given=body.source,
            known=sorted(acquire.PATHS),
        )

    if not body.wait:
        from services.parts.tasks import enqueue_import

        job_id = enqueue_import(
            source=body.source,
            locator=body.locator,
            recorded_source=body.recorded_source,
            external_provenance=body.external_provenance,
        )
        response.status_code = 202
        return {
            "job_id": job_id,
            "source": body.source,
            "locator": body.locator,
            # Said plainly, because a 202 whose caller does not know where to look is a
            # 202 that gets polled against the wrong thing.
            "poll": "GET /api/v1/parts to see the part once the job completes",
        }

    result = _run_import(session, body, parts_root=parts_root)
    return {
        "part_id": result.part_id,
        "created": result.created,
        "part_kind": result.part_kind,
        "manifest_sha256": result.manifest_sha256,
        "splits": list(result.splits),
        # Carried on the import's own answer, not only on the part row: a caveat must be
        # visible at the point of use (FR-010.17), and an import's result is the first
        # such point.
        "caveats": [dict(caveat) for caveat in result.caveats],
        "licence_classes": list(result.licence_classes),
    }


def _run_import(session: Session, body: ImportRequest, *, parts_root: Path) -> Any:
    """The one import dispatch. **Shared with the worker**, so the two cannot diverge."""
    if body.source == "dir":
        return acquire.from_dir(
            session,
            Path(body.locator),
            parts_root=parts_root,
            source=body.recorded_source,
            external_provenance=body.external_provenance,
        )
    if body.source == "http":
        return acquire.from_http(session, body.locator, parts_root=parts_root)
    return acquire.from_hub()


@router.post(
    "/validate", summary="Validate a manifest without importing - the refusal is the product"
)
def validate_manifest(
    body: Annotated[ValidateRequest, Body()],
) -> dict[str, Any]:
    """Validate a candidate ``midataworks.dataset-version/v1`` document.

    Returns ``{"valid": true, ...}`` for a conformant document, and **refuses** otherwise
    with every failure: the failing JSON Pointer, the schema keyword, the schema path and
    what was expected. Same validator as ingest.
    """
    found = failures(body.manifest)
    if found:
        from services.parts.contract_validator import invalid_refusal

        raise invalid_refusal(found, source="POST /api/v1/parts/validate")

    version = validate(body.manifest, source="POST /api/v1/parts/validate")
    from services.parts.kind_map import resolve

    kind = resolve(version)
    reference = version.contract_reference
    return {
        "valid": True,
        "part_kind": kind.part_kind,
        "target_kind": kind.target_kind,
        "content_kind": kind.content_kind,
        # `null` when the version carries none. Conformant, and never synthesised: F7
        # refuses to bind such a part to a pack slot with `part_unpinnable`.
        "contract_reference": None if reference is None else reference.as_dict(),
        "rendered": None if reference is None else reference.render(),
        "splits": [split.name for split in version.splits],
        "caveats": [caveat.as_dict() for caveat in version.caveats],
        "licence_classes": list(version.licence_classes),
    }


@router.get("", summary="The part index")
def list_parts(
    session: Annotated[Session, Depends(session_dependency)],
    part_kind: Annotated[str | None, Query(description=f"one of {sorted(PART_KINDS)}")] = None,
    include_archived: Annotated[
        bool, Query(description="also list archived parts (hidden by default)")
    ] = False,
) -> dict[str, Any]:
    """Every indexed part, with everything needed to render a row in **one** call.

    If a UI has to make a second call to show a part's caveats or its contract reference,
    the shape is wrong (F10 FTID §6).

    **Archived parts are left out unless ``include_archived``** (operator 2026-10-20), and
    every row says ``archived_at`` (null = listed). The archive is applied here, over the
    index's own answer, so :func:`services.parts.index.list_parts` - the seam the reindex
    equivalence test compares - stays a picture of the disk.
    """
    if part_kind is not None and part_kind not in PART_KINDS:
        raise refuse(
            "part_kind_unknown",
            f"{part_kind!r} is not a part kind. miForge's kinds are {sorted(PART_KINDS)}.",
            given=part_kind,
            known=sorted(PART_KINDS),
        )
    archived = archive.archived_parts(session)
    items = [
        {**item, "archived_at": archived.get(str(item["id"]))}
        for item in index.list_parts(session, part_kind=part_kind)
    ]
    if not include_archived:
        items = [item for item in items if item["archived_at"] is None]
    return {"items": items, "total": len(items), "include_archived": include_archived}


@router.get("/{part_id}", summary="One part, with the manifest winning over the index")
def get_part(
    part_id: str,
    session: Annotated[Session, Depends(session_dependency)],
) -> dict[str, Any]:
    """One part.

    Read through :func:`services.parts.index.resolve_part`, so the **manifest wins** over
    the index and any disagreement is logged (ADR-004, foundation 4.5). A read that trusted
    the row would make the index the authority, which is the inversion ADR-004 forbids.
    """
    part = index.resolve_part(session, part_id)
    # An archived part still answers by id; it says when it was archived.
    return {**part, "archived_at": archive.part_archived_at(session, part_id)}
