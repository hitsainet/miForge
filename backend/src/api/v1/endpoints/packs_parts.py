"""Turning an indexed part into a pinnable slot reference  (FR-007.6, FR-007.23, E5).

One read route. It is the step between *"the operator picked a part from the listing"* and
*"the slot holds a pinned reference"*, and it is where the two source kinds separate for good:

* a **miDataworks** part becomes a ``{type, name, version}`` contract reference;
* an **external** part becomes its own provenance - ``{origin, content_sha256, taken_in_at}``.

**There is no ingestion route here**, and that is deliberate rather than missing. FR-007.23
reads as though F7 takes files in; BRD-02's 2026-10-06 amendment accepts files or a directory
*"only as a contract-conformant export"*, and **F10 has built exactly that** - digest
verification, traversal confinement, provenance in the gate contract's shape, and
``POST /api/v1/parts/import`` as the route. A second ingestion here would be a second hasher
and a second provenance writer for the same material.

**A part whose ``contract_reference`` is null is refused as ``part_unpinnable``** (E5). Such a
version is **conformant** - the field is ``anyOf [ContractReference, null]`` *and* in
``required``, so a null is an assertion miDataworks deliberately permits - and F10 records
that null faithfully. It names nothing reproducible, so it cannot pin a slot, and **nothing
synthesises a reference** from the version id or the dataset name.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.deps import session_dependency
from core.errors import Refusal
from packs import external as external_parts
from packs.references import ContractPartRef
from packs.rendering import contract_form, external_form
from packs.resolver import bind_indexed_part

router = APIRouter(prefix="/pack-part-refs", tags=["packs"])

SessionDep = Annotated[Session, Depends(session_dependency)]


@router.get("/{part_id}")
def pinnable_reference(part_id: str, session: SessionDep) -> dict[str, Any]:
    """The pinnable reference for an indexed part, with its own inner rendering.

    The response's ``ref`` is the document a slot carries, discriminated on ``source`` - so a
    client cannot put it in a slot without the discriminator travelling with it, which is
    what keeps an external part from being read as a contract reference one layer on
    (FR-007.24).

    ``display_form`` is **not** in this response. The display layer prefixes the pack a part
    is addressed **through**, and that is not known here: a part may be addressed through a
    different pack from the one holding the slot (FR-007.25), so the renderer takes the pack
    as an argument and this route has none to give. Inventing one would be the layer
    confusion R-02.35 forbids.
    """
    try:
        # The contract path first: `bind_indexed_part` is the one place `part_unpinnable`
        # is raised, and its refusal is the right answer for an indexed miDataworks version
        # that carries no reference.
        reference: Any = bind_indexed_part(session, part_id)
    except Refusal as refusal:
        if refusal.code != "part_unpinnable":
            raise
        # An EXTERNAL part reaches F10's lookup as "unpinnable" too, because that lookup
        # classifies on `contract_reference is not None` and an external part has none by
        # construction - its CHECK constraint forbids one. So the external branch is tried
        # here rather than treating the refusal as final.
        #
        # **This is a finding against F10's published entry point, not a workaround.** Its
        # own docstring says a pack slot may be pinned "an external part by the id of the row
        # that carries its provenance", and `resolve_one` refuses every such id. Reported;
        # the right fix is a provenance-aware outcome there, and F7 does not edit that file.
        reference = external_parts.ref_for_part(session, part_id)

    inner = (
        contract_form(reference)
        if isinstance(reference, ContractPartRef)
        else external_form(reference)
    )
    return {
        "part_id": part_id,
        "ref": reference.model_dump(mode="json"),
        "source": reference.source,
        "inner_form": inner,
    }
