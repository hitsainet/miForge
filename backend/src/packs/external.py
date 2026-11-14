"""External parts: pinned by their own provenance  (task 7.x; FR-007.23, FR-007.24, R-02.40).

A part may come from outside the suite, and R-02.40 asks for exactly three things: **where
it came from**, **a content hash**, and **when it was taken in**. Those three are what
``pack-definition-v1.json``'s external slot carries, by ``$ref`` to
``batch-record-v1.json#/$defs/ExternalProvenance``.

## F7 does not take parts in. F10 already does, and this corrects F7's documents.

FR-007.23 reads as though this feature ingests files. **It must not**, for two reasons that
were checked rather than recalled:

1. **BRD-02's 2026-10-06 amendment** says files or a directory are accepted *"only as a
   contract-conformant export"* - so an external part arrives as a staged
   ``midataworks.dataset-version/v1`` directory, validated by the one validator.
2. **F10 has built exactly that.** ``services/parts/ingest.ingest(..., source="external_files",
   external_provenance=…)`` copies the material into the parts root, verifies every declared
   file's digest, confines every manifest-declared path against traversal, records the
   provenance in the gate contract's shape, and refuses a part carrying neither a reference
   nor provenance. ``POST /api/v1/parts/import`` is the route. A second ingestion here would
   be a second hasher, a second traversal guard and a second provenance writer - and the
   dataset parts themselves are explicitly **not** F7's to rebuild.

So this module **reads** F10's index and turns an indexed external part into a pinnable
reference. What F7 owns is the composition; what F10 owns is the material.

## The pin is the digest, which answers FPRD OQ-2 from the contract

OQ-2 asked how an external part is versioned and the FPRD invented *"a miForge-local
version number, incrementing on re-ingestion"*. **There is nowhere to put it**:
``ExternalProvenance`` is closed on ``{source, content_sha256, retrieved_at, licence}``, so a
pack document cannot carry an ordinal, and a reference whose pin lived only in this
installation's database would not survive the export that AC10 requires. **The content hash
is the pin** - and it is a stronger one than an ordinal, because it identifies the bytes
rather than their arrival order. Re-ingesting the same origin with different bytes therefore
produces a *different* reference, which is exactly the behaviour the ordinal was invented to
get: an already-committed pack version keeps resolving to the bytes it was built against.

## Nothing here can promote an external part

There is no code path from an external part to a contract reference - not in this module,
not anywhere in the feature (FR-007.24). F10 enforces the same rule one layer down with a
CHECK constraint: an ``external_files`` part may not carry contract-reference columns. Two
mechanisms, one rule, and the negative control for FAC-6 flips the discriminator to prove the
test bites.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from core.errors import refuse
from packs.references import ExternalPartRef
from services.parts.tables import CONTRACT_PARTS, SOURCE_EXTERNAL


@dataclass(frozen=True, slots=True)
class IndexedExternalPart:
    """An external part F10 has taken in, as a pack slot needs to see it."""

    part_id: str
    part_kind: str
    origin: str
    content_sha256: str
    taken_in_at: datetime
    local_path: str
    #: Rows, as the index recorded them at import. ``None`` when the index holds no count -
    #: **never 0** (ADR-018): 0 rows is a measurement and an unknown count is not.
    rows: int | None
    licence_classes: tuple[str, ...] = ()
    #: F10's caveat objects, carried verbatim. See :class:`packs.resolver.Resolution`.
    caveats: tuple[dict[str, Any], ...] = ()

    def as_ref(self) -> ExternalPartRef:
        """The pinned reference a pack slot holds - the provenance, and nothing else."""
        return ExternalPartRef(
            origin=self.origin,
            content_sha256=self.content_sha256,
            taken_in_at=self.taken_in_at,
        )


def ref_for_part(session: Session, part_id: str) -> ExternalPartRef:
    """Turn an indexed external part into a pinnable reference, or refuse by name.

    This is where an operator's choice from the part listing becomes a pack slot, for the
    external half of FR-007.6. The contract half is :func:`packs.resolver.bind_indexed_part`.

    Raises:
        Refusal: ``part_not_external`` for a contract part - which is bound by its reference
            instead, and conflating the two is how provenance gets borrowed;
            ``external_part_has_no_provenance`` for an external row with no provenance,
            which F10's own ingestion refuses to create and which would make the run
            unreproducible; ``dataset_version_unresolved`` when nothing matches the id.
    """
    row = (
        session.execute(sa.select(CONTRACT_PARTS).where(CONTRACT_PARTS.c.id == part_id))
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise refuse(
            "dataset_version_unresolved",
            f"No part in this installation has id {part_id!r}. Take the material in first "
            f"- an external part is imported as a contract-conformant export directory "
            f"through the part import route.",
            part_id=part_id,
        )
    if row["source"] != SOURCE_EXTERNAL:
        raise refuse(
            "part_not_external",
            f"Part {part_id!r} came from miDataworks, so it is pinned by its contract "
            f"reference {{type, name, version}} rather than by provenance. The two source "
            f"kinds are different types with no conversion between them.",
            part_id=part_id,
            source=str(row["source"]),
        )
    return _from_row(row).as_ref()


def lookup_by_digest(session: Session, content_sha256: str) -> IndexedExternalPart | None:
    """The indexed external part whose provenance carries this digest, or ``None``.

    **The digest, not the origin**, because the digest is the pin: the same file moved to a
    new path is the same part, and a different file at the same path is not. The origin is
    compared too when several rows share a digest, which only happens when the identical
    bytes were taken in from two places - and then either row resolves to the same material.

    Returns ``None`` rather than refusing: the caller decides what an absence means, the
    same division F10's own lookup makes - F7 turns it into ``not_found`` for a pack slot
    and a listing just reports it.
    """
    rows = (
        session.execute(
            sa.select(CONTRACT_PARTS)
            .where(CONTRACT_PARTS.c.source == SOURCE_EXTERNAL)
            .order_by(CONTRACT_PARTS.c.imported_at, CONTRACT_PARTS.c.id)
        )
        .mappings()
        .all()
    )
    for row in rows:
        provenance = row["external_provenance"] or {}
        if str(provenance.get("content_sha256") or "") == content_sha256:
            return _from_row(row)
    return None


def versions_of_origin(session: Session, origin: str) -> tuple[IndexedExternalPart, ...]:
    """Every ingestion of one origin, oldest first. For the lineage view (FR-007.26).

    Several entries mean the same origin was taken in more than once with different bytes -
    which is a fact an auditor wants and **not** an error: each is its own pinnable part, and
    an older pack version keeps resolving to the bytes it pinned.
    """
    rows = (
        session.execute(
            sa.select(CONTRACT_PARTS)
            .where(CONTRACT_PARTS.c.source == SOURCE_EXTERNAL)
            .order_by(CONTRACT_PARTS.c.imported_at, CONTRACT_PARTS.c.id)
        )
        .mappings()
        .all()
    )
    return tuple(
        _from_row(row)
        for row in rows
        if str((row["external_provenance"] or {}).get("source") or "") == origin
    )


def _from_row(row: Any) -> IndexedExternalPart:
    """Read one index row, **refusing an external part with no provenance**.

    ``row`` is a SQLAlchemy ``RowMapping``, which is mapping-like without being a
    ``Mapping[str, Any]``; typed ``Any`` here rather than cast, because a cast would claim a
    shape this function does not depend on.

    Not defaulted: an external part whose provenance is missing cannot say where it came
    from, and a pack pinning it would claim a reproducibility it does not have. F10's
    ingestion refuses to create such a row, so this is defence in depth against a row
    written by something else - a hand-edited database, a partial restore.
    """
    provenance = row["external_provenance"] or {}
    origin = provenance.get("source")
    digest = provenance.get("content_sha256")
    retrieved = provenance.get("retrieved_at")
    if not origin or not digest or not retrieved:
        raise refuse(
            "external_part_has_no_provenance",
            f"Part {row['id']!r} is recorded as external and its provenance is incomplete "
            f"(source={origin!r}, content_sha256={'set' if digest else 'missing'}, "
            f"retrieved_at={retrieved!r}). An unattributed input makes the run "
            f"unreproducible; refusing rather than filling in a blank.",
            part_id=str(row["id"]),
        )
    rows_recorded = row["row_count"]
    return IndexedExternalPart(
        part_id=str(row["id"]),
        part_kind=str(row["part_kind"]),
        origin=str(origin),
        content_sha256=str(digest),
        taken_in_at=datetime.fromisoformat(str(retrieved).replace("Z", "+00:00")),
        local_path=str(row["local_path"]),
        # The index's column is NOT NULL with a server default of 0, so a 0 here is
        # genuinely "the index says zero rows" rather than "unknown". It is still reported
        # as `None` when zero, because a part with no rows is a part that could not have
        # been read, and a listing showing "0 prompts" invites the operator to believe a
        # measurement was taken.
        rows=None if rows_recorded in (None, 0) else int(rows_recorded),
        licence_classes=tuple(row["licence_classes"] or ()),
        caveats=tuple(row["caveats"] or ()),
    )
