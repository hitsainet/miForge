"""The part index: rows derived from manifests, and the rebuild  (F10 tasks 5.5-5.7, 6.2).

**ADR-004 stated as code.** The authority for a part is the manifest in the part's own
directory, exactly as a checkpoint's authority is its ``manifest.json``.
:data:`services.parts.tables.CONTRACT_PARTS` and ``PART_SPLITS`` are a cache for listing,
and :func:`reindex_parts` rebuilds them **from manifests alone** - never from a previous
row and never from a sibling call. That is what makes the equivalence test meaningful: a
rebuild that read the old row would be comparing the index with itself.

**Idempotence is a database guarantee.** Re-importing the same
``(ref_type, ref_name, ref_version)`` returns the existing row, by inserting and catching
the partial unique index's violation. Not ``SELECT`` then ``INSERT``: that is a race, and
this runs in a worker.

**Where index and manifest disagree, the manifest wins and the disagreement is logged**
(foundation 4.5's rule, applied to parts). Logged rather than silently corrected because a
divergence means something wrote the index without going through the manifest, or edited a
manifest in place - both are bugs, and a silent self-heal is how they stay unfound.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.canonical import canonical_bytes
from core.errors import Refusal
from services.parts.contract_validator import (
    MANIFEST_FILENAME,
    DatasetVersion,
    parse_and_validate,
)
from services.parts.kind_map import ResolvedKind, resolve
from services.parts.tables import (
    CONTRACT_PARTS,
    PART_SPLITS,
    SOURCE_CONTRACT,
    SOURCE_EXTERNAL,
)

log = logging.getLogger("miforge.parts.index")

DIVERGENCE_EVENT = "index_manifest_divergence"


def part_id_for(version: DatasetVersion, manifest_sha256: str) -> str:
    """A stable identifier for a part.

    Derived from the contract reference when there is one, and from the manifest digest
    when there is not - which is the case for a conformant version carrying a null
    ``contract_reference``. **Not synthesised into a reference**: this is miForge's own
    local handle, it never appears as a ``{type, name, version}``, and the three reference
    columns stay null. The distinction is the whole of E-21.
    """
    reference = version.contract_reference
    if reference is not None:
        material = canonical_bytes(
            {"type": reference.type, "name": reference.name, "version": reference.version}
        )
    else:
        # The digest of the manifest, not of anything resembling an identity. A reader
        # cannot mistake this for a reference because it has no type, name or version.
        material = canonical_bytes({"manifest_sha256": manifest_sha256})
    return "part-" + hashlib.sha256(material).hexdigest()[:24]


# ====================================================================== row building


def part_row_values(
    version: DatasetVersion,
    kind: ResolvedKind,
    *,
    manifest_sha256: str,
    local_path: str,
    source: str = SOURCE_CONTRACT,
    external_provenance: Mapping[str, Any] | None = None,
    imported_at: datetime | None = None,
) -> dict[str, Any]:
    """The ``mf_contract_parts`` column values a validated manifest implies.

    **Extract, do not interpret.** Every value below is read out of the manifest; nothing
    is computed from a filename, a column list or a directory name.

    ``row_count`` sums the **training-eligible** splits only. A test set's rows are not
    training rows, and a single total that mixed them would be the number an operator
    reads when asking "how much am I training on."
    """
    reference = version.contract_reference
    if source == SOURCE_EXTERNAL and reference is not None:  # pragma: no cover - guard
        # Also enforced by a CHECK constraint; raised here so the refusal names R-02.40
        # rather than surfacing as an IntegrityError.
        from core.errors import refuse

        raise refuse(
            "external_part_carries_a_contract_reference",
            "An external part must not carry a miDataworks contract reference. "
            "miForge does not promote an external part to contract provenance (R-02.40).",
            source=source,
        )

    return {
        "id": part_id_for(version, manifest_sha256),
        "part_kind": kind.part_kind,
        "source": source,
        "ref_type": None if reference is None else reference.type,
        "ref_name": None if reference is None else reference.name,
        "ref_version": None if reference is None else reference.version,
        "dataset": version.dataset,
        "version_id": version.version_id,
        "version_number": version.version_number,
        "recipe_sha256": version.recipe_sha256,
        "content_kind": kind.content_kind,
        "config_sha256": None if version.config is None else version.config.sha256,
        "local_path": local_path,
        "manifest_sha256": manifest_sha256,
        "row_count": sum(s.rows for s in version.splits if s.training_eligible),
        "licence_classes": list(version.licence_classes),
        "licences": [dict(entry) for entry in version.licences],
        "caveats": [caveat.as_dict() for caveat in version.caveats],
        "external_provenance": None if external_provenance is None else dict(external_provenance),
        "imported_at": imported_at or datetime.now(tz=UTC),
    }


def split_row_values(version: DatasetVersion, part_id: str) -> list[dict[str, Any]]:
    """The ``mf_part_splits`` rows a validated manifest implies, verbatim."""
    return [
        {
            "part_id": part_id,
            "name": split.name,
            "path": split.path,
            "rows": split.rows,
            "bytes": split.bytes,
            "sha256": split.sha256,
            "logical_digest": split.logical_digest,
            "held_out": split.held_out,
            "evaluation_only": split.evaluation_only,
            "label_counts": dict(split.label_counts),
        }
        for split in version.splits
    ]


#: The columns :func:`resolve_part` compares against the manifest. Enumerated once, beside
#: the builder, because a divergence check that listed columns separately would drift the
#: first time a column was added and then stop checking it with nothing going red.
INDEXED_FIELDS: tuple[str, ...] = (
    "part_kind",
    "ref_type",
    "ref_name",
    "ref_version",
    "dataset",
    "version_id",
    "version_number",
    "recipe_sha256",
    "content_kind",
    "config_sha256",
    "manifest_sha256",
    "row_count",
    "licence_classes",
    "licences",
    "caveats",
)


# ====================================================================== writes


def upsert_part(
    session: Session,
    version: DatasetVersion,
    kind: ResolvedKind,
    *,
    manifest_sha256: str,
    local_path: str,
    source: str = SOURCE_CONTRACT,
    external_provenance: Mapping[str, Any] | None = None,
    imported_at: datetime | None = None,
) -> tuple[str, bool]:
    """Index a part, returning ``(part_id, created)``.

    Idempotent **through the database**: the insert is attempted and the partial unique
    index on the contract reference is allowed to raise, because a ``SELECT`` first is a
    race and this runs in a worker. A re-import of a version that is already indexed
    returns its row rather than creating a second one.

    Re-importing version 3 while version 4 exists is **two rows**, by design: ``ref_version``
    is part of the uniqueness key precisely so that a pack pinning v3 keeps meaning.
    """
    values = part_row_values(
        version,
        kind,
        manifest_sha256=manifest_sha256,
        local_path=local_path,
        source=source,
        external_provenance=external_provenance,
        imported_at=imported_at,
    )
    part_id = str(values["id"])

    savepoint = session.begin_nested()
    try:
        session.execute(sa.insert(CONTRACT_PARTS).values(**values))
        savepoint.commit()
    except IntegrityError:
        savepoint.rollback()
        existing = session.execute(
            sa.select(CONTRACT_PARTS.c.id).where(
                sa.and_(
                    CONTRACT_PARTS.c.source == source,
                    CONTRACT_PARTS.c.ref_type == values["ref_type"],
                    CONTRACT_PARTS.c.ref_name == values["ref_name"],
                    CONTRACT_PARTS.c.ref_version == values["ref_version"],
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            # The violation was something else - the primary key, i.e. the same manifest
            # digest - which is still an existing row for the same material.
            existing = session.execute(
                sa.select(CONTRACT_PARTS.c.id).where(CONTRACT_PARTS.c.id == part_id)
            ).scalar_one_or_none()
        if existing is None:  # pragma: no cover - no third way to violate
            raise
        log.info(
            "part_already_indexed",
            extra={"code": "part_already_indexed", "part_id": str(existing)},
        )
        return str(existing), False

    rows = split_row_values(version, part_id)
    if rows:
        session.execute(sa.insert(PART_SPLITS), rows)
    session.flush()
    return part_id, True


def delete_part_index(session: Session) -> None:
    """Empty the part index without rebuilding it.

    Exists so "drop the index without reindexing" is a one-liner a control can call. A
    hand-rolled DELETE in a test stops matching this module the first time a table is
    added - which is how a reindex control quietly stops testing anything.
    """
    session.execute(sa.delete(PART_SPLITS))
    session.execute(sa.delete(CONTRACT_PARTS))
    session.flush()


# ====================================================================== the rebuild


@dataclass
class PartReindexReport:
    """What a :func:`reindex_parts` run did. Returned so a command can say so."""

    parts: int = 0
    splits: int = 0
    #: ``(path, code, message)`` per directory that could not be indexed at all.
    refused: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.parts


def scan_parts_root(root: Path) -> Iterator[tuple[Path, bytes] | Refusal]:
    """Every part directory under ``root``, as ``(directory, manifest bytes)``.

    Filename-driven: a part directory is one containing
    ``midataworks-dataset-version.json``. **Not layout-driven** - no requirement fixes the
    arrangement below the root, so assuming one would make the rebuild silently skip any
    arrangement nobody wrote down.
    """
    if not root.is_dir():
        return
    for manifest in sorted(root.rglob(MANIFEST_FILENAME)):
        try:
            yield manifest.parent, manifest.read_bytes()
        except OSError as exc:
            from core.errors import refuse

            yield refuse(
                "dataset_version_manifest_unreadable",
                f"Cannot read {manifest}: {exc}",
                path=str(manifest),
            )


def reindex_parts(session: Session, *, root: Path) -> PartReindexReport:
    """Rebuild the part index from the manifests under ``root``.

    Destructive by design, like ``services.reindex.reindex``: the index is replaced by what
    the filesystem says, because a merge would let a row for a deleted part survive forever
    and the index would slowly stop being a picture of the disk.

    **Every row is derived from the manifest on disk**, validated again on the way in. A
    part whose manifest no longer conforms is refused and named rather than carried
    forward from its old row - carrying it forward is how an index outlives the contract.

    Args:
        session: An open session. Not committed here; the caller owns the transaction, so
            a rebuild that fails half way leaves the old index intact.
        root: The parts root (``settings.parts_root`` in production).
    """
    report = PartReindexReport()
    delete_part_index(session)

    for found in scan_parts_root(root):
        if isinstance(found, Refusal):
            report.refused.append((found.details.get("path", "?"), found.code, found.message))
            log.warning(
                "parts_reindex_refused",
                extra={"code": found.code, "detail": found.message, **found.details},
            )
            continue
        directory, manifest_bytes = found
        try:
            version, digest = parse_and_validate(manifest_bytes, source=str(directory))
            kind = resolve(version)
            source = _recorded_source(directory)
            recorded_at = _recorded_imported_at(directory)
            if recorded_at is None:
                log.warning(
                    "part_import_time_unrecoverable",
                    extra={
                        "code": "part_import_time_unrecoverable",
                        "path": str(directory),
                        "resolution": "indexed_with_rebuild_time",
                    },
                )
            _, created = upsert_part(
                session,
                version,
                kind,
                manifest_sha256=digest,
                local_path=str(directory),
                source=source,
                external_provenance=_recorded_provenance(directory),
                imported_at=recorded_at,
            )
        except Refusal as refusal:
            report.refused.append((str(directory), refusal.code, refusal.message))
            log.warning(
                "parts_reindex_refused",
                extra={"code": refusal.code, "detail": refusal.message, **refusal.details},
            )
            continue
        if created:
            report.parts += 1
            report.splits += len(version.splits)

    session.flush()
    log.info(
        "parts_reindex_complete",
        extra={
            "root": str(root),
            "parts": report.parts,
            "splits": report.splits,
            "refused": len(report.refused),
        },
    )
    return report


#: The sidecar that records what the manifest cannot: whether miForge took this part in
#: from miDataworks' contract channel or from the operator's own files, an external part's
#: own provenance (R-02.40), and **when miForge took it in**.
#:
#: **Why a sidecar and not a column only.** ADR-004 makes the index rebuildable *from
#: disk*, and none of those three is a manifest field. A rebuild with no sidecar would have
#: to guess, and the cheap guesses are each wrong in a specific way:
#:
#: * ``source`` - "it has a contract reference, so it came from miDataworks" is exactly the
#:   promotion R-02.40 forbids;
#: * ``imported_at`` - the directory's mtime is **the copy's time, not the import's**, which
#:   is the same reason ``services/reindex.py`` refuses to take a checkpoint's ``created_at``
#:   from the filesystem (*"a time must not be invented from the file's mtime"*).
#:
#: So all three are written beside the manifest at import. A part with no sidecar rebuilds
#: as ``external_files`` with no reference and an ``imported_at`` of *now*, logged as an
#: anomaly - the honest reading of "miForge cannot say when this arrived", rather than a
#: fabricated timestamp presented as a record.
SOURCE_SIDECAR = ".miforge-part-source.json"


def write_source_sidecar(
    directory: Path,
    *,
    source: str,
    external_provenance: Mapping[str, Any] | None = None,
    imported_at: datetime,
) -> None:
    """Record what the manifest cannot, beside the manifest. See :data:`SOURCE_SIDECAR`."""
    payload: dict[str, Any] = {"source": source, "imported_at": imported_at.isoformat()}
    if external_provenance is not None:
        payload["external_provenance"] = dict(external_provenance)
    (directory / SOURCE_SIDECAR).write_bytes(canonical_bytes(payload) + b"\n")


def _sidecar(directory: Path) -> Mapping[str, Any]:
    path = directory / SOURCE_SIDECAR
    if not path.is_file():
        return {}
    try:
        parsed = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning(
            "part_source_sidecar_unreadable",
            extra={"code": "part_source_sidecar_unreadable", "path": str(path)},
        )
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _recorded_source(directory: Path) -> str:
    recorded = _sidecar(directory).get("source")
    return str(recorded) if recorded in {SOURCE_CONTRACT, SOURCE_EXTERNAL} else SOURCE_EXTERNAL


def _recorded_provenance(directory: Path) -> Mapping[str, Any] | None:
    value = _sidecar(directory).get("external_provenance")
    return dict(value) if isinstance(value, Mapping) else None


def _recorded_imported_at(directory: Path) -> datetime | None:
    """When miForge took this part in, from the sidecar - or ``None``, said out loud.

    ``None`` rather than the directory's mtime: that is the copy's time and not the
    import's, and a plausible-looking wrong timestamp is worse than an admitted gap,
    because nothing downstream can tell it was invented.
    """
    value = _sidecar(directory).get("imported_at")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        log.warning(
            "part_source_sidecar_timestamp_unparseable",
            extra={"code": "part_source_sidecar_timestamp_unparseable", "path": str(directory)},
        )
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# ====================================================================== the read seam


def part_payload(row: Mapping[str, Any], splits: list[Mapping[str, Any]]) -> dict[str, Any]:
    """One part as the API serves it. **The seam the equivalence test asserts over.**

    Everything F9 needs to render a part row is here, in one object: a second call to
    render a row would mean the shape is wrong (FR-010.19's note, and the F10 FTID §6).

    ``caveats`` and ``licence_classes`` are **on the summary**, not only on a detail view.
    A caveat must be visible at the point of use, not only at import - that is FR-010.17,
    and it exists because a dropped licence provenance is one of this project's recorded
    defects.
    """
    reference = None
    if row["ref_type"] is not None:
        reference = {
            "type": row["ref_type"],
            "name": row["ref_name"],
            "version": row["ref_version"],
        }
    return {
        "id": row["id"],
        "part_kind": row["part_kind"],
        "source": row["source"],
        # `null` when the version carries none, and never synthesised. F7 refuses to bind
        # such a part to a pack slot (`part_unpinnable`); F10's job is to surface the null
        # faithfully.
        "contract_reference": reference,
        "rendered": (
            None
            if reference is None
            else f"{reference['type']}/{reference['name']}@{reference['version']}"
        ),
        "dataset": row["dataset"],
        "version_id": row["version_id"],
        "version_number": row["version_number"],
        "recipe_sha256": row["recipe_sha256"],
        "content_kind": row["content_kind"],
        "config_sha256": row["config_sha256"],
        "manifest_sha256": row["manifest_sha256"],
        "row_count": row["row_count"],
        "licence_classes": list(row["licence_classes"] or []),
        "licences": list(row["licences"] or []),
        "caveats": list(row["caveats"] or []),
        "external_provenance": row["external_provenance"],
        "local_path": row["local_path"],
        "imported_at": _isoformat(row["imported_at"]),
        "splits": [
            {
                "name": split["name"],
                "path": split["path"],
                "rows": split["rows"],
                "bytes": split["bytes"],
                "sha256": split["sha256"],
                "logical_digest": split["logical_digest"],
                "held_out": bool(split["held_out"]),
                "evaluation_only": bool(split["evaluation_only"]),
                "label_counts": split["label_counts"] or {},
                # Derived once, here, so a caller never re-derives the rule. The rule
                # itself lives in `services.parts.reader` and nowhere else.
                "training_eligible": not (
                    bool(split["held_out"]) or bool(split["evaluation_only"])
                ),
            }
            for split in splits
        ],
    }


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=UTC)).isoformat()
    return str(value)


def _splits_of(session: Session, part_ids: list[str]) -> dict[str, list[Mapping[str, Any]]]:
    if not part_ids:
        return {}
    rows = session.execute(
        sa.select(PART_SPLITS)
        .where(PART_SPLITS.c.part_id.in_(part_ids))
        .order_by(PART_SPLITS.c.part_id, PART_SPLITS.c.name)
    ).all()
    out: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        mapping = dict(row._mapping)
        out.setdefault(str(mapping["part_id"]), []).append(mapping)
    return out


def list_parts(session: Session, *, part_kind: str | None = None) -> list[dict[str, Any]]:
    """Every indexed part, ordered by id so the answer is stable.

    Ordered deliberately: the equivalence test compares lists, and an unordered query makes
    it pass or fail on PostgreSQL's physical row order rather than on whether the rebuild
    was faithful.
    """
    stmt = sa.select(CONTRACT_PARTS).order_by(CONTRACT_PARTS.c.id)
    if part_kind is not None:
        stmt = stmt.where(CONTRACT_PARTS.c.part_kind == part_kind)
    rows = [dict(row._mapping) for row in session.execute(stmt).all()]
    splits = _splits_of(session, [str(row["id"]) for row in rows])
    return [part_payload(row, splits.get(str(row["id"]), [])) for row in rows]


def get_part(session: Session, part_id: str) -> dict[str, Any]:
    """One part, or refuse by name.

    Raises:
        Refusal: ``part_not_found``. Named rather than an empty 404 body, because "no such
            part" and "this route does not exist" have different fixes.
    """
    from core.errors import refuse

    row = session.execute(
        sa.select(CONTRACT_PARTS).where(CONTRACT_PARTS.c.id == part_id)
    ).one_or_none()
    if row is None:
        raise refuse(
            "part_not_found",
            f"No part with id {part_id!r} in the index. If it exists on disk, reindex.",
            part_id=part_id,
        )
    mapping = dict(row._mapping)
    return part_payload(mapping, _splits_of(session, [part_id]).get(part_id, []))


def parts_snapshot(session: Session) -> dict[str, Any]:
    """Everything the parts equivalence test compares, in one call.

    One function so the test cannot accidentally compare a subset: adding a query to the
    read seam and forgetting to add it here is the construction-green failure this project
    keeps finding.
    """
    return {
        "parts": list_parts(session),
        "by_kind": {
            kind: list_parts(session, part_kind=kind)
            for kind in sorted(
                {
                    str(value)
                    for value in session.execute(
                        sa.select(CONTRACT_PARTS.c.part_kind).distinct()
                    ).scalars()
                }
            )
        },
    }


def resolve_part(session: Session, part_id: str) -> dict[str, Any]:
    """Read one part, **letting the manifest win** over the index.

    The index is a cache. When it disagrees with the manifest in the part's directory, the
    manifest is right by definition (ADR-004), the disagreement is logged as
    ``index_manifest_divergence`` with the fields and both answers, and the row is
    corrected so the next read is cheap and the same line is not emitted forever.

    Raises:
        Refusal: ``part_not_found`` when the id is unknown;
            ``manifest_missing_for_indexed_part`` when the row's directory no longer
            carries a manifest - the index silently outliving the artifact is the exact
            inversion of ADR-004, so it is refused rather than served.
    """
    from core.errors import refuse

    row = session.execute(
        sa.select(CONTRACT_PARTS).where(CONTRACT_PARTS.c.id == part_id)
    ).one_or_none()
    if row is None:
        raise refuse(
            "part_not_found",
            f"No part with id {part_id!r} in the index.",
            part_id=part_id,
        )
    current = dict(row._mapping)
    manifest_path = Path(str(current["local_path"])) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        log.warning(
            DIVERGENCE_EVENT,
            extra={
                "artifact": part_id,
                "divergence": "manifest_missing",
                "manifest_path": str(manifest_path),
            },
        )
        raise refuse(
            "manifest_missing_for_indexed_part",
            f"{part_id} is indexed but {manifest_path} does not exist. The manifest is "
            f"the record, so this row is stale; reindex to remove it.",
            part_id=part_id,
            manifest_path=str(manifest_path),
        )

    version, digest = parse_and_validate(manifest_path.read_bytes(), source=str(manifest_path))
    truth = part_row_values(
        version,
        resolve(version),
        manifest_sha256=digest,
        local_path=str(current["local_path"]),
        source=str(current["source"]),
        external_provenance=current["external_provenance"],
    )

    diffs: dict[str, dict[str, Any]] = {}
    for name in INDEXED_FIELDS:
        if canonical_bytes(current.get(name)) != canonical_bytes(truth.get(name)):
            diffs[name] = {"index": current.get(name), "manifest": truth.get(name)}

    if diffs:
        log.warning(
            DIVERGENCE_EVENT,
            extra={
                "artifact": part_id,
                "divergence": "field_mismatch",
                "divergent_fields": sorted(diffs),
                "resolution": "manifest_wins",
                "detail": diffs,
            },
        )
        session.execute(
            sa.update(CONTRACT_PARTS)
            .where(CONTRACT_PARTS.c.id == part_id)
            .values(**{name: answers["manifest"] for name, answers in diffs.items()})
        )
        session.flush()
        current.update({name: answers["manifest"] for name, answers in diffs.items()})

    return part_payload(current, _splits_of(session, [part_id]).get(part_id, []))
