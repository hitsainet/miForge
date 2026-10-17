"""Taking a part in: validate, verify, place atomically, index  (F10 tasks 4.4-4.9).

**The order is the requirement.**

.. code-block:: text

    stage -> validate -> map -> verify material -> ATOMIC PLACE -> index -> summary

**Index after the rename, never before.** The manifest landing in its final directory is
the commit point - ADR-004's authority rule applied to an *input* rather than an output. A
crash before the rename leaves a temp directory and no part; a crash after it leaves a part
the next ``reindex`` finds. There is no intermediate state an API can observe, which is why
a part has no "importing" status column: it is imported or it is not.

**The atomic writer is the shared one** (:func:`core.atomic_dir.atomic_dir`, ADR-015). Two
implementations of that guarantee are two chances to get ``fsync`` wrong, which is why F5's
checkpoint writer, F1's trainer-state writer, F8's adapter writer and this all call one
function.

## Hostile input has exactly one plausible mechanism here

Every path the manifest supplies - ``content.splits[].path``, ``content.config.path`` - is
resolved and **confined** under the staged directory. ``../``, an absolute path and a
symlink pointing out are each refused. Resolution and containment, not string inspection:
a check for the substring ``..`` passes a symlink and fails a legitimate filename.

## Digests are verified, and the contract's are recorded

Verification and recording are different acts (FR-010.14). miForge streams each split and
compares it against the ``sha256`` **the contract carries**; what it *records* is the
contract's digest, not a second competing hash of the same bytes. ``logical_digest`` is
miDataworks' row-level digest and depends on its key scheme - recorded, **never
recomputed**, because recomputing it would mean reimplementing someone else's key scheme.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from core.atomic_dir import atomic_dir
from core.errors import refuse
from services.parts.contract_validator import (
    MANIFEST_FILENAME,
    DatasetVersion,
    parse_and_validate,
)
from services.parts.index import upsert_part, write_source_sidecar
from services.parts.kind_map import ResolvedKind, resolve
from services.parts.tables import SOURCE_CONTRACT, SOURCE_EXTERNAL

log = logging.getLogger("miforge.parts.ingest")

#: Bytes per read when digesting a split. Streamed because a pretraining-scale corpus will
#: be larger than RAM and OQ-5 leaves that volume open: ``path.read_bytes()`` on a split is
#: an OOM waiting for the first large corpus.
DIGEST_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class IngestResult:
    """What an import produced. Returned rather than logged away."""

    part_id: str
    created: bool
    part_kind: str
    local_path: str
    manifest_sha256: str
    #: Carried onto the result as well as into the row, because FR-010.17 requires a
    #: caveat be visible **at the point of use** and an import's result is the first such
    #: point. Dropping them here is one of the mutations §10 requires to go red.
    caveats: tuple[Mapping[str, Any], ...]
    licence_classes: tuple[str, ...]
    splits: tuple[str, ...]


def stream_sha256(path: Path, *, chunk: int = DIGEST_CHUNK_BYTES) -> str:
    """Streaming SHA-256 of a file's bytes.

    A local helper rather than :func:`core.canonical.sha256_of_file` only in signature -
    it delegates, so there is still one implementation. Kept as a name here because this
    package's digests are of *received* files rather than of written ones, and a reader
    tracing FR-010.14 should find the call under this package.
    """
    from core.canonical import sha256_of_file

    return sha256_of_file(path, chunk=chunk)


def confine(base: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``base``, refusing an escape.

    Args:
        base: The staged part directory.
        relative: A path exactly as the manifest gives it.

    Returns:
        The resolved path.

    Raises:
        Refusal: ``dataset_version_path_escapes_part`` for ``../``, an absolute path or a
            symlink whose target is outside ``base``.

    Symlinks are **followed** during resolution on purpose: a check that resolved without
    following would pass a symlink pointing at ``/etc/passwd``, and that is the only
    plausible attack this feature has.
    """
    root = base.resolve()
    candidate = (root / relative).resolve()
    if candidate == root or root not in candidate.parents:
        raise refuse(
            "dataset_version_path_escapes_part",
            f"The manifest names the path {relative!r}, which resolves to {candidate} - "
            f"outside the part's own directory {root}. Refused: a manifest is untrusted "
            f"data and may not reach outside the files it ships with.",
            path=relative,
            resolved=str(candidate),
            part_directory=str(root),
        )
    return candidate


def verify_material(staged: Path, version: DatasetVersion, kind: ResolvedKind) -> None:
    """Every file the manifest names is present and matches the digest it carries.

    Run **before** anything is indexed and before any reader can see the part. A part that
    exists and is unverified is a part something will eventually read.

    Raises:
        Refusal: ``dataset_version_split_missing`` naming path and split;
            ``dataset_version_digest_mismatch`` naming split, expected and found;
            ``dataset_version_config_missing`` / ``_digest_mismatch`` for a config part.
    """
    for split in version.splits:
        path = confine(staged, split.path)
        if not path.is_file():
            raise refuse(
                "dataset_version_split_missing",
                f"The manifest names split {split.name!r} at {split.path!r}, and that "
                f"file is not in the export. Nothing is imported: a part missing a split "
                f"it declares would read as complete.",
                split=split.name,
                path=split.path,
            )
        found = stream_sha256(path)
        if found != split.sha256:
            raise refuse(
                "dataset_version_digest_mismatch",
                f"Split {split.name!r} does not match the digest the contract carries. "
                f"Expected {split.sha256}, found {found}. The bytes changed after "
                f"miDataworks published them, so the part is refused and never used.",
                split=split.name,
                path=split.path,
                expected=split.sha256,
                found=found,
            )

    config = version.config
    if config is not None:
        path = confine(staged, config.path)
        if not path.is_file():
            raise refuse(
                "dataset_version_config_missing",
                f"The manifest names a configuration body at {config.path!r} and that "
                f"file is not in the export.",
                path=config.path,
                config_name=config.name,
            )
        found = hashlib.sha256(path.read_bytes()).hexdigest()
        if found != config.sha256:
            raise refuse(
                "dataset_version_digest_mismatch",
                f"The configuration body {config.name!r} does not match the digest the "
                f"contract carries. Expected {config.sha256}, found {found}. This hash is "
                f"what R-02.10 records as the selector or grader identity, so a mismatch "
                f"would make a checkpoint cite a configuration it was not trained with.",
                config_name=config.name,
                path=config.path,
                expected=config.sha256,
                found=found,
            )
    elif kind.is_config:  # pragma: no cover - the schema's discriminator guarantees this
        raise refuse(
            "dataset_version_invalid",
            "A config target carries no configuration body.",
            target_kind=kind.target_kind,
        )


def read_manifest(staged: Path) -> bytes:
    """The staged manifest's bytes, or refuse naming the filename expected.

    Raises:
        Refusal: ``dataset_version_manifest_missing``. Names the file, because the single
            most common way this fails is an operator pointing at the export's parent
            directory.
    """
    path = staged / MANIFEST_FILENAME
    if not path.is_file():
        raise refuse(
            "dataset_version_manifest_missing",
            f"{staged} contains no {MANIFEST_FILENAME}. A miDataworks export carries its "
            f"handoff manifest under that exact name - note there is no `-v1` in it; the "
            f"`-v1` file is the schema, not the instance. Point at the export directory "
            f"itself.",
            expected_filename=MANIFEST_FILENAME,
            directory=str(staged),
        )
    return path.read_bytes()


def ingest(
    session: Session,
    staged: Path,
    *,
    parts_root: Path,
    source: str = SOURCE_CONTRACT,
    external_provenance: Mapping[str, Any] | None = None,
    origin: str | None = None,
) -> IngestResult:
    """Take a staged export directory in as a part.

    Args:
        session: An open session. The caller owns the transaction.
        staged: A directory holding ``midataworks-dataset-version.json`` and the files it
            names. It is **never written to** - a copy is made into the parts root.
        parts_root: Where parts live (``settings.parts_root``).
        source: ``midataworks_contract`` or ``external_files``. An external part's
            contract-reference columns stay **null** (R-02.40), enforced by a CHECK
            constraint as well as here.
        external_provenance: The gate contract's ``ExternalProvenance`` object for an
            external part - ``{source, content_sha256, retrieved_at}`` plus an optional
            ``licence``. Required for an external part: *a part with neither a reference
            nor provenance is refused, because an unattributed training input makes the
            run unreproducible.*
        origin: What to name in a refusal - an export id, a Hub repo, a directory.

    Returns:
        The :class:`IngestResult`, carrying the caveats and licence classes.

    Raises:
        Refusal: any of this package's named refusals. Nothing is indexed unless the
            whole sequence succeeded.
    """
    where = origin or str(staged)

    if source not in {SOURCE_CONTRACT, SOURCE_EXTERNAL}:
        raise refuse(
            "part_source_unknown",
            f"source={source!r} is not one miForge records. It is "
            f"{SOURCE_CONTRACT!r} or {SOURCE_EXTERNAL!r}; nothing is guessed.",
            source=source,
        )
    if source == SOURCE_EXTERNAL and not external_provenance:
        raise refuse(
            "external_part_has_no_provenance",
            "An external part must carry its own provenance - a source, a content hash "
            "and when it was retrieved. A part with neither a contract reference nor "
            "provenance is unattributed, and an unattributed training input makes the run "
            "unreproducible: nothing downstream could say where a behaviour came from.",
            source=source,
        )

    manifest_bytes = read_manifest(staged)
    version, manifest_sha256 = parse_and_validate(manifest_bytes, source=where)
    kind = resolve(version)
    verify_material(staged, version, kind)

    from services.parts.index import part_id_for

    part_id = part_id_for(version, manifest_sha256)
    destination = parts_root / part_id

    # One timestamp for the row AND the sidecar, so the index and the thing it is rebuilt
    # from cannot disagree about when this part arrived.
    imported_at = datetime.now(tz=UTC)

    if destination.exists():
        # Already placed. The index may or may not know about it - it is a cache - so the
        # row is (re-)asserted and the directory left exactly as it is. Re-copying would
        # rewrite bytes whose digest is the part's identity.
        resolved_id, created = upsert_part(
            session,
            version,
            kind,
            manifest_sha256=manifest_sha256,
            local_path=str(destination),
            source=source,
            external_provenance=external_provenance,
            imported_at=imported_at,
        )
        return _result(resolved_id, created, kind, destination, manifest_sha256, version)

    with atomic_dir(destination, latest_link=False) as tmp:
        # `latest_link=False`: a parts root carries no `latest` pointer. A part is
        # addressed by its identity, and a "latest part" would be a second answer to
        # "which version is pinned" - which is exactly what a pack's reference is for.
        for entry in sorted(staged.iterdir()):
            target = tmp / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target, symlinks=False)
            else:
                shutil.copy2(entry, target)
        write_source_sidecar(
            tmp,
            source=source,
            external_provenance=external_provenance,
            imported_at=imported_at,
        )

    resolved_id, created = upsert_part(
        session,
        version,
        kind,
        manifest_sha256=manifest_sha256,
        local_path=str(destination),
        source=source,
        external_provenance=external_provenance,
        imported_at=imported_at,
    )
    log.info(
        "part_imported",
        extra={
            "code": "part_imported",
            "part_id": resolved_id,
            "part_kind": kind.part_kind,
            "contract_reference": (
                None if version.contract_reference is None else version.contract_reference.render()
            ),
            "source": source,
            "origin": where,
        },
    )
    return _result(resolved_id, created, kind, destination, manifest_sha256, version)


def _result(
    part_id: str,
    created: bool,
    kind: ResolvedKind,
    destination: Path,
    manifest_sha256: str,
    version: DatasetVersion,
) -> IngestResult:
    return IngestResult(
        part_id=part_id,
        created=created,
        part_kind=kind.part_kind,
        local_path=str(destination),
        manifest_sha256=manifest_sha256,
        caveats=tuple(caveat.as_dict() for caveat in version.caveats),
        licence_classes=version.licence_classes,
        splits=tuple(split.name for split in version.splits),
    )
