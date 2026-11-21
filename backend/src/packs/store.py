"""The pack store on disk  (FR-007.2, FR-007.14, FR-007.18; ADR-004, ADR-005).

**A pack version is a file, and the file is the truth.** That is the gate contract's own
decision, stated in ``pack-definition-v1.json``'s description: *"THIS IS AN ON-DISK FILE,
'pack.json' ... packs were the ONE artifact with no on-disk manifest, so losing the database
lost every unexported pack and ``reindex`` could not rebuild them."* A pack is references,
not data - a few kilobytes - so paying one small writer keeps ADR-004 universal instead of
carving out an exception every future reader has to remember.

**This corrects F7's own documents.** The FPRD §5.2, the FTDD §4.4 and the FTID all say
packs are *database-authoritative*, that *"``reindex`` cannot reconstruct a pack"*, and that
a pack assertion *"must not be added to the reindex-equivalence test"*. Each was true when
written and the gate contract settled it the other way afterwards; the finding is reported
rather than split between two beliefs. With a pack on disk, the database is what it is
everywhere else in miForge: **a rebuildable index** (:func:`packs.registry.rebuild_index`).

## Layout, and why each piece of it

``<packs_root>/<name>/<version>/pack.json``     one committed pack version
``<packs_root>/.drafts/<draft_id>.json``        an unfinished authoring flow (FR-007.18)
``<packs_root>/.external/<digest>.json``        the external-part registry (:mod:`packs.external`)

* **The version is a directory, not a filename**, so a later decision to put anything beside
  a pack - an export, a dry-run transcript - needs no move.
* **A committed version's file is never rewritten.** :func:`write_committed` refuses an
  existing path with ``pack_version_committed``: editing produces a **new version**, because
  a pinned input that can change is not a pinned input (FR-007.2, E10).
* **Every write goes through** :func:`core.canonical.write_canonical`, which is both the
  hashing path and the writing path (ADR-005) and is atomic within the file - temp file,
  ``fsync``, ``rename``. A half-written ``pack.json`` that still parses is worse than one
  that is absent, because the absent one is obviously absent.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from core.canonical import write_canonical
from core.errors import refuse
from packs.definition import PackVersionDefinition
from packs.errors import PACK_VERSION_COMMITTED, PackRefusal
from packs.settings import get_pack_settings

PACK_FILENAME = "pack.json"
DRAFTS_DIRNAME = ".drafts"

#: Directories that are bookkeeping rather than packs. Listed once so a listing and a
#: rebuild cannot disagree about what counts as a pack - the dot prefix is the convention and
#: this is the check.
RESERVED_DIRNAMES: frozenset[str] = frozenset({DRAFTS_DIRNAME, ".external"})


def packs_root() -> Path:
    """The configured pack root. **Read-only: this does not create it.**

    *It used to create it on demand, and that was right for a writer and wrong everywhere
    else.* The creation sat on every path, so a read of the registry on an installation where
    the root is not writable raised ``PermissionError`` — an ``OSError`` that no caller
    handles — rather than answering "no packs". It has cost this project twice already:

    * it **aborted the whole reindex**, the one-subsystem-kills-everything failure that module
      guards against run-by-run elsewhere (AGENT.md, 2026-10-14);
    * and ``GET /api/v1/packs`` answered **500 `PermissionError: /data/packs`**, so the pack
      registry screen was down on exactly the installation this product ships as — the
      container, uid 10001 against a root-owned ``/data``. Reproduced 2026-10-14 by an agent
      sweeping F9's screens, which is the second victim of one line.

    Use :func:`ensure_packs_root` on a path that is about to write. A reader that finds no
    directory has found no packs, which is an answer; a reader that creates one has changed
    the system in order to report on it.
    """
    return get_pack_settings().packs_root


def ensure_packs_root() -> Path:
    """The configured pack root, **created** — for callers that are about to write.

    Separated from :func:`packs_root` so the creation is a decision at the call site rather
    than a property of asking where the root is.
    """
    root = get_pack_settings().packs_root
    root.mkdir(parents=True, exist_ok=True)
    return root


def pack_dir(name: str, version: str) -> Path:
    """``<packs_root>/<name>/<version>``.

    ``name`` is slug-shaped by the schema's pattern and ``version`` is a non-empty string,
    both validated by :class:`packs.definition.PackVersionDefinition` before anything
    reaches here - so neither can contain a path separator. The containment check below is
    kept anyway, because "validated upstream" is a claim about another function.
    """
    root = packs_root()
    candidate = (root / name / version).resolve()
    if root.resolve() not in candidate.parents:
        # Not one of the nine taxonomy codes, deliberately: this is not a refusal an
        # operator can act on, it is a guard against an identity that should have been
        # rejected by the schema's pattern two layers up.
        raise refuse(
            "pack_identity_invalid",
            f"{name}@{version} does not name a path inside the pack root. A pack name is "
            f"slug-shaped and a version is a plain string; neither may traverse.",
            pack_id=name,
            version=version,
        )
    return candidate


def pack_file(name: str, version: str) -> Path:
    return pack_dir(name, version) / PACK_FILENAME


def exists(name: str, version: str) -> bool:
    return pack_file(name, version).is_file()


def write_committed(definition: PackVersionDefinition) -> Path:
    """Write a committed pack version. **Refuses to overwrite one** (FR-007.2, E10).

    Returns:
        The path written.

    Raises:
        PackRefusal: ``pack_version_committed`` when the version is already on disk. The
            refusal says what to do instead - commit a new version - because that is the
            whole mechanism by which *"a pack version names a reproducible run input"* stays
            true.
    """
    path = pack_file(definition.name, definition.version)
    if path.is_file():
        raise PackRefusal(
            PACK_VERSION_COMMITTED,
            f"Pack version {definition.name}@{definition.version} is committed and its "
            f"slots are frozen. Commit a new version instead: a pinned input that can "
            f"change is not a pinned input, and every run that cited this version would "
            f"silently come to mean something else.",
            pack_id=definition.name,
            version=definition.version,
        )
    # The writer creates what it writes into. `packs_root()` deliberately does not create
    # the root any more - it is a read accessor - so the creation is here, at the one call
    # site that is actually about to write a file.
    path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical(path, definition.document())
    return path


def read(name: str, version: str) -> PackVersionDefinition:
    """Read one committed pack version, **validated against the schema on the way in**.

    Raises:
        PackRefusal: ``pack_document_invalid`` for a file that is not a conforming pack,
            whose digest does not match, or whose ``schema_version`` is not this one -
            which is also what a tampered or half-restored file looks like.
        Refusal: ``pack_not_found`` when there is no such version.
    """
    path = pack_file(name, version)
    if not path.is_file():
        raise refuse(
            "pack_not_found",
            f"No pack version {name}@{version} is in the pack store at {packs_root()}. "
            f"Import it, or list the packs that are here.",
            pack_id=name,
            version=version,
        )
    return PackVersionDefinition.from_document(_load(path))


def iter_versions() -> Iterator[PackVersionDefinition]:
    """Every committed pack version on disk, pack by pack, version by version.

    The source of a rebuild. A directory that is not a pack - no ``pack.json`` - is skipped
    silently; a ``pack.json`` that does not conform **refuses**, because a listing that
    quietly omitted a malformed pack would hide the one thing the operator needs to know.
    """
    root = packs_root()
    if not root.is_dir():
        # An absent store holds no packs, and saying so is the answer. Creating it here
        # would make a READ change the filesystem - and on a root-owned `/data` it raises
        # `PermissionError` instead, which is how this listing used to 500.
        return
    for pack in sorted(item for item in root.iterdir() if item.is_dir()):
        if pack.name in RESERVED_DIRNAMES:
            continue
        for version in sorted(item for item in pack.iterdir() if item.is_dir()):
            file = version / PACK_FILENAME
            if file.is_file():
                yield PackVersionDefinition.from_document(_load(file))


def delete(name: str, version: str) -> None:
    """Remove one pack version's directory. Callers check usage first (E11).

    Deleting a pack deletes **no part**: miForge does not own them, and an external part's
    files stay exactly where the operator put them - which the API's response says, because
    an operator who deleted a pack should not have to wonder whether their data went too.
    """
    directory = pack_dir(name, version)
    if not directory.is_dir():
        return
    for item in sorted(directory.rglob("*"), reverse=True):
        if item.is_file():
            item.unlink()
        else:
            item.rmdir()
    directory.rmdir()
    parent = directory.parent
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


# ------------------------------------------------------------------------------- drafts


def drafts_root() -> Path:
    root = ensure_packs_root() / DRAFTS_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def draft_file(draft_id: str) -> Path:
    """One draft's file. The id is miForge-issued (a uuid4 hex), so it cannot traverse."""
    return drafts_root() / f"{draft_id}.json"


def write_draft(draft_id: str, payload: Mapping[str, Any]) -> Path:
    """Persist a draft, atomically.

    Called after **every** step (FR-007.18). The authoring screen's subtitle says *"progress
    is saved as you go"*, and a save that is claimed must have happened - so the write is
    synchronous and atomic rather than deferred to a commit.
    """
    return _write(draft_file(draft_id), payload)


def read_draft(draft_id: str) -> dict[str, Any] | None:
    path = draft_file(draft_id)
    return None if not path.is_file() else dict(_load(path))


def delete_draft(draft_id: str) -> None:
    draft_file(draft_id).unlink(missing_ok=True)


def iter_drafts() -> Iterator[dict[str, Any]]:
    """Every draft. **Drafts are not packs** and never appear in the registry listing (E13)."""
    root = packs_root() / DRAFTS_DIRNAME
    if not root.is_dir():
        return
    for file in sorted(root.glob("*.json")):
        yield dict(_load(file))


def _write(path: Path, payload: Mapping[str, Any]) -> Path:
    write_canonical(path, dict(payload))
    return path


def _load(path: Path) -> Mapping[str, Any]:
    document: Any = json.loads(path.read_text("utf-8"))
    if not isinstance(document, dict):
        raise refuse(
            "pack_store_corrupt",
            f"{path} does not hold a JSON object. The pack store is the authority for a "
            f"pack and this file cannot be read as one; refusing rather than "
            f"treating it as empty.",
            path=str(path),
        )
    return document
