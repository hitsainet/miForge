"""``reindex``, and the queries it is proved against  (tasks 4.3, 4.4, 4.5; ADR-004).

**ADR-004, stated as code.** The on-disk manifest is authoritative; PostgreSQL holds a
rebuildable index. Three consequences live in this module and each is a test rather than a
claim:

1. :func:`reindex` walks the checkpoint root and reconstructs **every** artifact index row
   **from manifests alone**. It is a first-class command, not a recovery script.
2. :func:`list_checkpoints`, :func:`list_adapters` and :func:`checkpoint_tree` are the
   read seam. Task 4.4 seeds, drops the index, reindexes, and asserts these answer
   **identically** - which is what makes ADR-004's claim true rather than asserted.
3. :func:`resolve_checkpoint` is the disagreement path (task 4.5). **Where the index and
   the manifest disagree the manifest wins, and the disagreement is logged** - and the row
   is corrected in place, so the same divergence is not re-reported forever.

**What this module deliberately does NOT rebuild.** ``mf_runs``, ``mf_steps`` and
``mf_leases`` hold state no artifact carries: a run's status, a lease's expiry. Those stay
record tables.

**Packs are no longer the exception, and that is a reversal recorded rather than quietly
applied.** This paragraph said, correctly when written, that *"a pack has no on-disk manifest
at all (ADR-003 puts it in PostgreSQL while ADR-004 makes the filesystem authoritative), so
losing the database loses every unexported pack"*, and concluded that **a pack assertion must
not be added to task 4.4's equivalence test** until someone decided whether packs get a disk
mirror. That was the 2026-10-13 review's finding 10.

**The decision was taken by the gate contract.** ``pack-definition-v1.json``'s own
description reads *"THIS IS AN ON-DISK FILE, 'pack.json', beside the pack's directory - not a
database-only record"*, and F7 implemented it: a committed version is
``<packs_root>/<name>/<version>/pack.json``, written atomically through
``core.canonical.write_canonical``, and ``packs.registry.rebuild_index`` reconstructs
``mf_packs`` and ``mf_pack_parts`` from those files alone. So losing the database now loses no
pack, ``mf_packs`` and ``mf_pack_parts`` are index tables, and the equivalence assertion
*should* cover them.

**F7 built the rebuild and did not wire it**, reporting accurately that *"declaring a
mechanism is not wiring it - the capability is tested, the wiring is owed"* and that this file
is not its to edit. It is wired below, which makes it the **fourth** rebuilder in this module
to have been written without a production caller; the first three were closed on 2026-10-14
for the same reason. The reachability test is what keeps it wired: delete the call and it goes
red.

**Refuse versus record.** Two different failures, treated differently on purpose:

* **Refuse** when the artifact has no usable identity - unparseable JSON, the wrong
  ``kind``, no ``name``. There is nothing to index and nothing to guess at (ADR-018).
* **Record, and log the anomaly**, when a manifest is identifiable but nonconforming - a
  checkpoint missing its required ``domain``, an adapter carrying one. Refusing here
  would make an artifact that exists on disk **invisible**, which is strictly worse than
  an indexed row carrying an honest ``absent``: a silently missing checkpoint is the
  failure mode ADR-013 calls going quietly dark.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy import false as sa_false
from sqlalchemy.orm import Session

from core.atomic_dir import is_temp_dir
from core.canonical import canonical_bytes, sha256_of
from core.errors import Refusal, refuse
from db.base import Base
from models.adapter import Adapter
from models.checkpoint import Checkpoint
from models.domain_state import (
    ArtifactKind,
    DomainClassification,
    DomainState,
    classify_domain,
    render_domain,
)
from models.run import Run
from services import tree as tree_service

log = logging.getLogger("miforge.reindex")

#: The manifest filename per artifact kind. Both names are **normative**: the checkpoint
#: manifest is ``manifest.json`` beside ``model.safetensors`` (R-02.20), and the adapter's
#: is ``adapter_manifest.json`` beside ``adapter_model.safetensors``
#: (``adapter-manifest-v1.json``'s own "ON-DISK LAYOUT IS NORMATIVE" note).
MANIFEST_FILENAMES: dict[ArtifactKind, str] = {
    ArtifactKind.CHECKPOINT: "manifest.json",
    ArtifactKind.ADAPTER: "adapter_manifest.json",
}

#: The ``kind`` const each schema pins. Checked rather than inferred from the filename: a
#: directory can contain a ``manifest.json`` that is not ours, and treating it as a
#: checkpoint would index an artifact that does not exist.
MANIFEST_KINDS: dict[ArtifactKind, str] = {
    ArtifactKind.CHECKPOINT: "miforge.checkpoint-manifest",
    ArtifactKind.ADAPTER: "miforge.adapter-manifest",
}

#: Where the vendored contracts are. Resolved by walking up from this file, with an env
#: override for a container that installs the package somewhere else - the same shape the
#: schema-sync test uses, and for the same reason: a hard-coded absolute path works on one
#: workstation.
_SCHEMA_DIR_ENV = "MIFORGE_SCHEMA_DIR"
_DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parents[3] / "docs" / "schemas"

SCHEMA_FILENAMES: dict[ArtifactKind, str] = {
    ArtifactKind.CHECKPOINT: "checkpoint-manifest-v1.json",
    ArtifactKind.ADAPTER: "adapter-manifest-v1.json",
}


def schema_dir() -> Path:
    return Path(os.environ.get(_SCHEMA_DIR_ENV, str(_DEFAULT_SCHEMA_DIR)))


_schema_cache: dict[ArtifactKind, Mapping[str, Any]] = {}


def load_schema(kind: ArtifactKind) -> Mapping[str, Any]:
    """The vendored JSON Schema for ``kind``, parsed once.

    Parsed, not described: this module's job is to agree with the file, and the only way
    to agree with a file is to read it.
    """
    if kind not in _schema_cache:
        path = schema_dir() / SCHEMA_FILENAMES[kind]
        if not path.is_file():
            raise refuse(
                "schema_not_vendored",
                f"{SCHEMA_FILENAMES[kind]} is not present in {schema_dir()}; "
                f"set {_SCHEMA_DIR_ENV} to the directory holding the vendored contracts.",
                schema=SCHEMA_FILENAMES[kind],
                searched=str(schema_dir()),
            )
        _schema_cache[kind] = json.loads(path.read_text("utf-8"))
    return _schema_cache[kind]


def schema_violations(manifest: Mapping[str, Any], kind: ArtifactKind) -> list[str]:
    """Every way ``manifest`` fails its vendored schema, as short strings.

    Collected rather than raised on the first one, because the caller records them as
    anomalies beside a row it is still going to write.
    """
    import jsonschema

    validator = jsonschema.Draft202012Validator(load_schema(kind))
    return [
        f"{'/'.join(str(p) for p in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(dict(manifest)), key=lambda e: list(e.path))
    ]


# ======================================================================== scanning


@dataclass(frozen=True, slots=True)
class ScannedArtifact:
    """One artifact as its manifest describes it, ready to become an index row."""

    kind: ArtifactKind
    name: str
    path: Path
    relative_path: str
    manifest: Mapping[str, Any]
    manifest_sha256: str
    domain: DomainClassification
    anomalies: tuple[str, ...]


@dataclass
class ReindexReport:
    """What a :func:`reindex` run did. Returned so the command can say so, not logged away."""

    checkpoints: int = 0
    adapters: int = 0
    #: Dataset parts, rebuilt by F10's ``reindex_parts`` from each part's own manifest.
    #: Here rather than in a separate report because ADR-004's guarantee is about **the
    #: index**, singular: a caller that has to remember to run a second command for one
    #: table is a caller that will forget, and then the index is a picture of most of the
    #: disk - which is the failure this module is written against.
    parts: int = 0
    split_rows: int = 0
    #: Rebuilt from the RUN LOG rather than from manifests - the log is authoritative for
    #: all three (ADR-004), and each of these rebuilders had no production caller until
    #: 2026-10-14.
    steps: int = 0
    batches: int = 0
    scores: int = 0
    #: Run definitions, rebuilt by F6's ``run_launch.rebuild_definition_index`` from each
    #: run log's ``definition`` block (R-02.53). The three columns on ``mf_runs`` are an
    #: index over that block even though the row itself is a *record* - so these are
    #: repaired, and the row is never recreated.
    definitions: int = 0
    #: Pack versions, rebuilt by F7's ``registry.rebuild_index`` from each committed
    #: version's ``pack.json``. Packs were the one artifact with no on-disk manifest until
    #: the gate contract made ``pack.json`` normative; see the module docstring.
    packs: int = 0
    #: ``(path, code, message)`` per artifact that could not be indexed at all.
    refused: list[tuple[str, str, str]] = field(default_factory=list)
    #: ``(name, anomaly)`` per artifact indexed despite a nonconforming manifest.
    anomalies: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.checkpoints + self.adapters


def _parse_manifest(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_text("utf-8")
    except OSError as exc:
        raise refuse("manifest_unreadable", f"Cannot read {path}: {exc}", path=str(path)) from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise refuse(
            "manifest_not_json",
            f"{path} is not valid JSON ({exc.msg} at line {exc.lineno}); "
            f"fix or remove the file, then reindex.",
            path=str(path),
        ) from exc
    if not isinstance(parsed, dict):
        raise refuse(
            "manifest_not_an_object",
            f"{path} parses to {type(parsed).__name__}, not an object.",
            path=str(path),
        )
    return parsed


def scan_artifact(path: Path, kind: ArtifactKind, root: Path) -> ScannedArtifact:
    """Read one manifest into a :class:`ScannedArtifact`, or refuse by name.

    Args:
        path: The manifest file.
        kind: Which manifest schema it claims to be, from its filename.
        root: The artifact root, so the stored path is relative to it.

    Raises:
        Refusal: When the manifest has no usable identity. See the module docstring for
            the refuse-versus-record rule.
    """
    manifest = _parse_manifest(path)

    declared = manifest.get("kind")
    if declared != MANIFEST_KINDS[kind]:
        raise refuse(
            "manifest_wrong_kind",
            f"{path} declares kind={declared!r}; expected {MANIFEST_KINDS[kind]!r}. "
            f"A file of another kind is not indexed rather than guessed at.",
            path=str(path),
            declared_kind=declared,
        )

    name = manifest.get("name")
    if not isinstance(name, str) or not name:
        raise refuse(
            "manifest_has_no_identity",
            f"{path} carries no 'name'; there is nothing to index it under and a name "
            f"must not be synthesised from the directory.",
            path=str(path),
        )

    anomalies = list(schema_violations(manifest, kind))
    domain = classify_domain(manifest, kind=kind)
    if domain.anomaly is not None:
        anomalies.append(domain.anomaly)

    return ScannedArtifact(
        kind=kind,
        name=name,
        path=path.parent,
        relative_path=str(path.parent.relative_to(root)),
        manifest=manifest,
        # The digest of the manifest's CANONICAL bytes, from the one serialiser
        # (ADR-005). Not of the file's bytes: two byte-identical manifests differing only
        # in whitespace are the same manifest, and hashing the file would say otherwise.
        manifest_sha256=sha256_of(manifest),
        domain=domain,
        anomalies=tuple(anomalies),
    )


def scan_root(root: Path) -> Iterator[ScannedArtifact | Refusal]:
    """Walk ``root`` and yield every artifact found, or a :class:`Refusal` describing one.

    Recursive and filename-driven rather than layout-driven. The layout below the root is
    not fixed by any requirement - only the step directory's zero-padded rendering is
    (BRD-01 R-01.24) - so assuming ``<root>/<run>/<step>/`` would make ``reindex`` fail
    silently on any arrangement nobody wrote down.
    """
    if not root.is_dir():
        return
    for kind, filename in MANIFEST_FILENAMES.items():
        for path in sorted(root.rglob(filename)):
            # SKIP IN-FLIGHT AND ORPHANED STAGING DIRECTORIES. `atomic_dir` cleans up when
            # its block RAISES - but a killed process never runs that cleanup, so a crash
            # leaves a complete-looking `.tmp-save-<uuid>/` with a real manifest inside.
            #
            # Two failures, and the second is the one that matters. Without this predicate
            # the orphan is indexed as a genuine checkpoint whose `path` is a dot-prefixed
            # directory - breaking FR-005.4, which requires EVERY discovery path to exclude
            # them, and acceptance criterion 8, which requires that no partial checkpoint be
            # *discoverable*. Atomicity alone never gave that: the temp directory must be a
            # sibling of the destination for `rename` to be atomic, so it necessarily sits
            # inside the walked tree.
            #
            # And when the orphan carries the SAME `name` as a published checkpoint - which
            # is exactly what a killed *resave* leaves behind - the insert raises
            # `IntegrityError: UNIQUE constraint failed: mf_checkpoints.name` and **the
            # whole rebuild fails**. ADR-004's recovery path is then unavailable until
            # somebody finds and deletes a dot-prefixed directory by hand. A reindex that
            # cannot run is worse than an index that is stale, because the stale one still
            # answers.
            #
            # `is_temp_dir` is exported from `core.atomic_dir` for exactly this, so the
            # writer and every reader agree on what an in-flight write looks like rather
            # than each testing the prefix itself and one of them eventually not.
            # Found by F5 while building the save path.
            if any(is_temp_dir(part) for part in path.relative_to(root).parts):
                continue
            try:
                yield scan_artifact(path, kind, root)
            except Refusal as refusal:
                yield refusal


# ======================================================================== row building


def _parse_timestamp(value: Any, *, path: str) -> datetime:
    if not isinstance(value, str):
        raise refuse(
            "manifest_timestamp_missing",
            f"{path} has no usable 'created_at'; a time must not be invented from the "
            f"file's mtime, which is the copy's time and not the save's.",
            path=path,
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise refuse(
            "manifest_timestamp_unparseable",
            f"{path} has created_at={value!r}, which is not RFC 3339.",
            path=path,
        ) from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


#: The index columns each artifact kind derives from its manifest. Named here because
#: :func:`resolve_checkpoint` compares exactly these and nothing else: a divergence check
#: that enumerated columns separately from the builder would drift the first time a column
#: was added, and then stop checking it with no test going red.
CHECKPOINT_INDEXED_FIELDS = (
    "run_id",
    "step",
    "domain",
    "domain_state",
    "signal_source",
    "parent",
    "base_model",
    "merged_adapter",
    "weights_sha256",
    "selector_hash",
    "grader_hash",
    "optimizer",
    "test",
    "retention",
)
ADAPTER_INDEXED_FIELDS = (
    "run_id",
    "step",
    "domain",
    "domain_state",
    "base_name",
    "base_manifest_sha256",
    "weights_sha256",
    "adapter_config",
    "optimizer",
    "parts",
    "test",
    "retention",
)


def checkpoint_row_values(artifact: ScannedArtifact) -> dict[str, Any]:
    """The ``mf_checkpoints`` column values implied by a checkpoint manifest."""
    manifest = artifact.manifest
    return {
        "name": artifact.name,
        "run_id": str(manifest.get("run_id", "")),
        "step": int(manifest.get("step", 0)),
        # The value and the state travel together; see models/domain_state.py. The state
        # comes from the classifier, never from `manifest.get("domain")`.
        "domain": artifact.domain.value,
        "domain_state": artifact.domain.state.value,
        "signal_source": str(manifest.get("signal_source", "corpus")),
        "parent": manifest.get("parent"),
        "base_model": str(manifest.get("base_model", "")),
        "merged_adapter": manifest.get("merged_adapter"),
        "weights_sha256": str(manifest.get("weights_sha256", "")),
        "manifest_sha256": artifact.manifest_sha256,
        "selector_hash": manifest.get("selector_hash"),
        "grader_hash": manifest.get("grader_hash"),
        "optimizer": manifest.get("optimizer"),
        "test": manifest.get("test"),
        "retention": manifest.get("retention"),
        "path": artifact.relative_path,
        "created_at": _parse_timestamp(manifest.get("created_at"), path=artifact.relative_path),
    }


def adapter_row_values(artifact: ScannedArtifact) -> dict[str, Any]:
    """The ``mf_adapters`` column values implied by an adapter manifest."""
    manifest = artifact.manifest
    base = manifest.get("base") or {}
    if not isinstance(base, Mapping) or not base.get("name"):
        raise refuse(
            "adapter_manifest_has_no_base",
            f"{artifact.name} names no base. An adapter is meaningless without one and "
            f"its base must not be inferred from its directory.",
            adapter=artifact.name,
        )
    return {
        "name": artifact.name,
        "run_id": str(manifest.get("run_id", "")),
        "step": int(manifest.get("step", 0)),
        # Always NULL / not_in_kind. The CHECK constraints refuse anything else, so an
        # attempt to fill these from `base` fails at the database rather than shipping.
        "domain": artifact.domain.value,
        "domain_state": artifact.domain.state.value,
        "base_name": str(base["name"]),
        "base_manifest_sha256": base.get("manifest_sha256"),
        "weights_sha256": str(manifest.get("weights_sha256", "")),
        "manifest_sha256": artifact.manifest_sha256,
        "adapter_config": manifest.get("adapter"),
        "optimizer": manifest.get("optimizer"),
        "parts": manifest.get("parts"),
        "test": manifest.get("test"),
        "retention": manifest.get("retention"),
        "path": artifact.relative_path,
        "created_at": _parse_timestamp(manifest.get("created_at"), path=artifact.relative_path),
    }


# ======================================================================== the command


def reindex(session: Session, *, root: Path | None = None) -> ReindexReport:
    """Rebuild ``mf_checkpoints`` and ``mf_adapters`` from the manifests under ``root``.

    **A first-class command, not a recovery script** (ADR-004). It is destructive by
    design: the index is replaced by what the filesystem says, because a merge would let a
    row for a deleted checkpoint survive forever and the index would slowly stop being a
    picture of the disk.

    Args:
        session: An open session. Not committed here - the caller owns the transaction,
            so a reindex that fails half-way leaves the old index intact.
        root: The artifact root. Defaults to ``settings.checkpoint_root``.

    Returns:
        A :class:`ReindexReport` naming what was indexed, refused and recorded-with-anomaly.
    """
    if root is None:
        from core.config import get_settings

        root = get_settings().checkpoint_root

    report = ReindexReport()

    session.execute(delete(Checkpoint))
    session.execute(delete(Adapter))

    for found in scan_root(root):
        if isinstance(found, Refusal):
            report.refused.append((found.details.get("path", "?"), found.code, found.message))
            log.warning(
                "reindex_refused_artifact",
                extra={"code": found.code, "detail": found.message, **found.details},
            )
            continue

        for anomaly in found.anomalies:
            report.anomalies.append((found.name, anomaly))
            log.warning(
                "reindex_manifest_nonconforming",
                extra={"artifact": found.name, "anomaly": anomaly, "kind": str(found.kind)},
            )

        try:
            if found.kind is ArtifactKind.CHECKPOINT:
                session.add(Checkpoint(**checkpoint_row_values(found)))
                report.checkpoints += 1
            else:
                session.add(Adapter(**adapter_row_values(found)))
                report.adapters += 1
        except Refusal as refusal:
            report.refused.append((found.relative_path, refusal.code, refusal.message))
            log.warning(
                "reindex_refused_artifact",
                extra={"code": refusal.code, "detail": refusal.message, **refusal.details},
            )

    session.flush()

    # Dataset parts, from PARTS_ROOT. Imported here rather than at module scope so this
    # module keeps working if F10's package is absent - the parts index is optional in a
    # way checkpoints are not. A missing package is reported, never swallowed.
    try:
        from services.parts.index import reindex_parts
    except ImportError:  # pragma: no cover - F10 is present in this tree
        log.warning("reindex_parts_unavailable", extra={"reason": "services.parts absent"})
    else:
        # Imported here for the same reason the root default is: `get_settings` is a
        # function-local name in this scope, so a module-level reference would be an
        # UnboundLocalError for every call that passes an explicit root.
        from core.config import get_settings as _get_settings

        parts_report = reindex_parts(session, root=_get_settings().parts_root)
        report.parts = parts_report.parts
        report.split_rows = parts_report.splits
        report.refused.extend(parts_report.refused)
        session.flush()

    # ---- the three run-log rebuilders -----------------------------------------------
    # `mf_steps`, `mf_batches` and `mf_scores` are rebuilt from the RUN LOG rather than
    # from manifests, which is why they are not part of the walk above - but they are part
    # of the same guarantee. Each of these three existed with **nothing calling it**: F1's,
    # F2's and F4's rebuilders were reachable only from their own tests, and by ADR-016 #1
    # that means `reindex` did not rebuild them in production at all. Three features
    # independently reported the same one-line gap, each declining to edit this file
    # because it is not theirs - which was right, and is why it accumulated here.
    #
    # One call, one guarantee: *"a caller that has to remember to run a second command for
    # one table is a caller that will forget."*
    _rebuild_run_log_indexes(session, report)

    log.info(
        "reindex_complete",
        extra={
            "root": str(root),
            "checkpoints": report.checkpoints,
            "adapters": report.adapters,
            "parts": report.parts,
            "split_rows": report.split_rows,
            "definitions": report.definitions,
            "refused": len(report.refused),
            "anomalies": len(report.anomalies),
        },
    )
    return report


def _rebuild_run_log_indexes(session: Session, report: ReindexReport) -> None:
    """Rebuild the three run-log-derived indexes, per run, tolerating absent features.

    Each rebuilder is imported lazily and a missing module is **reported, never swallowed**:
    a feature that is not in the tree is a fact, and a silent skip would make a partial
    reindex look complete. The run ids come from ``mf_runs`` because the rebuilders are
    per-run - the run row is the only enumeration of which runs exist, and ADR-004 makes
    that a *record* table rather than an index, so it survives the drop above.
    """
    from sqlalchemy.exc import OperationalError, ProgrammingError

    from core.config import get_settings as _get_settings

    settings = _get_settings()
    run_ids = [row[0] for row in session.execute(select(Run.id)).all()]

    #: A rebuilder whose table has never been migrated. **Reported, never swallowed:** a
    #: reindex that quietly skipped a table would look complete while leaving part of the
    #: index empty, and the operator's next question - "why is this screen blank" - would
    #: have no trail. Logged with the table named, and recorded in the report so a caller
    #: sees it in the return value rather than only in a log nobody is reading.
    missing = (OperationalError, ProgrammingError)

    try:
        from training.step_record import rebuild_step_index
        from training.tasks import run_log_root
    except ImportError:  # pragma: no cover - F1 is present in this tree
        log.warning("reindex_steps_unavailable", extra={"reason": "training.step_record absent"})
    else:
        log_root = run_log_root(None)
        try:
            for run_id in run_ids:
                # PER RUN, and a run whose log is absent is reported rather than fatal.
                # The rebuilder refuses `run_log_absent` on its own, which is right when
                # someone rebuilds ONE run deliberately - *"an empty index would assert
                # that the run took none"*. But in a global rebuild a run row with no log
                # on disk is ordinary: a run that never started, or one whose log was
                # moved. Refusing the whole operation because one run of fifty has no log
                # would make ADR-004's recovery path unusable, which is the opposite of
                # what it is for.
                try:
                    report.steps += rebuild_step_index(session, log_root=log_root, run_id=run_id)
                except Refusal as per_run:
                    report.refused.append((run_id, per_run.code, per_run.message))
                    log.warning(
                        "reindex_run_skipped",
                        extra={
                            "code": per_run.code,
                            "run_id": run_id,
                            "table": "mf_steps",
                        },
                    )
            session.flush()
        except missing as exc:
            session.rollback()
            report.refused.append(("mf_steps", "index_table_absent", str(exc)[:200]))
            log.warning(
                "reindex_table_absent",
                extra={"code": "index_table_absent", "table": "mf_steps"},
            )

    try:
        from training.signals.batch_index import rebuild_batch_index
        from training.tasks import run_log_root as _log_root
    except ImportError:  # pragma: no cover - F2 is present in this tree
        log.warning("reindex_batches_unavailable", extra={"reason": "training.signals absent"})
    else:
        root = _log_root(None)
        try:
            for run_id in run_ids:
                # PER RUN, and a run whose log is absent is reported rather than fatal.
                # The rebuilder refuses `run_log_absent` on its own, which is right when
                # someone rebuilds ONE run deliberately - *"an empty index would assert
                # that the run took none"*. But in a global rebuild a run row with no log
                # on disk is ordinary: a run that never started, or one whose log was
                # moved. Refusing the whole operation because one run of fifty has no log
                # would make ADR-004's recovery path unusable, which is the opposite of
                # what it is for.
                try:
                    report.batches += rebuild_batch_index(session, log_root=root, run_id=run_id)
                except Refusal as per_run:
                    report.refused.append((run_id, per_run.code, per_run.message))
                    log.warning(
                        "reindex_run_skipped",
                        extra={
                            "code": per_run.code,
                            "run_id": run_id,
                            "table": "mf_batches",
                        },
                    )
            session.flush()
        except missing as exc:
            session.rollback()
            report.refused.append(("mf_batches", "index_table_absent", str(exc)[:200]))
            log.warning(
                "reindex_table_absent",
                extra={"code": "index_table_absent", "table": "mf_batches"},
            )

    # F6's definition index (R-02.53). Per run, from the run log's `definition` block.
    # Wired HERE rather than shipped as a fourth unwired rebuilder: `mf_steps`,
    # `mf_batches` and `mf_scores` each existed with nothing calling them, three features
    # independently reported the same one-line gap, and each rightly declined to edit this
    # file - which is exactly how they accumulated. A fourth would be the same defect with
    # the lesson already written down.
    try:
        from services.run_launch import rebuild_definition_index
        from training.tasks import run_log_root as _root_defs
    except ImportError:  # pragma: no cover - F6 is present in this tree
        log.warning(
            "reindex_definitions_unavailable",
            extra={"reason": "services.run_launch absent"},
        )
    else:
        defs_root = _root_defs(None)
        try:
            for run_id in run_ids:
                try:
                    report.definitions += rebuild_definition_index(
                        session, log_root=defs_root, run_id=run_id
                    )
                except Refusal as per_run:
                    report.refused.append((run_id, per_run.code, per_run.message))
                    log.warning(
                        "reindex_run_skipped",
                        extra={
                            "code": per_run.code,
                            "run_id": run_id,
                            "table": "mf_runs.definition_*",
                        },
                    )
            session.flush()
        except missing as exc:
            session.rollback()
            report.refused.append(("mf_runs.definition_*", "index_table_absent", str(exc)[:200]))
            log.warning(
                "reindex_table_absent",
                extra={"code": "index_table_absent", "table": "mf_runs.definition_*"},
            )

    try:
        from services.scoring.reindex_scores import reindex_scores
        from training.tasks import run_log_root as _root3
    except ImportError:  # pragma: no cover - F4 is present in this tree
        log.warning("reindex_scores_unavailable", extra={"reason": "services.scoring absent"})
    else:
        try:
            scores = reindex_scores(
                session,
                run_log_root=_root3(None),
                checkpoint_root=settings.checkpoint_root,
            )
            report.scores = int(scores.get("passes", 0) or 0)
            session.flush()
        except missing as exc:
            session.rollback()
            report.refused.append(("mf_scores", "index_table_absent", str(exc)[:200]))
            log.warning(
                "reindex_table_absent",
                extra={"code": "index_table_absent", "table": "mf_scores"},
            )

    # F7's pack registry. NOT per-run and not from the run log - a pack version is rebuilt
    # from its own `pack.json`, so this one call covers the whole store. It is here rather
    # than in a function of its own because the caller's guarantee is about *the index*,
    # singular: a caller who must remember a second command is a caller who will forget.
    try:
        from packs.registry import rebuild_index as rebuild_pack_index
    except ImportError:  # pragma: no cover - F7 is present in this tree
        log.warning("reindex_packs_unavailable", extra={"reason": "packs.registry absent"})
    else:
        # DOES THE STORE EXIST, checked WITHOUT creating it. `packs.store.packs_root()`
        # mkdirs "on demand", which is right for a writer and wrong here: a read path that
        # creates a directory turned this into an attempt to mkdir `/data/packs`, and on a
        # machine without it that raised `PermissionError`.
        #
        # An absent store is skipped **entirely** - not dropped and not rebuilt - and the
        # distinction matters. `rebuild_index` deletes before it inserts, so running it
        # against a store that is merely *not mounted yet* would clear every pack row on the
        # grounds that no file backs them. "The directory is gone" and "the volume is not
        # attached" are indistinguishable from here, and one of them makes dropping the index
        # precisely the data loss that the 2026-10-13 prohibition was written to prevent. A
        # fresh install has no pack root yet, exactly as it has no checkpoint root, and this
        # module already treats that as ordinary rather than an error.
        try:
            from packs.settings import get_pack_settings

            configured_root = get_pack_settings().packs_root
        except Exception as exc:  # pragma: no cover - settings are importable in this tree
            report.refused.append(("mf_packs", "pack_settings_unreadable", str(exc)[:200]))
            configured_root = None

        if configured_root is None:
            pass
        elif not configured_root.is_dir():
            log.info(
                "reindex_packs_no_store",
                extra={"code": "pack_store_absent", "path": str(configured_root)},
            )
        else:
            try:
                report.packs += rebuild_pack_index(session)
                session.flush()
            except Refusal as refusal:
                # A pack directory that is not a pack is skipped by the rebuilder itself. A
                # refusal here is about the store as a whole, and is reported rather than fatal
                # for the same reason a run with no log is.
                report.refused.append(("mf_packs", refusal.code, refusal.message))
                log.warning(
                    "reindex_packs_skipped",
                    extra={"code": refusal.code, "table": "mf_packs"},
                )
            except OSError as exc:
                # AN UNREADABLE PACKS ROOT MUST NOT KILL THE WHOLE REINDEX, and the first
                # version of this wiring let it. With `PACKS_ROOT` unset the default is
                # `/data/packs`; on a machine without it, `store.iter_versions()` raises
                # `PermissionError`, which is an `OSError` and was caught by neither clause - so
                # 21 tests went red and, far worse, a production reindex would have aborted
                # before reporting the checkpoints and adapters it had already rebuilt.
                #
                # That is the precise failure this module guards against one line at a time
                # elsewhere: *"refusing the whole operation because one run of fifty has no log
                # would make ADR-004's recovery path unusable, which is the opposite of what it
                # is for."* A reindex is run when something is already wrong, so every subsystem
                # it touches has to be able to fail on its own.
                #
                # Reported, never swallowed - an operator whose packs are missing from the
                # registry must find the reason in the return value, not infer it from a blank
                # screen.
                #
                # **NO ROLLBACK HERE, and that is the second half of the same lesson.** The first
                # fix added `session.rollback()` by symmetry with the `index_table_absent`
                # handlers below, and it discarded every checkpoint and adapter row the reindex
                # had already flushed - so the operation "succeeded" having indexed nothing, and
                # a further 17 tests went red. The two cases are not alike: a missing table
                # raises from the database and poisons the transaction, so a rollback is the only
                # way forward; an unreadable directory raises from the **filesystem** and leaves
                # the transaction perfectly healthy. Rolling it back throws away good work to
                # tidy up after a failure that did not touch it.
                report.refused.append(("mf_packs", "pack_store_unreadable", str(exc)[:200]))
                log.warning(
                    "reindex_packs_unreadable",
                    extra={"code": "pack_store_unreadable", "table": "mf_packs"},
                )
            except missing as exc:
                session.rollback()
                report.refused.append(("mf_packs", "index_table_absent", str(exc)[:200]))
                log.warning(
                    "reindex_table_absent",
                    extra={"code": "index_table_absent", "table": "mf_packs"},
                )


def drop_index(session: Session) -> None:
    """Empty the artifact index without rebuilding it.

    Exists for task 4.4's and 10.2's controls: "drop the index without reindexing" has to
    be a one-liner a test can call, or the control gets written as a hand-rolled DELETE
    that stops matching this module the first time a table is added.

    **It must empty every table in :data:`INDEX_TABLES`.** A drop that misses one makes the
    equivalence control vacuous for that table: the rows were never removed, so "identical
    after a rebuild" is satisfied by rows the rebuild never had to produce. That is the
    construction-green failure this project keeps finding, in the one test whose whole job is
    to prove the index is reconstructible.

    ``mf_pack_parts`` is deleted before ``mf_packs``: the child rows carry the FK.
    """
    session.execute(delete(Checkpoint))
    session.execute(delete(Adapter))
    try:
        from models.pack import Pack, PackPart
    except ImportError:  # pragma: no cover - F7 is present in this tree
        pass
    else:
        session.execute(delete(PackPart))
        session.execute(delete(Pack))
    session.flush()


#: Index tables - the ones ``reindex`` owns and task 4.4 may assert over. Record tables
#: are excluded; see the module docstring, and in particular the pack registry.
def _part_index_tables() -> tuple[str, ...]:
    """F10's part tables, if its package is present.

    Read from F10's own module rather than hard-coded, so a rename there cannot leave this
    tuple naming a table that no longer exists - which would make task 4.4's equivalence
    assertion silently skip the very rows it was added to cover.
    """
    try:
        from services.parts.tables import PART_INDEX_TABLES
    except ImportError:  # pragma: no cover
        return ()
    return tuple(PART_INDEX_TABLES)


def _pack_index_tables() -> tuple[str, ...]:
    """F7's pack tables, if its package is present.

    Read from F7's own module for the same reason as F10's: a rename there must not leave
    this tuple naming a table that no longer exists, which would make task 4.4's equivalence
    assertion silently skip the rows it was added to cover.

    These are index tables **as of 2026-10-14**, when ``pack.json`` became normative and
    ``packs.registry.rebuild_index`` could rebuild them from disk. Before that they were
    record tables and belonged in :data:`RECORD_TABLES`; the module docstring records the
    reversal.
    """
    try:
        from packs.registry import PACK_INDEX_TABLES
    except ImportError:  # pragma: no cover - F7 is present in this tree
        return ()
    return tuple(PACK_INDEX_TABLES)


INDEX_TABLES: tuple[str, ...] = (
    Checkpoint.__tablename__,
    Adapter.__tablename__,
    *_part_index_tables(),
    *_pack_index_tables(),
)
RECORD_TABLES: tuple[str, ...] = tuple(sorted(set(Base.metadata.tables) - set(INDEX_TABLES)))


# ======================================================================== the read seam


def _checkpoint_payload(row: Checkpoint) -> dict[str, Any]:
    """One checkpoint as an API response carries it.

    **This is the seam task 4.4 asserts over and the API layer serves.** The equivalence
    test compares these dicts before and after a drop-and-reindex, so an endpoint that
    builds its own body from the ORM row instead of calling this is outside the guarantee
    - which is the one thing worth knowing before wiring ``/api/v1``.
    """
    return {
        "name": row.name,
        "run_id": row.run_id,
        "step": row.step,
        "domain": row.domain,
        "domain_state": row.domain_state,
        "domain_display": render_domain(row.domain, DomainState(row.domain_state)),
        "signal_source": row.signal_source,
        "parent": row.parent,
        "base_model": row.base_model,
        "merged_adapter": row.merged_adapter,
        "weights_sha256": row.weights_sha256,
        "manifest_sha256": row.manifest_sha256,
        "selector_hash": row.selector_hash,
        "grader_hash": row.grader_hash,
        "optimizer": row.optimizer,
        "test": row.test,
        "retention": row.retention,
        "path": row.path,
        "created_at": row.created_at.isoformat(),
    }


def _adapter_payload(row: Adapter) -> dict[str, Any]:
    """One adapter as an API response carries it. See :func:`_checkpoint_payload`."""
    return {
        "name": row.name,
        "run_id": row.run_id,
        "step": row.step,
        "domain": row.domain,
        "domain_state": row.domain_state,
        "domain_display": render_domain(row.domain, DomainState(row.domain_state)),
        "base_name": row.base_name,
        "base_manifest_sha256": row.base_manifest_sha256,
        "weights_sha256": row.weights_sha256,
        "manifest_sha256": row.manifest_sha256,
        "adapter_config": row.adapter_config,
        "optimizer": row.optimizer,
        "parts": row.parts,
        "test": row.test,
        "retention": row.retention,
        "path": row.path,
        "created_at": row.created_at.isoformat(),
    }


def list_checkpoints(session: Session, *, run_id: str | None = None) -> list[dict[str, Any]]:
    """Every indexed checkpoint, ordered by name so the answer is stable.

    Ordered deliberately: task 4.4 compares lists, and an unordered query makes the
    equivalence test pass or fail on PostgreSQL's physical row order rather than on
    whether the rebuild was faithful.
    """
    stmt = select(Checkpoint).order_by(Checkpoint.name)
    if run_id is not None:
        stmt = stmt.where(Checkpoint.run_id == run_id)
    return [_checkpoint_payload(row) for row in session.execute(stmt).scalars()]


def list_adapters(session: Session, *, run_id: str | None = None) -> list[dict[str, Any]]:
    """Every indexed adapter, ordered by name. See :func:`list_checkpoints`."""
    stmt = select(Adapter).order_by(Adapter.name)
    if run_id is not None:
        stmt = stmt.where(Adapter.run_id == run_id)
    return [_adapter_payload(row) for row in session.execute(stmt).scalars()]


def _forest_divergences(
    checkpoint_rows: Sequence[Checkpoint],
    adapter_rows: Sequence[Adapter],
    root: Path,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Compare each row against its manifest on disk. **The manifest wins** (ADR-004).

    The read-only half of :func:`resolve_checkpoint`, over a page of the forest rather than
    one artifact, and it corrects **nothing**: the forest is a read and a request that
    rewrites rows would make the same request answer differently the second time, which the
    equivalence comparison would then be unable to distinguish from a faithful rebuild.

    Three outcomes per row, and the third is the one that is easy to get wrong:

    1. **The manifest agrees** - no entry, nothing logged.
    2. **The manifest differs** - the manifest's values are returned as an overlay, the row
       is named stale, and ``index_manifest_divergence`` is logged with both answers
       (FR-009.29; ADR-004 *"logged rather than silently resolved"*).
    3. **The manifest cannot be read** - missing file, unparseable JSON, no identity. The
       row is named stale and logged, and **its index values are kept**: there is nothing
       to overwrite them with, and blanking a node because its manifest is unreadable would
       hide a rendered fact behind an empty one. :func:`resolve_checkpoint` refuses in this
       case because it serves one artifact and the caller asked for *that* one; the forest
       must not make an artifact disappear from the tree, which is FR-009.28's reasoning
       applied to a second absence.

    Returns:
        ``(overlay_by_name, stale_names)``. ``overlay_by_name`` carries only the fields that
        differed, so a caller merges rather than replaces.
    """
    overlays: dict[str, dict[str, Any]] = {}
    stale: set[str] = set()

    def compare(
        name: str,
        row: Checkpoint | Adapter,
        kind: ArtifactKind,
        fields: tuple[str, ...],
        row_values: Any,
    ) -> None:
        manifest_file = _manifest_path(root, row.path, kind)
        try:
            artifact = scan_artifact(manifest_file, kind, root)
        except (Refusal, OSError) as exc:
            stale.add(name)
            log.warning(
                DIVERGENCE_EVENT,
                extra={
                    "artifact": name,
                    "divergence": "manifest_unreadable",
                    "manifest_path": str(manifest_file),
                    "resolution": "index_row_kept_and_named_stale",
                    "detail": str(exc),
                },
            )
            return
        truth = row_values(artifact)
        current = {field_name: getattr(row, field_name) for field_name in fields}
        diffs = _differences(current, truth, fields)
        if not diffs:
            return
        stale.add(name)
        overlays[name] = {field_name: answers["manifest"] for field_name, answers in diffs.items()}
        log.warning(
            DIVERGENCE_EVENT,
            extra={
                "artifact": name,
                "divergence": "field_mismatch",
                "divergent_fields": sorted(diffs),
                "resolution": "manifest_wins",
                "detail": diffs,
            },
        )

    for row in checkpoint_rows:
        compare(
            row.name, row, ArtifactKind.CHECKPOINT, CHECKPOINT_INDEXED_FIELDS, checkpoint_row_values
        )
    for arow in adapter_rows:
        compare(arow.name, arow, ArtifactKind.ADAPTER, ADAPTER_INDEXED_FIELDS, adapter_row_values)

    return overlays, stale


def _overlaid(payload: dict[str, Any], overlay: Mapping[str, Any] | None) -> dict[str, Any]:
    """Apply a manifest overlay to a rendered payload, re-deriving what depends on it.

    ``domain_display`` is a *rendering* of ``domain`` and ``domain_state``, so overlaying
    the two without re-deriving the third would leave a node showing a stale sentence
    beside a corrected value - the manifest winning on one field and losing on its own
    presentation.
    """
    if not overlay:
        return payload
    payload.update(overlay)
    if "domain" in overlay or "domain_state" in overlay:
        payload["domain_display"] = render_domain(
            payload.get("domain"), DomainState(payload["domain_state"])
        )
    return payload


def checkpoint_tree(
    session: Session,
    *,
    run_id: str | None = None,
    domain: tree_service.DomainFilter | None = None,
    kind: str | None = None,
    limit: int = tree_service.DEFAULT_LIMIT,
    offset: int = 0,
    root: Path | None = None,
) -> dict[str, Any]:
    """R-02.29's artifact forest: two node kinds, three edges, one mechanism.

    A checkpoint joins on ``parent``; an adapter joins on the base **its own manifest
    names**; a merged checkpoint has **both** edges - ``parent`` to its base and
    ``merged_adapter`` to the adapter baked into it. A node's domain may be null, and an
    adapter's is ``not_in_kind``: the tree carries the state, not a blank, so a renderer
    cannot confuse "no domain" with "not recorded".

    **This was widened on 2026-10-14 rather than wrapped, and that is the point.** It
    carried ``{name, kind, run_id, step, domain, domain_state}`` per node and
    ``{child, parent, edge}`` per edge - a strict subset of what R-02.29 asks for, with no
    ``test`` score (the requirement's fourth fact) and no edge states, so it could not have
    served FR-009.27's ``truncated``/``root`` distinction or FR-009.28's unindexed run even
    once it were mounted. It also had **no production caller**, which ADR-016 #1 says is not
    a shipped capability. A second tree builder beside it would have been a second source of
    truth for the one join in the product - the defect family this project calls the
    ``DB_RUN_TYPE`` translation map - so the join moved to the pure
    :func:`services.tree.compose` and this function became its only query path. Because
    :func:`index_snapshot` calls it, **foundation task 4.4's equivalence test now covers the
    whole of the tree screen's payload** rather than a subset of it.

    Args:
        session: an open session.
        run_id, domain, kind: the filters. ``domain`` is a parsed
            :class:`services.tree.DomainFilter`; an unparseable value is refused by its
            parser and **never widened to no filter**.
        limit, offset: flat pagination over nodes, not by subtree.
        root: the artifact root the manifests are compared against. Defaults to
            ``settings.checkpoint_root``.

    Returns:
        The :class:`schemas.tree.TreeResponse` as JSON.
    """
    if root is None:
        from core.config import get_settings

        root = get_settings().checkpoint_root
    domain = domain if domain is not None else tree_service.DomainFilter.parse(None)

    checkpoint_stmt = select(Checkpoint).order_by(Checkpoint.name)
    adapter_stmt = select(Adapter).order_by(Adapter.name)
    if run_id is not None:
        checkpoint_stmt = checkpoint_stmt.where(Checkpoint.run_id == run_id)
        adapter_stmt = adapter_stmt.where(Adapter.run_id == run_id)
    if domain.kind is tree_service.DomainFilterKind.NAMED:
        checkpoint_stmt = checkpoint_stmt.where(Checkpoint.domain == domain.name)
        # An adapter manifest declares no `domain`, so no adapter can match a named domain.
        # Excluded here rather than filtered out later, because reading the base's domain to
        # make one match would be the fabricated value ADR-018 forbids (FPRD F-11).
        adapter_stmt = adapter_stmt.where(sa_false())
    elif domain.kind is tree_service.DomainFilterKind.NO_DOMAIN:
        # `asserted_null` ONLY. `absent` is a missing required key and `not_in_kind` is an
        # adapter; neither is an artifact saying "I have no domain" (FR-009.25).
        checkpoint_stmt = checkpoint_stmt.where(
            Checkpoint.domain_state == DomainState.ASSERTED_NULL.value
        )
        adapter_stmt = adapter_stmt.where(sa_false())
    if kind == "checkpoint":
        adapter_stmt = adapter_stmt.where(sa_false())
    elif kind == "adapter":
        checkpoint_stmt = checkpoint_stmt.where(sa_false())

    checkpoint_rows = list(session.execute(checkpoint_stmt).scalars())
    adapter_rows = list(session.execute(adapter_stmt).scalars())

    # Every name the INDEX knows, not merely the ones this filter kept. A target found here
    # but absent from the page is `not_loaded`, which is not `truncated`: a paging or
    # filtering artefact must not read as a broken lineage (FTDD section 5.2).
    all_names = {
        *session.execute(select(Checkpoint.name)).scalars(),
        *session.execute(select(Adapter.name)).scalars(),
    }
    known_run_ids = set(
        session.execute(
            select(Run.id).where(
                Run.id.in_(
                    {row.run_id for row in checkpoint_rows} | {r.run_id for r in adapter_rows}
                )
            )
        ).scalars()
    )

    overlays, stale = _forest_divergences(checkpoint_rows, adapter_rows, root)

    response = tree_service.compose(
        [_overlaid(_checkpoint_payload(row), overlays.get(row.name)) for row in checkpoint_rows],
        [_overlaid(_adapter_payload(row), overlays.get(row.name)) for row in adapter_rows],
        known_run_ids,
        all_names,
        index_stale=stale,
        limit=limit,
        offset=offset,
    )
    return response.model_dump(mode="json")


def index_snapshot(session: Session, *, root: Path | None = None) -> dict[str, Any]:
    """Everything task 4.4 compares, in one call.

    One function so the equivalence test cannot accidentally compare a subset: adding a
    query to the read seam and forgetting to add it here is the construction-green
    failure this project keeps finding, so the seam is enumerated in one place.

    ``root`` is threaded through to :func:`checkpoint_tree`, which compares every row
    against its manifest. **Passing it is not optional in a test**: left to default, the
    comparison runs against ``/data/checkpoints``, every manifest is unreadable, and the
    snapshot reports a forest of stale nodes - equal before and after, so the equivalence
    assertion would still pass while asserting nothing about the thing it names.
    """
    return {
        "checkpoints": list_checkpoints(session),
        "adapters": list_adapters(session),
        "tree": checkpoint_tree(session, root=root),
    }


# ======================================================================== task 4.5


DIVERGENCE_EVENT = "index_manifest_divergence"


def _manifest_path(root: Path, relative_path: str, kind: ArtifactKind) -> Path:
    return root / relative_path / MANIFEST_FILENAMES[kind]


def _differences(
    row_values: Mapping[str, Any], manifest_values: Mapping[str, Any], fields: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """Which of ``fields`` disagree, with both answers.

    Compared through :func:`core.canonical.canonical_bytes` rather than ``==`` so that a
    JSON column round-tripped by two different drivers does not read as a divergence on
    key order alone - and so the comparison uses the project's one serialiser (ADR-005).
    """
    out: dict[str, dict[str, Any]] = {}
    for name in fields:
        mine = row_values.get(name)
        theirs = manifest_values.get(name)
        if canonical_bytes(mine) != canonical_bytes(theirs):
            out[name] = {"index": mine, "manifest": theirs}
    return out


def resolve_checkpoint(session: Session, name: str, *, root: Path | None = None) -> dict[str, Any]:
    """Read one checkpoint, **letting the manifest win** over the index (task 4.5).

    The index is a cache. When it disagrees with the manifest beside the weights, the
    manifest is right by definition (ADR-004), the disagreement is logged as
    ``index_manifest_divergence`` with the fields and both answers, and the row is
    corrected so the next read is cheap and the same line is not emitted forever.

    Logged rather than silently corrected because a divergence means something wrote the
    index without going through the manifest, or edited a manifest in place - both are
    bugs, and a silent self-heal is how they stay unfound.

    Args:
        session: An open session.
        name: The checkpoint's fully qualified name.
        root: The artifact root. Defaults to ``settings.checkpoint_root``.

    Raises:
        Refusal: When neither the index nor the filesystem knows the name.
    """
    if root is None:
        from core.config import get_settings

        root = get_settings().checkpoint_root

    row = session.get(Checkpoint, name)
    if row is None:
        raise refuse(
            "checkpoint_not_indexed",
            f"No checkpoint named {name!r} in the index; run reindex if it exists on disk.",
            checkpoint=name,
        )

    manifest_file = _manifest_path(root, row.path, ArtifactKind.CHECKPOINT)
    if not manifest_file.is_file():
        # The authority is gone. Do NOT serve the index row as though it were still
        # backed: that is the index silently outliving the artifact, which is the exact
        # inversion of ADR-004.
        log.warning(
            DIVERGENCE_EVENT,
            extra={
                "artifact": name,
                "divergence": "manifest_missing",
                "manifest_path": str(manifest_file),
            },
        )
        raise refuse(
            "manifest_missing_for_indexed_checkpoint",
            f"{name} is indexed but {manifest_file} does not exist. The manifest is the "
            f"record, so this row is stale; reindex to remove it.",
            checkpoint=name,
            manifest_path=str(manifest_file),
        )

    artifact = scan_artifact(manifest_file, ArtifactKind.CHECKPOINT, root)
    truth = checkpoint_row_values(artifact)
    current = {field_name: getattr(row, field_name) for field_name in CHECKPOINT_INDEXED_FIELDS}
    diffs = _differences(current, truth, CHECKPOINT_INDEXED_FIELDS)

    if diffs:
        log.warning(
            DIVERGENCE_EVENT,
            extra={
                "artifact": name,
                "divergence": "field_mismatch",
                "divergent_fields": sorted(diffs),
                "resolution": "manifest_wins",
                "detail": diffs,
            },
        )
        for field_name, answers in diffs.items():
            setattr(row, field_name, answers["manifest"])
        row.manifest_sha256 = truth["manifest_sha256"]
        session.flush()

    return _checkpoint_payload(row)
