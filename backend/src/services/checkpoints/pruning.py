"""The acting half: read manifests, then delete a directory **and** its index row  (OQ-4).

:mod:`services.checkpoints.lifecycle` decides and this destroys. Nothing here chooses what
goes.

------------------------------------------------------------------------------------
WHAT A DELETION IS, AND WHAT HAPPENS IF THE PROCESS DIES IN THE MIDDLE OF ONE
------------------------------------------------------------------------------------

A checkpoint directory is 2-10 GB. ``shutil.rmtree`` over it is not atomic and not quick,
and a worker that is OOM-killed part-way through one leaves a directory holding a
``manifest.json``, a ``config.json`` and **half a ``model.safetensors``** - which is
discoverable, indexable, and reads as a whole checkpoint. That is acceptance criterion 8's
failure (*no partial checkpoint may be discoverable*) arriving through the delete path
rather than the write path.

So a deletion is **rename first**:

1. ``os.rename`` the directory to a sibling ``.tmp-save-prune-<uuid>``. One syscall, atomic
   within the filesystem, and :func:`core.atomic_dir.is_temp_dir` already excludes that
   prefix from **every** discovery path - the listing API, the index upsert, the
   ``reindex`` walk and ``latest`` resolution. The checkpoint stops existing for every
   reader at that instant, whole or not.
2. Delete the ``mf_checkpoints`` row and **commit**, per artifact.
3. ``rmtree`` the staged directory. A failure here is logged and not raised: the bytes are
   still on the volume, but nothing can find them and the next sweep's ``rmtree`` or an
   operator's ``rm -rf`` finishes the job.

**Reusing F5's own ``TMP_PREFIX`` rather than inventing ``.prune-`` is the whole reason
step 1 works.** A fresh prefix would not be in ``is_temp_dir``, so a staged directory would
be walked by ``reindex`` and re-indexed as a live checkpoint - and on a killed sweep it
would be re-indexed *half deleted*. The prefix is extended with ``prune-`` so an operator
finding an orphan can tell a killed save from a killed prune, which is the one thing the
shared prefix costs.

**The commit is per artifact, not per sweep.** A sweep deleting thirty directories under
one transaction, killed at the twentieth, would leave twenty directories gone and thirty
rows present. Per artifact, the divergence window is one artifact wide - and ADR-004 makes
even that recoverable, because the manifest is the authority and ``reindex`` reconstructs
the index from what is actually on disk. *The direction matters: this order can leave a row
for a checkpoint that is gone, which ``reindex`` and ``resolve_checkpoint`` already handle
by name. The reverse order can leave a directory nothing points at, which only a human
finds.*

------------------------------------------------------------------------------------
AN UNREADABLE MANIFEST STOPS THE WHOLE SWEEP
------------------------------------------------------------------------------------

:func:`collect` refuses ``checkpoint_lifecycle_unreadable_manifest`` when any manifest
under the root cannot be parsed, and the sweep then deletes **nothing**.

This is fail-closed and it is not caution for its own sake: the ancestry set is built from
manifest fields, so a manifest nobody can read is a set of three edges nobody can see. One
of them may be the only thing naming a checkpoint this policy is about to delete, and the
prune would detach a subtree while every check passed. ``reindex`` already reports such a
file by name; an operator fixes or removes it and the next sweep proceeds. *A visible
refusal is recoverable; a silently wrong deletion is not* (ADR-013).
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import delete as sa_delete

from core.atomic_dir import TMP_PREFIX, is_temp_dir
from core.errors import Refusal, refuse
from models.adapter import Adapter
from models.checkpoint import Checkpoint
from models.domain_state import ArtifactKind
from services.checkpoints.lifecycle import ArtifactRef, Decision, LifecyclePlan
from services.checkpoints.paths import resolve_latest, root_of
from services.reindex import ScannedArtifact, scan_root

log = logging.getLogger("miforge.checkpoints.pruning")

__all__ = [
    "PRUNE_PREFIX",
    "Collected",
    "PruneReport",
    "collect",
    "execute",
    "staging_name",
]

#: The staged-for-deletion directory name. **Starts with F5's ``TMP_PREFIX``** so
#: :func:`core.atomic_dir.is_temp_dir` already hides it from every discovery path; see the
#: module docstring for why a new prefix would be a defect rather than a tidier name.
PRUNE_PREFIX = f"{TMP_PREFIX}prune-"


def staging_name() -> str:
    """A unique staged name. Separate so a test can assert the prefix invariant on it."""
    return f"{PRUNE_PREFIX}{uuid4().hex}"


@dataclass(frozen=True)
class Collected:
    """Everything :func:`services.checkpoints.lifecycle.plan` needs, read from disk."""

    artifacts: tuple[ArtifactRef, ...] = ()
    #: The names ``latest`` resolves to, at most one per run.
    latest_names: frozenset[str] = field(default_factory=frozenset)


@dataclass
class PruneReport:
    """What one execution did, and what it declined."""

    deleted: list[str] = field(default_factory=list)
    #: ``(name, error)`` - the rename or the row delete failed, so nothing was removed.
    failed: list[tuple[str, str]] = field(default_factory=list)
    #: ``(name, error)`` - staged and de-indexed, but the bytes are still there. Not
    #: discoverable, so reported rather than raised.
    orphaned: list[tuple[str, str]] = field(default_factory=list)
    #: True when nothing was touched because the caller asked for a dry run.
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "deleted": list(self.deleted),
            "deleted_count": len(self.deleted),
            "failed": [{"name": n, "error": e} for n, e in self.failed],
            "orphaned": [{"name": n, "error": e} for n, e in self.orphaned],
            "dry_run": self.dry_run,
        }


def _ref_from(artifact: ScannedArtifact) -> ArtifactRef:
    """One :class:`ArtifactRef` from one scanned manifest, reading all three edges."""
    manifest = artifact.manifest
    parent = manifest.get("parent")
    base = manifest.get("base")
    merged = manifest.get("merged_adapter")
    return ArtifactRef(
        name=artifact.name,
        kind=artifact.kind,
        run_id=str(manifest.get("run_id", "")),
        step=int(manifest.get("step", 0)),
        relative_path=artifact.relative_path,
        parent=parent if isinstance(parent, str) else None,
        base_name=(base.get("name") if isinstance(base, dict) else None),
        merged_adapter_name=(merged.get("name") if isinstance(merged, dict) else None),
    )


def collect(root: Path | None = None) -> Collected:
    """Read every manifest under ``root`` into refs, and resolve each run's ``latest``.

    **Manifests, never ``mf_checkpoints``** (ADR-004): which checkpoints are ancestors is a
    correctness answer and the index holds none of those. It is also what makes this work
    on a root that was merely copied here, where there are no run rows at all.

    Raises:
        Refusal: ``checkpoint_lifecycle_unreadable_manifest`` when any manifest under the
            root cannot be parsed. See the module docstring - the sweep then deletes
            nothing, because an unreadable manifest is three invisible ancestor edges.
    """
    base = root_of(root)
    refs: list[ArtifactRef] = []
    unreadable: list[str] = []

    for found in scan_root(base):
        if isinstance(found, Refusal):
            unreadable.append(f"{found.code}: {found.message}")
            continue
        refs.append(_ref_from(found))

    if unreadable:
        raise refuse(
            "checkpoint_lifecycle_unreadable_manifest",
            f"{len(unreadable)} manifest(s) under {base} could not be read, so the "
            f"ancestor edges they carry are invisible and no checkpoint can be proved "
            f"unreferenced. Nothing was deleted. Fix or remove them, then reindex: "
            + "; ".join(unreadable[:5]),
            root=str(base),
            unreadable=unreadable,
        )

    latest: set[str] = set()
    for run_id in sorted({ref.run_id for ref in refs if ref.is_checkpoint and ref.run_id}):
        try:
            step = resolve_latest(run_id, root=base)
        except Refusal:
            # A run with no resolvable `latest` and no published step. Nothing to protect,
            # and refusing the sweep over it would make one odd run block every other.
            continue
        for ref in refs:
            if ref.is_checkpoint and ref.run_id == run_id and ref.step == step:
                latest.add(ref.name)
                break
        else:
            log.warning(
                "latest resolves to a step with no readable manifest",
                extra={
                    "code": "checkpoint_lifecycle_latest_unmatched",
                    "run_id": run_id,
                    "step": step,
                },
            )

    return Collected(artifacts=tuple(refs), latest_names=frozenset(latest))


def execute(
    session: Any,
    plan: LifecyclePlan,
    *,
    root: Path | None = None,
    dry_run: bool = True,
) -> PruneReport:
    """Carry out ``plan``. **Nothing happens when ``dry_run`` is true.**

    Args:
        session: an open session. **Committed here, once per artifact** - see the module
            docstring; a sweep-wide transaction turns a killed sweep into a wholesale
            divergence instead of a one-artifact one.
        plan: from :func:`services.checkpoints.lifecycle.plan`.
        root: the artifact root.
        dry_run: the default, and it is ``True`` on purpose. ``False`` means delete.

    Returns:
        A :class:`PruneReport`. A refusal is never raised for one artifact's failure: a
        permission error on one directory must not stop the other twenty-nine, which is
        the one-subsystem-kills-everything shape ``reindex`` was fixed for.
    """
    base = root_of(root)
    report = PruneReport(dry_run=dry_run)

    if dry_run:
        for decision in plan.delete:
            log.info(
                "checkpoint prune (dry run) would delete",
                extra={
                    "code": "checkpoint_prune_dry_run",
                    "checkpoint": decision.name,
                    "run_id": decision.run_id,
                    "step": decision.step,
                    "path": decision.relative_path,
                    "reason": decision.reason,
                },
            )
        return report

    for decision in plan.delete:
        directory = base / decision.relative_path
        if is_temp_dir(directory.name):
            # Cannot happen through `collect` - `scan_root` skips them - so if it does,
            # something handed this function a plan it did not produce.
            report.failed.append((decision.name, "refusing to act on a temp directory"))
            continue
        if not directory.is_dir():
            # Already gone: a previous sweep that died between the rename and the commit.
            # The row is the only thing left, and removing it is the repair.
            _forget_row(session, decision)
            report.deleted.append(decision.name)
            log.warning(
                "checkpoint prune found the directory already gone; removed the index row",
                extra={
                    "code": "checkpoint_prune_row_only",
                    "checkpoint": decision.name,
                    "path": decision.relative_path,
                },
            )
            continue

        staged = directory.parent / staging_name()
        try:
            # ---- (1) the atomic part: it stops being discoverable here ----
            os.rename(directory, staged)
        except OSError as exc:
            report.failed.append((decision.name, f"{type(exc).__name__}: {exc}"))
            log.error(
                "checkpoint prune could not stage a directory for deletion",
                extra={
                    "code": "checkpoint_prune_stage_failed",
                    "checkpoint": decision.name,
                    "path": decision.relative_path,
                    "error": type(exc).__name__,
                },
            )
            continue

        try:
            # ---- (2) the index row, committed before the bytes go ----
            _forget_row(session, decision)
            session.commit()
        except Exception as exc:  # pragma: no cover - a database fault mid-sweep
            session.rollback()
            report.failed.append((decision.name, f"{type(exc).__name__}: {exc}"))
            log.error(
                "checkpoint prune staged a directory and could not de-index it",
                extra={
                    "code": "checkpoint_prune_deindex_failed",
                    "checkpoint": decision.name,
                    "staged": staged.name,
                    "error": type(exc).__name__,
                },
            )
            continue

        # ---- (3) the slow part. Already invisible, so a failure is a report ----
        try:
            shutil.rmtree(staged)
        except OSError as exc:
            report.orphaned.append((decision.name, f"{type(exc).__name__}: {exc}"))
            log.error(
                "checkpoint prune left bytes behind; they are staged and undiscoverable",
                extra={
                    "code": "checkpoint_prune_orphaned",
                    "checkpoint": decision.name,
                    "staged": str(staged),
                    "error": type(exc).__name__,
                },
            )

        report.deleted.append(decision.name)
        log.warning(
            "checkpoint pruned",
            extra={
                "code": "checkpoint_pruned",
                "checkpoint": decision.name,
                "run_id": decision.run_id,
                "step": decision.step,
                "path": decision.relative_path,
                "reason": decision.reason,
            },
        )

    return report


def _forget_row(session: Any, decision: Decision) -> None:
    """Delete the index row for one artifact. Adapters are included for completeness only.

    By ``name``, which is the primary key of both index tables and is manifest-derived, so
    the row removed is the row for the artifact just staged and not one that happens to
    share a path.
    """
    table = Checkpoint if decision.kind is ArtifactKind.CHECKPOINT else Adapter
    session.execute(sa_delete(table).where(table.name == decision.name))
