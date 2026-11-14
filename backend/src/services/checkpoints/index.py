"""The index row for one checkpoint - a projection with no authority  (FR-005.17, 18).

**This module writes one row and owns no answer.** ADR-004: the on-disk manifest is
authoritative and PostgreSQL holds a rebuildable index, so every value below is derived
from a manifest and nothing here is consulted for a correctness decision.

## It builds the row through ``reindex``'s own builder, deliberately

:func:`services.reindex.scan_artifact` and :func:`services.reindex.checkpoint_row_values`
are what a rebuild uses. Calling them here is what makes **AC-005.9** - seed saves, capture
every response, drop the index, ``reindex``, assert the responses are *identical* - a real
assertion rather than a comparison of two code paths that happen to agree today. A second
row builder in this module would drift on the first column anyone added, and the
equivalence test would go red long after the drift, pointing at the rebuild.

So there is no ``upsert`` here that knows any column names. The only thing this module adds
to the foundation's machinery is *when*: after the rename, never before.

## What is deliberately not here

No ``reindex`` command - foundation task 4.3 owns it and this feature contributes into it
rather than writing a second walk. No divergence rule - that is
:func:`services.reindex.resolve_checkpoint`, where the manifest wins and the disagreement
is logged with both values. No ``provider`` column and no ``depth`` column: depth is a
property of a walk rather than of a node, and nothing filters by provider.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from models.checkpoint import Checkpoint
from models.domain_state import ArtifactKind
from services.checkpoints.manifest import MANIFEST_NAME
from services.checkpoints.paths import root_of
from services.reindex import MANIFEST_FILENAMES, checkpoint_row_values, scan_artifact

log = logging.getLogger("miforge.checkpoints.index")

__all__ = ["checkpoint_is_indexed_by_reindex", "upsert_from_manifest"]


def upsert_from_manifest(session: Any, directory: Path, *, root: Path | None = None) -> Checkpoint:
    """Index the checkpoint published at ``directory``. **After the rename, never before.**

    Args:
        session: an open session. Not committed here - the caller owns the transaction, so
            a save whose caller rolls back leaves no row for a checkpoint it did not keep.
        directory: the published step directory.
        root: the checkpoint root, so the stored path is relative to it. Relative so the
            index survives the root being mounted somewhere else, which it is between a
            developer's machine and the GPU node's PVC.

    Raises:
        Refusal: whatever ``scan_artifact`` refuses - an unparseable manifest, the wrong
            ``kind``, no ``name``. Nothing is guessed from the directory.
    """
    base = root_of(root)
    artifact = scan_artifact(directory / MANIFEST_NAME, ArtifactKind.CHECKPOINT, base)
    values = checkpoint_row_values(artifact)

    row = session.get(Checkpoint, artifact.name)
    if row is None:
        session.add(Checkpoint(**values))
    else:
        for column, value in values.items():
            setattr(row, column, value)
    session.flush()

    if artifact.anomalies:
        # Recorded, not refused: a nonconforming manifest that is still identifiable should
        # be visible in the index rather than invisible on disk. Going quietly dark is the
        # failure ADR-013 names.
        for anomaly in artifact.anomalies:
            log.warning(
                "checkpoint_manifest_nonconforming",
                extra={"artifact": artifact.name, "anomaly": anomaly},
            )

    return session.get(Checkpoint, artifact.name)  # type: ignore[no-any-return]


def checkpoint_is_indexed_by_reindex() -> bool:
    """Whether ``reindex`` still scans for checkpoint manifests at all.

    **The wiring assertion, not an existence assertion.** A test that imported this module
    and called :func:`upsert_from_manifest` would prove a function runs, not that a rebuild
    reaches checkpoints - which is the thing AC-005.9 depends on and the thing that goes
    silently missing. ``reindex`` drives its walk from ``MANIFEST_FILENAMES``, so this reads
    that table rather than describing it.
    """
    return ArtifactKind.CHECKPOINT in MANIFEST_FILENAMES
