"""F5 - save, load, continue, branch  (R-02.20, R-02.21, R-02.22, R-02.23, R-02.51).

**What this package is.** Everything upstream of it is ephemeral: F1 holds tensors in one
process's memory, F2 feeds them, F3 and F4 judge them, and killing the worker loses all of
it. This package is what turns a step into an **artifact** - a standard Hugging Face
directory that outlives the process, outlives the database, loads in software that never
heard of miForge, and carries enough of its own history to be found again in a tree of a
hundred siblings.

## Three mechanisms, each in one place

1. **The atomic directory write** - and it is *not here*. It is
   :func:`core.atomic_dir.atomic_dir`, written in the foundation because F1 is a wave
   earlier and imports it. Two implementations of the guarantee that replaced the
   torn-weights apparatus would be two answers (ADR-005 applied to a guarantee rather than
   to a hash), so this package is its **second caller, not its second implementation**.
2. **The manifest**, assembled from six other features' values. :mod:`.manifest` owns the
   write and never the value: every borrowed field is an explicit keyword argument and an
   absent one is a named refusal, because the failure mode is a plausible default
   (``signal_source: "corpus"``, a zeroed score, an omitted ``domain``) baked permanently
   into the one artifact meant to outlive everything (ADR-018).
3. **The resume plan** - :mod:`.resume`. Resolve a checkpoint, **verify its hash before a
   single tensor is read**, resolve trainer state through F1's reader, record
   ``restored`` or ``fresh``. A continue and a branch are the same plan with two
   differences: the ``run_id``, and what ``parent`` becomes.

## The two rules that shape everything else

**The manifest is the truth; the index is a projection** (ADR-004). Every read that must be
*correct* reads a manifest from disk - one checkpoint, the lineage walk, the resume plan,
verification. Every read that must be *fast over many rows* reads ``mf_checkpoints``. Where
they disagree the manifest wins and the disagreement is logged, which
:func:`services.reindex.resolve_checkpoint` implements - this package does not write a
second answer to that question.

**A checkpoint is addressed by ``(run_id, step)``**, never by a database surrogate id. Ids
are reassigned by a rebuild and foundation task 4.4 demands identical API responses across
one, so an exposed id would make ADR-004's claim untestable.

## What this package deliberately does not do

It calls no sibling application (ADR-010, ADR-022) and references no miLLM endpoint - not
even the ones R-02.21 and R-02.22 used to name before the 2026-10-13 corrections. It edits
no vendored schema: ``docs/schemas/checkpoint-manifest-v1.json`` is a cross-repo contract
and is cited here, never restated. It builds no export, because no requirement defines that
one and inventing a noun before someone asks for it is how this project got "packs" wrong
twice.

*This paragraph read "it builds no export, no pruning and no deletion" until 2026-10-16,
and the pruning half is now false: the operator settled OQ-4 and* :mod:`.lifecycle` *decides
what goes while* :mod:`.pruning` *removes it. Corrected rather than deleted, because the
reason it was true is still the standing rule - nobody had asked, and the policy arrived as
a decision rather than as an inference from the mockup. Someone did ask.*

**Deletion is still not an API surface.** There is no route; the only caller is
``miforge.maintenance.checkpoint_prune`` on the beat schedule, and it is off and dry-run by
default. ``endpoints/checkpoints.py``'s own docstring still says *"Nothing defines pruning,
deletion or archival"* - true of the HTTP surface it describes, stale about the project, and
owed a correction by whoever owns that file next.
"""

from __future__ import annotations

from services.checkpoints.lineage import Lineage, LineageNode, walk
from services.checkpoints.manifest import (
    ManifestFields,
    assert_no_dropped_keys,
    assert_no_undeclared_keys,
    build,
    read,
    resolve_base_model,
)
from services.checkpoints.paths import (
    checkpoint_dir,
    parse_qualified_name,
    qualified_name,
    resolve_latest,
    run_dir,
    step_dir_name,
)
from services.checkpoints.resume import ResumePlan, resume_plan
from services.checkpoints.service import CHECKPOINT_SAVED_EVENT, SaveOutcome, save
from services.checkpoints.writer import assert_file_set, verify_weights, write

__all__ = [
    "CHECKPOINT_SAVED_EVENT",
    "Lineage",
    "LineageNode",
    "ManifestFields",
    "ResumePlan",
    "SaveOutcome",
    "assert_file_set",
    "assert_no_dropped_keys",
    "assert_no_undeclared_keys",
    "build",
    "checkpoint_dir",
    "parse_qualified_name",
    "qualified_name",
    "read",
    "resolve_base_model",
    "resolve_latest",
    "resume_plan",
    "run_dir",
    "save",
    "step_dir_name",
    "verify_weights",
    "walk",
    "write",
]
