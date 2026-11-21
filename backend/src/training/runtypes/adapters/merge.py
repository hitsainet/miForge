"""Merge an adapter down into a new checkpoint  (R-02.45; FR-008.35 - FR-008.41).

Four steps, and **the order is the requirement**:

1. read the adapter's manifest (which refuses an unrecognised scaling rule);
2. **refuse unless a checkpoint of the unmerged base exists** - before any tensor is
   touched, not before the write;
3. apply the adapter;
4. hand the result to F5's checkpoint writer with **both** ancestors.

## Why the refusal comes first, and why it exists at all

A merge **cannot be undone by a second merge**: once the base has moved, the adapter's
contribution cannot be subtracted back out. So R-02.45 refuses a merge that would destroy
the only copy of the unmerged base, and AC17 is that refusal. Running it before step 3
rather than before step 4 is not tidiness - computing merged tensors for a merge that will
be refused wastes the card and makes a refusal look like a failure.

**The strict, fail-closed reading is implemented** (FPRD D15): a base loaded from a Hub repo
at a pinned revision is still *obtainable* but is **not a checkpoint**, and the refusal does
not admit it. If the operator's intent is recoverability rather than local presence, R-02.45
could admit a pinned revision; the strict reading costs one save and is the safer default
meanwhile. Recorded, not decided here.

## No barrier, and no second checkpoint writer

Standalone, the protection is the **atomic directory write plus this refusal** - a directory
write cannot tear (ADR-015, AC16 as split 2026-10-13). The commit-bracket form of a merge,
applied to a serving process's resident tensors inside quiesce -> ``commit-begin`` -> write
-> ``commit-end`` -> resume, is section 5.10's and F11's, and nothing on this path references
a miLLM endpoint (ADR-022). Step 4 is F5's writer: a second checkpoint writer would be a
second manifest.

## The adapter survives its own merge

Merging does not consume, move or delete the adapter directory (FR-008.41), because *"a
served adapter can be taken off, swapped, or compared against the bare base"* and a merged
one cannot. The merged checkpoint is an additional artifact, not a replacement.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.canonical import sha256_of
from core.errors import refuse
from services.checkpoints import manifest as checkpoint_manifest
from services.checkpoints import service as checkpoint_service
from services.checkpoints.manifest import MANIFEST_NAME as CHECKPOINT_MANIFEST_NAME
from services.checkpoints.manifest import ManifestFields
from services.checkpoints.paths import checkpoint_dir, parse_qualified_name
from training.runtypes.adapters import load as adapter_load
from training.runtypes.adapters import paths as adapter_paths
from training.runtypes.adapters.config import validate as validate_adapter_config

log = logging.getLogger(__name__)

#: The outcomes R-02.45 allows to be *recorded*. A refusal is raised rather than returned -
#: a caller that can ignore a return value can ignore a refusal - so ``refused`` is the
#: value the record carries, set from the raised refusal's code by the caller.
OUTCOME_WRITTEN = "written"
OUTCOME_REFUSED = "refused"
OUTCOME_INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class BaseCheckpoint:
    """A located, unmerged base checkpoint: the thing AC17 requires to survive."""

    name: str
    directory: Path
    manifest: dict[str, Any]
    manifest_sha256: str

    @property
    def base_model(self) -> str:
        return str(self.manifest["base_model"])


@dataclass(frozen=True)
class MergeOutcome:
    """What one merge did. The record of an operation whose artifact is a checkpoint.

    **A merge is not a run** and gets no ``mf_runs`` row: F6's FR-006.55 gives the
    mechanical reason F8's own answer lacked - a stepless run row would be refused at submit
    for having no stop condition, never heartbeated and therefore reaped by the janitor, and
    would be a run whose step series is empty, which every stop condition and every curve in
    F6 assumes is impossible.

    It is **not persisted**: ``mf_merges`` does not exist in the schema, and adding it is a
    migration this feature does not own. See the feature report for the DDL. The consequence
    the operation-not-a-run answer created - that a merge appears in no run list - is
    reported too; F8 builds no operations view on a feature document's say-so.
    """

    outcome: str
    adapter_name: str
    adapter_step: int
    adapter_manifest_sha256: str
    base_checkpoint_name: str
    result_checkpoint_name: str | None = None
    refusal_code: str | None = None


def resolve_base_checkpoint(base: Mapping[str, Any], *, root: Path | None = None) -> BaseCheckpoint:
    """Locate the checkpoint the **adapter** names as its base, or refuse.

    The identity resolved is ``base.name``, in the order the schema's own description fixes:
    by ``manifest_sha256`` when present - the exact base, even if its directory moved -
    otherwise by name. A base whose ``manifest_sha256`` is ``null`` was, by the schema's own
    words, *"not itself a saved checkpoint"*, so there is nothing to locate and AC17's
    refusal fires.

    Raises:
        Refusal: ``merge_refused_no_unmerged_base`` (AC17, E-11) when the base is not a
            saved checkpoint, when its directory is gone, or when its manifest hash no
            longer matches the one the adapter recorded - in which case the checkpoint on
            disk is **a different parameter set under the same name**, and merging onto it
            would produce a model whose lineage is a lie.
    """
    name = str(base.get("name") or "")
    recorded = base.get("manifest_sha256")

    if not recorded:
        raise refuse(
            "merge_refused_no_unmerged_base",
            f"This adapter's base ({name!r}) was never saved as a miForge checkpoint, so "
            f"merging would leave no copy of the unmerged base and the merge cannot be "
            f"undone by a second merge. Save a checkpoint of the base first, then merge. "
            f"(A base loaded from a Hub revision is still obtainable but is not a "
            f"checkpoint; this refusal is deliberately strict.)",
            base=name,
        )

    parsed = parse_qualified_name(name)
    if parsed is None:
        raise refuse(
            "merge_refused_no_unmerged_base",
            f"The base name {name!r} is not a checkpoint's qualified name "
            f"(<base_model>:<run_id>:<step>), so the unmerged base cannot be located. "
            f"Refusing rather than searching for something that looks close.",
            base=name,
        )
    _, run_id, step = parsed
    directory = checkpoint_dir(run_id, step, root=root)
    # F5's own constant for the filename, not a second spelling of it - and F8 only ever
    # *reads* a checkpoint manifest: R-02.20 is F5's and a second manifest writer would be
    # a second answer about what a checkpoint claims.
    manifest_path = directory / CHECKPOINT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise refuse(
            "merge_refused_no_unmerged_base",
            f"No checkpoint of the unmerged base is on disk: {manifest_path} is missing. A "
            f"merge is irreversible, so it is refused rather than performed against a base "
            f"that would then exist nowhere. Save a checkpoint of the base first.",
            base=name,
            expected_path=str(manifest_path),
        )

    doc = checkpoint_manifest.read(manifest_path)
    actual = sha256_of(doc)
    if actual != recorded:
        raise refuse(
            "merge_refused_no_unmerged_base",
            f"The checkpoint at {directory} carries the base's name but a different "
            f"manifest ({actual[:12]}... against the {str(recorded)[:12]}... this adapter "
            f"recorded), so it is a different parameter set under the same name. Merging "
            f"onto it would produce a checkpoint whose lineage is a lie.",
            base=name,
            recorded_manifest_sha256=recorded,
            found_manifest_sha256=actual,
        )

    return BaseCheckpoint(name=name, directory=directory, manifest=doc, manifest_sha256=actual)


def apply_adapter(base_module: Any, adapter_directory: Path) -> Any:
    """Return a module with the adapter folded into the base's weights.

    **``peft``'s own merge is used, and that is a deliberate departure from this feature's
    TID**, which sketches ``W + (B @ A) * (alpha / rank)`` per target module by hand. The
    TID's own warning is the argument against its sketch: *"applying the scale twice, or not
    at all, produces a model that loads and answers differently"*, and the equivalence test
    is the only check that would catch it. ``peft``'s ``merge_and_unload`` is the **same code
    path a server applies when it serves the adapter**, so base-plus-adapter and the merged
    checkpoint agree structurally rather than because a hand-written scale happened to
    match. Hand-rolling the arithmetic to own it would put the one silent-failure line in
    this feature into code nothing else exercises.

    The scaling rule is validated **before** this is reached - ``load_manifest`` refuses an
    unrecognised one - so ``peft``'s alpha-over-rank is the rule the manifest declares and
    not an assumption.

    Raises:
        Refusal: ``ml_runtime_unavailable`` when ``peft`` is absent. Refused by name, never
            replaced with a local implementation.
    """
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise refuse(
            "ml_runtime_unavailable",
            "peft is not installed in this process, so an adapter cannot be merged. "
            "miForge will not hand-roll the arithmetic instead: peft's merge is the same "
            "code path a server applies when it serves the adapter, which is what makes "
            "the merged checkpoint and base-plus-adapter agree.",
            detail=str(exc),
        ) from exc

    wrapped = PeftModel.from_pretrained(base_module, str(adapter_directory))
    return wrapped.merge_and_unload()


def merge(
    *,
    session: Any,
    adapter_name: str,
    base_module: Any,
    tokenizer: Any,
    run_id: str,
    step: int,
    domain: str | None,
    signal_source: str,
    optimizer: Mapping[str, Any],
    held_base_name: str | None = None,
    root: Path | None = None,
    emit: Any = None,
) -> MergeOutcome:
    """Merge ``adapter_name`` into a new checkpoint. **An explicitly requested operation.**

    Not a side effect of a save and not an automatic end to an adapter run (FR-008.35):
    *"on request"* is R-02.45's first phrase, and a merge is irreversible.

    Args:
        adapter_name: the adapter's qualified name, as its manifest carries it.
        base_module: the base the adapter names, already in hand.
        run_id, step: where the **merged checkpoint** is written. A merged checkpoint is a
            checkpoint of the run that asked for it.
        held_base_name: the base in hand, when the caller knows it. Checked against the
            adapter's ``base.name`` (E-13): a merge is only defined against the base the
            adapter names.

    Raises:
        Refusal: ``merge_refused_no_unmerged_base`` (AC17), ``adapter_base_mismatch``
            (E-13), ``adapter_scaling_unrecognised`` (E-8), ``adapter_manifest_invalid``,
            ``ml_runtime_unavailable``.
    """
    parsed = adapter_paths.parse_adapter_name(adapter_name)
    if parsed is None:
        raise refuse(
            "adapter_manifest_missing",
            f"{adapter_name!r} is not an adapter's qualified name "
            f"(<base_model>:<run_id>:adapter-<step>), so no adapter directory can be "
            f"resolved from it. Refusing rather than guessing a path.",
            adapter=adapter_name,
        )
    _, adapter_run_id, adapter_step = parsed
    directory = adapter_paths.adapter_dir(adapter_run_id, adapter_step, root=root)

    # ---- 1. the manifest, which refuses an unrecognised scaling rule --------- E-8
    doc = adapter_load.load_manifest(directory / adapter_paths.MANIFEST_NAME)
    if held_base_name is not None:
        adapter_load.assert_base_matches(doc, held_base_name)
    validate_adapter_config(
        {
            "rank": doc["adapter"]["rank"],
            "alpha": doc["adapter"]["alpha"],
            "target_modules": tuple(doc["adapter"]["target_modules"]),
            "dropout": doc["adapter"].get("dropout", 0.0),
        }
    )
    manifest_sha256 = sha256_of(doc)

    # ---- 2. AC17, BEFORE any tensor arithmetic ------------------------------- AC17
    base = resolve_base_checkpoint(doc["base"], root=root)

    log.info(
        "merge admitted",
        extra={
            "code": "merge_admitted",
            "run_id": run_id,
            "adapter": doc["name"],
            "base_checkpoint": base.name,
            "scaling": doc["adapter"].get("scaling"),
        },
    )

    # ---- 3. the arithmetic, through peft's own merge ----------------- FR-008.41
    merged = apply_adapter(base_module, directory)

    # ---- 4. F5's writer, with BOTH ancestors -------------------- FR-008.36, .37
    # `parent` carries the base; `merged_adapter` names the adapter. A merged checkpoint
    # has TWO ancestors, and without the second field two checkpoints at the same step
    # with the same parent - one merged and one not - would be indistinguishable.
    # All three members are written, and the schema requires all three.
    outcome = checkpoint_service.save(
        merged,
        tokenizer,
        session=session,
        fields=ManifestFields(
            run_id=run_id,
            step=step,
            domain=domain,
            signal_source=signal_source,
            base_model=base.base_model,
            parent=base.name,
            optimizer=optimizer,
            merged_adapter={
                "name": doc["name"],
                "step": doc["step"],
                "manifest_sha256": manifest_sha256,
            },
        ),
        root=root,
        emit=emit,
    )

    # The adapter directory is untouched: not consumed, not moved, not deleted.
    if not (directory / adapter_paths.MANIFEST_NAME).is_file():  # pragma: no cover - guard
        raise refuse(
            "adapter_consumed_by_merge",
            "The adapter directory disappeared during its own merge. A merged adapter "
            "cannot be taken off, which is exactly why the separate artifact must survive.",
            adapter=doc["name"],
        )

    log.info(
        "merge written",
        extra={
            "code": "merge_written",
            "run_id": run_id,
            "adapter": doc["name"],
            "base_checkpoint": base.name,
            "result_checkpoint": outcome.name,
            "outcome": OUTCOME_WRITTEN,
        },
    )
    return MergeOutcome(
        outcome=OUTCOME_WRITTEN,
        adapter_name=doc["name"],
        adapter_step=int(doc["step"]),
        adapter_manifest_sha256=manifest_sha256,
        base_checkpoint_name=base.name,
        result_checkpoint_name=outcome.name,
    )
