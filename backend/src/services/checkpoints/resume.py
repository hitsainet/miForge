"""Continue and branch: verify first, then decide, then say what you did  (FR-005.28-38).

**A continue and a branch are the same plan with two differences**: the ``run_id``, and
what ``parent`` becomes. Everything else - resolve, verify, read the lineage root, resolve
trainer state - is one path, which is why they are one function here rather than two that
drift.

## Nothing in this path touches a sibling

R-02.22's own current text is *"start miForge with the same ``run_id`` and the checkpoint to
resume from. **miForge loads the checkpoint itself** (R-02.46 case 1), **verifies its
``weights_sha256`` before use** (R-02.21)"*. Before 2026-10-13 it said *"load the checkpoint
into miLLM (BRD-01 R-01.25)"*, which made crash recovery - acceptance criteria 6 and 20 -
depend on an application that is specced and unimplemented. It was corrected on this
feature's finding, so the standalone form below **is** the requirement rather than a reading
of it.

## Verification comes first, and it refuses having changed nothing

R-02.21 carries it in so many words: *"On its own load path - a continue, a branch, or a
conformance check - miForge verifies the checkpoint's ``weights_sha256`` before using a
single tensor, and refuses on mismatch having changed nothing."* That sentence is **new**,
and it is new because fixing the sibling citation above would otherwise have dropped the
only hash check in the whole contract along with it - a corrupted checkpoint would have
resumed as valid, which is the silently-wrong answer ADR-013 forbids.

**A requirement with no test behind it is a requirement nothing enforces.** Nothing in this
code would notice if the ``verify_weights`` call below were deleted, so the mutation control
that deletes it is what notices, and a green suite under that mutation means the closure is
unpinned.

## Three trainer-state outcomes, and the fourth that is a refusal

* state for **this** step, complete - ``restored``;
* **no** directory for this step - ``fresh``, with a warning naming the directory looked
  for (E-7, R-02.22's own words);
* a directory for **another** step of the same run - ``fresh``, a warning, and the plan
  **lists the steps that do have state** so the operator can continue from one of those
  instead of silently accepting a discontinuity (E-8, AMB-1). A mismatched optimizer is
  never restored onto different weights: an AdamW moment buffer belongs to the parameters
  it was computed against;
* a directory for this step that is **torn** - refused by name through F1's
  :func:`schemas.trainer_state.require_complete`. This is the one that is not a judgement
  call: continuing from a partial state does not crash, it *runs*, and produces a loss
  discontinuity that looks exactly like a real regression, so the time is lost twice.

**A branch is always ``fresh`` by construction**, because trainer state is addressed by
``run_id`` and a branch has a new one (R-02.15, FR-005.38). So the loss discontinuity
R-02.22 exists to prevent is unavoidable on every branch as the requirements stand. That is
recorded and logged; a copy-forward of another run's trainer state is **not** invented here,
because no requirement authorises it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from core.errors import Refusal, refuse
from schemas.trainer_state import STEP_DIR_PREFIX, is_complete, require_complete
from services.checkpoints import manifest as manifest_module
from services.checkpoints import writer as writer_module
from services.checkpoints.paths import (
    checkpoint_dir,
    iter_step_dirs,
    resolve_latest,
    root_holding_run,
)
from training import trainer_state

log = logging.getLogger("miforge.checkpoints.resume")

__all__ = [
    "ResumePlan",
    "ResumeTarget",
    "branch_plan",
    "continue_plan",
    "later_artifacts",
    "manifest_file_sha256",
    "refuse_unless_latest",
    "resolve_resume_from",
    "resume_plan",
    "steps_with_trainer_state",
]

#: ``Final`` so MyPy types each as its ``Literal`` - the optimizer identity's ``state`` field
#: accepts exactly these two, and a plain ``str`` constant could not be passed to it.
RESTORED: Final = "restored"
FRESH: Final = "fresh"


@dataclass(frozen=True)
class ResumePlan:
    """Everything a continue or a branch needs, resolved and verified, before submission.

    Built **before** anything is submitted so a continue refuses before step 1 rather than
    at step *n* (F1's preflight rule, extended): the checkpoint resolves, its manifest is
    valid, its hash verifies, and the trainer-state decision is taken - all without a GPU
    lease, a tensor read or a run row moving.
    """

    #: The run the resumed work belongs to: the same ``run_id`` for a continue, a new one
    #: for a branch.
    run_id: str
    #: The run whose checkpoint is being loaded. Equal to ``run_id`` on a continue.
    source_run_id: str
    step: int
    directory: Path
    manifest: dict[str, Any]
    #: The lineage root, **carried forward** from the loaded checkpoint's own manifest
    #: (FR-005.13). Never the loaded directory's identity.
    base_model: str
    #: The loaded checkpoint's fully qualified name - what the next save records as
    #: ``parent``.
    parent: str
    #: ``restored`` | ``fresh``. Recorded in every subsequent manifest.
    optimizer_state: str
    #: Why, in a word a log line and an API response can both carry.
    reason: str
    #: Steps of this run that do have complete trainer state. Empty for a branch.
    steps_with_state: tuple[int, ...]
    #: ``(recorded, configured)`` when the run configuration's optimizer hash differs from
    #: the checkpoint's. Restored anyway - R-02.22 says restore - with the **new** hash
    #: carried forward and both logged (FR-005.34).
    config_hash_changed: tuple[str | None, str | None] | None
    is_branch: bool

    @property
    def verified_weights_sha256(self) -> str:
        return str(self.manifest["weights_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_run_id": self.source_run_id,
            "step": self.step,
            "directory": str(self.directory),
            "name": self.parent,
            "base_model": self.base_model,
            "parent": self.parent,
            "optimizer": {"state": self.optimizer_state, "reason": self.reason},
            "steps_with_trainer_state": list(self.steps_with_state),
            "config_hash_changed": (
                None
                if self.config_hash_changed is None
                else {
                    "recorded": self.config_hash_changed[0],
                    "configured": self.config_hash_changed[1],
                }
            ),
            "is_branch": self.is_branch,
            "weights_sha256": self.verified_weights_sha256,
        }


def steps_with_trainer_state(run_id: str, *, trainer_state_root: Path | None = None) -> list[int]:
    """Steps of ``run_id`` that have **complete** trainer state, ascending.

    Complete rather than merely present: a torn directory is not a step the operator can
    continue from, so offering it in a refusal's ``details`` would be advice that fails.
    """
    base = trainer_state.run_root(run_id, root=trainer_state_root)
    if not base.is_dir():
        return []
    found: list[int] = []
    for entry in base.iterdir():
        if not entry.is_dir() or not entry.name.startswith(STEP_DIR_PREFIX):
            continue
        suffix = entry.name[len(STEP_DIR_PREFIX) :]
        if suffix.isdigit() and is_complete(entry):
            found.append(int(suffix))
    return sorted(found)


def _trainer_state_decision(
    run_id: str,
    step: int,
    *,
    trainer_state_root: Path | None,
) -> tuple[str, str, tuple[int, ...]]:
    """``(optimizer_state, reason, steps_with_state)`` for a continue. See the module docs."""
    path = trainer_state.step_path(run_id, step, root=trainer_state_root)

    if path.is_dir():
        # Present but possibly torn. `require_complete` is F1's refusal and it is used
        # rather than `is_complete` on purpose: "present and incomplete" is not the same
        # fact as "absent", and treating it as absent would start fresh from a crash whose
        # evidence is right there.
        require_complete(path)
        log.info(
            "trainer state restored",
            extra={
                "code": "trainer_state_restored",
                "run_id": run_id,
                "step": step,
                "path": str(path),
            },
        )
        return RESTORED, "trainer_state_restored", ()

    available = tuple(steps_with_trainer_state(run_id, trainer_state_root=trainer_state_root))
    if not available:
        log.warning(
            "no trainer state for the loaded step; starting a fresh optimizer",
            extra={
                "code": "trainer_state_absent",
                "run_id": run_id,
                "step": step,
                "looked_for": str(path),
            },
        )
        return FRESH, "trainer_state_absent", ()

    # AMB-1: R-02.22 specifies "matching the loaded step" and specifies the absent case; it
    # does not say what this is. Chosen behaviour, stated loudly rather than hidden: proceed
    # as `fresh` - the literal reading of "no optimizer state for this step exists" - log at
    # warning, and NAME THE STEPS THAT DO HAVE STATE so the operator can continue from one
    # of those instead of silently accepting a discontinuity. A refusal is defensible and is
    # a one-line change here; it is reported for the operator, not taken unilaterally.
    log.warning(
        "trainer state exists for other steps of this run; starting a fresh optimizer",
        extra={
            "code": "trainer_state_step_mismatch",
            "run_id": run_id,
            "step": step,
            "looked_for": str(path),
            "steps_with_state": list(available),
        },
    )
    return FRESH, "trainer_state_step_mismatch", available


def resume_plan(
    *,
    run_id: str,
    step: int | None = None,
    new_run_id: str | None = None,
    configured_config_hash: str | None = None,
    root: Path | None = None,
    trainer_state_root: Path | None = None,
) -> ResumePlan:
    """Plan a continue (``new_run_id`` absent) or a branch (``new_run_id`` given).

    Args:
        run_id: the run whose checkpoint is loaded.
        step: which step, or ``None`` for whatever ``latest`` resolves to.
        new_run_id: a **new** run id makes this a branch (R-02.23).
        configured_config_hash: the optimizer ``config_hash`` the resumed run configuration
            produces, when the caller knows it. A difference is restored anyway and logged
            with both values (FR-005.34).

    Raises:
        Refusal: ``checkpoint_not_found``, ``checkpoint_manifest_invalid``,
            ``checkpoint_hash_mismatch``, ``trainer_state_incomplete``,
            ``run_not_continuable``. Each **before** step 1, each having changed nothing.
    """
    is_branch = new_run_id is not None
    if new_run_id is not None and new_run_id == run_id:
        raise refuse(
            "run_not_continuable",
            f"A branch needs a NEW run_id; {run_id!r} is the run being branched from. "
            f"Reusing it would make the branch write into the lineage it descends from.",
            run_id=run_id,
        )

    # D47: an attached run's checkpoints may live under a read-only root (miLLM's).
    root = root_holding_run(run_id, root=root)
    resolved_step = step if step is not None else resolve_latest(run_id, root=root)
    directory = checkpoint_dir(run_id, resolved_step, root=root)
    doc = manifest_module.read(directory)

    # BEFORE A SINGLE TENSOR IS READ. R-02.21. Nothing above this line has opened a weights
    # file, taken a lease or moved a run row, and nothing below runs if this refuses.
    writer_module.verify_weights(directory, str(doc["weights_sha256"]))

    # The lineage root, carried forward from the loaded checkpoint's OWN manifest, through
    # F1's named function. The literal reading - "a load sets base_model to the loaded
    # model's identity" - is right for a Hugging Face repo and wrong here, where the thing
    # loaded IS a checkpoint: it would root the tree at the parent and terminate every walk
    # one hop short, on a tree that validates and looks plausible.
    base_model, parent = manifest_module.resolve_lineage(parent_manifest=doc)
    assert parent is not None  # a loaded checkpoint always names itself as the parent

    optimizer_state: str
    reason: str
    available: tuple[int, ...]
    if new_run_id is not None:
        # By construction, not by policy: trainer state is addressed by run_id (R-02.15) and
        # this run has a new one, so there is nothing of its own to restore. No copy-forward
        # is invented - no requirement authorises one (AMB-2, reported).
        optimizer_state, reason, available = FRESH, "branch_has_no_trainer_state", ()
        log.info(
            "branch starts with a fresh optimizer",
            extra={
                "code": "branch_optimizer_fresh",
                "run_id": new_run_id,
                "branched_from": parent,
                "step": resolved_step,
            },
        )
    else:
        optimizer_state, reason, available = _trainer_state_decision(
            run_id, resolved_step, trainer_state_root=trainer_state_root
        )

    recorded_hash = (doc.get("optimizer") or {}).get("config_hash")
    changed: tuple[str | None, str | None] | None = None
    if configured_config_hash is not None and configured_config_hash != recorded_hash:
        # R-02.22 says restore, so this does not refuse. The NEW configuration's hash is
        # what subsequent manifests carry, and both are logged: a restored AdamW state under
        # a changed learning-rate schedule is a legitimate operator act and an unexplained
        # loss discontinuity if nothing records it - the same reasoning the schema gives for
        # `optimizer.state`.
        changed = (recorded_hash, configured_config_hash)
        log.warning(
            "optimizer configuration changed across the resume",
            extra={
                "code": "optimizer_config_hash_changed",
                "run_id": new_run_id if new_run_id is not None else run_id,
                "step": resolved_step,
                "recorded_config_hash": recorded_hash,
                "configured_config_hash": configured_config_hash,
            },
        )

    return ResumePlan(
        run_id=new_run_id if new_run_id is not None else run_id,
        source_run_id=run_id,
        step=resolved_step,
        directory=directory,
        manifest=doc,
        base_model=base_model,
        parent=parent,
        optimizer_state=optimizer_state,
        reason=reason,
        steps_with_state=available,
        config_hash_changed=changed,
        is_branch=is_branch,
    )


def continue_plan(
    run_id: str,
    *,
    step: int | None = None,
    configured_config_hash: str | None = None,
    root: Path | None = None,
    trainer_state_root: Path | None = None,
) -> ResumePlan:
    """R-02.22: the same ``run_id``, the checkpoint to resume from, trainer state restored."""
    return resume_plan(
        run_id=run_id,
        step=step,
        configured_config_hash=configured_config_hash,
        root=root,
        trainer_state_root=trainer_state_root,
    )


def branch_plan(
    run_id: str,
    step: int,
    *,
    new_run_id: str,
    configured_config_hash: str | None = None,
    root: Path | None = None,
    trainer_state_root: Path | None = None,
) -> ResumePlan:
    """R-02.23: any checkpoint, a **new** ``run_id``, ``parent`` = the branch point."""
    return resume_plan(
        run_id=run_id,
        step=step,
        new_run_id=new_run_id,
        configured_config_hash=configured_config_hash,
        root=root,
        trainer_state_root=trainer_state_root,
    )


# =========================================================================== resume_from
#
# F11 FTASKS 18.0 (D52): an ATTACHED run resumes from a checkpoint it saved. What follows is
# the generic half - which checkpoint is meant, and the facts about it that the served model
# must match. It knows nothing of miLLM: comparing those facts with what the serving process
# reports is ``training/hotupdate/resume.py``'s, inside the fence (ADR-022).


@dataclass(frozen=True)
class ResumeTarget:
    """The checkpoint a ``resume_from`` names, resolved, with the facts it is matched on."""

    run_id: str
    step: int
    directory: Path
    manifest: dict[str, Any]
    #: SHA-256 of the ``manifest.json`` FILE'S BYTES - not :func:`manifest.digest`, which is
    #: a canonical-JSON hash of the parsed document. miLLM's resident ``manifest_hash`` is
    #: the hash of the bytes it read (``weights_checkpoint_service._verify``:
    #: ``_sha256_bytes(manifest_path.read_bytes())``), and the file is serialised by
    #: whichever process wrote it, so only the bytes' hash is comparable.
    manifest_file_sha256: str
    #: The architecture config saved beside the weights, or ``None`` when the directory has
    #: no ``config.json`` (then nothing is compared on it and that is said by name).
    config: dict[str, Any] | None

    @property
    def name(self) -> str:
        """The checkpoint's fully qualified name, as its own manifest states it."""
        return str(self.manifest["name"])

    @property
    def weights_sha256(self) -> str:
        return str(self.manifest["weights_sha256"])


def manifest_file_sha256(directory: Path) -> str:
    """SHA-256 of ``directory/manifest.json``'s bytes. See :class:`ResumeTarget`."""
    import hashlib

    return hashlib.sha256((directory / manifest_module.MANIFEST_NAME).read_bytes()).hexdigest()


def _read_config(directory: Path) -> dict[str, Any] | None:
    import json

    path = directory / "config.json"
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise refuse(
            "checkpoint_config_unreadable",
            f"{path} exists and is not a readable JSON object, so the architecture the "
            f"checkpoint was saved with cannot be compared with the served one. Refusing "
            f"rather than skipping the comparison.",
            path=str(path),
            error=f"{type(exc).__name__}: {exc}",
        ) from exc
    return loaded if isinstance(loaded, dict) else None


def resolve_resume_from(
    resume_from: Any,
    *,
    root: Path | None = None,
) -> ResumeTarget:
    """Resolve a ``resume_from`` (``run_id`` + ``step``, or ``path``) to its checkpoint.

    ``run_id`` + ``step`` resolves through :func:`root_holding_run`, so an attached run's
    checkpoints under a read-only root (D47) are found. A ``path`` must be the canonical
    step directory of the run its own manifest names, under the primary checkpoint root or a
    configured read root - **never an arbitrary directory**, because the thing resumed is a
    run's lineage and a copied-in directory has none here.

    Reads the manifest and the config. **Does not hash the weights** - that is
    :func:`resume_plan`'s, which every caller of this runs first or after.

    Raises:
        Refusal: ``checkpoint_not_found``, ``checkpoint_manifest_invalid``,
            ``resume_from_outside_roots``.
    """
    from core.config import get_settings

    path = getattr(resume_from, "path", None)
    if path is not None:
        directory = Path(path)
        doc = manifest_module.read(directory)
        run_id, step = str(doc["run_id"]), int(doc["step"])
        settings = get_settings()
        roots = (
            [root]
            if root is not None
            else [
                settings.checkpoint_root,
                *settings.checkpoint_read_root_paths,
            ]
        )
        canonical = None
        for candidate in roots:
            try:
                expected = checkpoint_dir(run_id, step, root=candidate)
            except (Refusal, OSError):  # a root that cannot render the path is not it
                continue
            if expected.resolve() == directory.resolve():
                canonical = expected
                break
        if canonical is None:
            raise refuse(
                "resume_from_outside_roots",
                f"{directory} is not run {run_id!r}'s step-{step} directory under the "
                f"checkpoint root or a configured read root. A resume continues a run's "
                f"lineage, so it names a checkpoint where that run's checkpoints live - "
                f"or by run_id and step.",
                path=str(directory),
                run_id=run_id,
                step=step,
                roots=[str(r) for r in roots],
            )
        directory = canonical
    else:
        run_id = str(resume_from.run_id)
        step = int(resume_from.step)
        holding = root_holding_run(run_id, root=root)
        directory = checkpoint_dir(run_id, step, root=holding)
        doc = manifest_module.read(directory)
    return ResumeTarget(
        run_id=run_id,
        step=step,
        directory=directory,
        manifest=doc,
        manifest_file_sha256=manifest_file_sha256(directory),
        config=_read_config(directory),
    )


def later_artifacts(
    run_id: str,
    step: int,
    *,
    root: Path | None = None,
    trainer_state_root: Path | None = None,
) -> dict[str, list[int]]:
    """Checkpoints and trainer states of ``run_id`` at steps AFTER ``step``.

    A continue numbers its steps from ``step + 1`` under the same ``run_id``, so its save
    points land on the same step numbers as these - and both writers refuse to overwrite a
    published step (miLLM ``CHECKPOINT_EXISTS``; trainer state ``artifact_already_exists``).
    """
    holding = root_holding_run(run_id, root=root)
    return {
        "checkpoints": [s for s, _ in iter_step_dirs(run_id, root=holding) if s > step],
        "trainer_states": [
            s
            for s in steps_with_trainer_state(run_id, trainer_state_root=trainer_state_root)
            if s > step
        ],
    }


def refuse_unless_latest(
    run_id: str,
    step: int,
    *,
    root: Path | None = None,
    trainer_state_root: Path | None = None,
) -> None:
    """Refuse a continue of ``run_id`` from a step that is not its latest artifact.

    Raises:
        Refusal: ``resume_not_from_latest``, naming the later steps and the alternative (a
            branch, which takes a new ``run_id`` and so collides with nothing).
    """
    later = later_artifacts(run_id, step, root=root, trainer_state_root=trainer_state_root)
    if later["checkpoints"] or later["trainer_states"]:
        raise refuse(
            "resume_not_from_latest",
            f"Run {run_id!r} has artifacts after step {step} (checkpoints "
            f"{later['checkpoints']}, trainer states {later['trainer_states']}). Continuing "
            f"under the same run_id would reach those step numbers again and every save "
            f"there would be refused as already published. Continue from the latest step, "
            f"or branch from step {step} into a new run_id.",
            run_id=run_id,
            step=step,
            **later,
        )
