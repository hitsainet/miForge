"""Checkpointing the optimizer's own state  (R-02.15; FR-001.28 - FR-001.31).

F1 is the **single writer** of foundation task 3.7's layout; F5 reads it to continue a run
(R-02.22) and F11 reads it to recover after an interrupted commit (R-02.30(b)). Three
readers of one directory, so the layout has exactly one definition -
:mod:`schemas.trainer_state` - and this module is its writer. **It invents nothing**:
``REQUIRED_FILES``, ``TrainerStateDir``, ``is_complete``, ``require_complete`` and
``latest_complete`` all come from there.

## The write order has one definition, and it is not here

:func:`core.atomic_dir.atomic_dir` is the backend's only place a directory is renamed into
place - ``fsync`` every file, ``fsync`` the temp directory, ``os.rename`` - and all three
features that write artifacts call it. This module **calls it with
``latest_link=False``**, because the trainer-state layout carries no ``latest`` pointer:
checkpoints have one, trainer state does not, and a "behaviour-preserving" refactor that
left the default on would add a symlink F1 never wrote. The parameter is passed explicitly
at the call site for exactly that reason.

*F1's own documents say it writes the order itself and that F5 refactors it onto the
primitive later. That is stale: the primitive is in the foundation and is imported here. A
second implementation of this guarantee would be two answers (ADR-005).*

## Why atomicity had to be stated, and the failure it prevents

R-02.20 gained atomicity explicitly in the standalone reframe and **R-02.15 was not
touched** - so the requirement that writes the optimizer state did not require an atomic
write while the one that writes the weights did. The failure is the expensive kind: a
half-written trainer-state directory restores as a **valid** optimizer state and produces
exactly the unexplained loss discontinuity R-02.22 exists to prevent, failing acceptance
criterion 4. It does not crash. The time is lost twice, once to the run and once to
diagnosing it. R-02.15 was amended on 2026-10-13 to require it in terms.

## Never under the weights checkpoint root

A checkpoint directory that gains optimizer tensors stops being a standard Hugging Face
directory, which breaks R-02.51 and acceptance criterion 23 - it could then not be loaded
by ``transformers``, vLLM, TGI or miLLM's own loader. :func:`assert_outside_checkpoint_root`
asserts it rather than a comment requesting it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.atomic_dir import atomic_dir, is_temp_dir
from core.canonical import write_canonical
from core.config import get_settings
from core.errors import refuse
from schemas.trainer_state import (
    OPTIMIZER_STATE,
    REQUIRED_FILES,
    RNG_STATE,
    SCHEDULER_STATE,
    STATE_MANIFEST,
    TrainerStateDir,
    is_complete,
    latest_complete,
    require_complete,
)
from training.optimizer import OptimizerIdentity

log = logging.getLogger(__name__)

__all__ = [
    "REQUIRED_FILES",
    "TrainerStateDir",
    "assert_outside_checkpoint_root",
    "clean_stale_temp_dirs",
    "is_complete",
    "latest_complete",
    "read",
    "require_complete",
    "run_root",
    "step_path",
    "write",
]


def run_root(run_id: str, *, root: Path | None = None) -> Path:
    """``<trainer_state_root>/<run_id>/`` - one directory per run.

    Per run rather than per step-under-one-root so that deleting a run's artifacts is a
    directory removal and `latest_complete` has a root to scan that contains only this
    run's states.
    """
    if not run_id:
        raise refuse(
            "trainer_state_run_id_required",
            "Trainer state must belong to a named run: it is restored by run_id and step, "
            "and an unattributed directory cannot be matched to the checkpoint it belongs "
            "with.",
        )
    base = root if root is not None else get_settings().trainer_state_root
    return base / run_id


def step_path(run_id: str, step: int, *, root: Path | None = None) -> Path:
    """The directory for one step's trainer state.

    **Rendered from the canonical integer at the point of use**, through
    :class:`schemas.trainer_state.TrainerStateDir` so the padding rule has one
    implementation. BRD-01 R-01.24: the step is one integer, paths render it zero-padded to
    six digits, the qualified name renders it unpadded, and nothing carries the step as a
    string of its own. Dropping the padding breaks the lexicographic ordering every reader
    relies on and is a mutation target.
    """
    if step < 0:
        raise refuse(
            "trainer_state_step_negative",
            f"A step number must be non-negative; got {step}.",
            step=step,
        )
    return TrainerStateDir(root=run_root(run_id, root=root), step=step).path


def assert_outside_checkpoint_root(path: Path, *, checkpoint_root: Path | None = None) -> None:
    """Refuse a trainer-state path that would land inside the weights checkpoint root.

    FR-001.30. **Asserted, not requested by a comment**: a checkpoint directory that gains
    optimizer tensors is no longer a standard Hugging Face directory, so it stops being
    loadable by any standard server - which breaks R-02.51 and acceptance criterion 23
    without breaking anything that a test of the *save* path would notice.
    """
    root = checkpoint_root if checkpoint_root is not None else get_settings().checkpoint_root
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return
    raise refuse(
        "trainer_state_inside_checkpoint_root",
        f"{path} is inside the weights checkpoint root {root}. Optimizer state is never "
        f"written into a checkpoint directory: a checkpoint that gains optimizer "
        f"tensors stops being a standard Hugging Face directory and can no longer be loaded "
        f"by transformers, vLLM, TGI or miLLM's loader. Point TRAINER_STATE_ROOT elsewhere.",
        path=str(path),
        checkpoint_root=str(root),
    )


def write(
    *,
    run_id: str,
    step: int,
    optimizer: Any,
    scheduler: Any = None,
    identity: OptimizerIdentity | None = None,
    root: Path | None = None,
    checkpoint_root: Path | None = None,
) -> Path:
    """Write one step's trainer state **atomically**. Returns the published directory.

    Guarantee: **absent or complete.** Nothing in between is ever visible, and a reader
    checks :func:`schemas.trainer_state.is_complete` anyway rather than trusting that the
    writer was atomic - the writer may have been killed before its own cleanup ran.

    Args:
        run_id: the owning run.
        step: the canonical integer step. The **same** step number as the weights save it
            accompanies (FR-001.29), so a continue can match the two.
        optimizer: the live optimizer. Its ``state_dict`` is what is saved.
        scheduler: the LR scheduler, when there is one. Saved **separately** from the
            optimizer because a continue may legally resume the optimizer while rebuilding
            the schedule - a curriculum stage boundary does exactly that, and one blob would
            make it impossible.
        identity: the optimizer identity, recorded in the directory's own manifest so a
            reader can tell which optimizer wrote these states before loading them.
        root: override the trainer-state root. Tests use it.
        checkpoint_root: override the weights root for the containment check.

    Raises:
        Refusal: ``trainer_state_inside_checkpoint_root``; ``artifact_already_exists`` from
            the primitive when this step's state is already published - a save never
            overwrites one, because the step is part of the identity and an overwrite means
            two different optimizer states claiming one name.
    """
    torch = _torch()
    destination = step_path(run_id, step, root=root)
    assert_outside_checkpoint_root(destination, checkpoint_root=checkpoint_root)

    # `latest_link=False`: the trainer-state layout has NO `latest` pointer. Passed
    # explicitly at the call site rather than relying on a default, because the default is
    # True for checkpoints and adapters and a silent inheritance would publish a symlink
    # this layout does not define (schemas/trainer_state.py says so in terms).
    with atomic_dir(destination, latest_link=False) as staging:
        torch.save(optimizer.state_dict(), staging / OPTIMIZER_STATE)
        # Written even when there is no scheduler. `REQUIRED_FILES` lists four files and
        # `is_complete` requires all four, so an omitted scheduler file would make every
        # state this writer produces read as incomplete - the reader cannot distinguish
        # "no scheduler" from "the write was cut short", which is the whole point of the
        # completeness check.
        torch.save({} if scheduler is None else scheduler.state_dict(), staging / SCHEDULER_STATE)
        torch.save(_rng_state(torch), staging / RNG_STATE)
        # The manifest is written LAST, which is what makes `is_complete` meaningful: a
        # directory that lost power mid-write reads as incomplete rather than as a valid
        # state missing its RNG.
        write_canonical(
            staging / STATE_MANIFEST,
            {
                "run_id": run_id,
                "step": step,
                "optimizer": None if identity is None else identity.as_manifest_block(),
                "has_scheduler": scheduler is not None,
                "files": list(REQUIRED_FILES),
            },
        )

    log.info(
        "trainer state written",
        extra={
            "code": "trainer_state_written",
            "run_id": run_id,
            "step": step,
            "path": str(destination),
        },
    )
    return destination


def _rng_state(torch: Any) -> dict[str, Any]:
    """RNG state for torch, numpy and python.

    **Without this a continue is not a continue.** A resumed run that reseeds draws a
    different sample order from the same data and produces a loss discontinuity
    indistinguishable from a real regression - which is the failure R-02.22 exists to
    prevent, arriving by a different route than a missing optimizer.

    CUDA state is included only when a device is present, and its absence is recorded as a
    fact rather than left implicit, so a state captured on the GPU node and restored on a
    laptop says what it is missing.
    """
    import random

    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "cuda": None,
        "numpy": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy

        state["numpy"] = numpy.random.get_state()
    except ImportError:
        pass
    return state


def read(run_id: str, step: int, *, root: Path | None = None) -> dict[str, Any]:
    """Load one step's trainer state, refusing an incomplete directory **by name**.

    F5 (R-02.22) and F11 (R-02.30(b)) read through here rather than re-spelling the layout.
    :func:`schemas.trainer_state.require_complete` is what refuses: continuing from a torn
    directory does not crash - it runs, and produces a loss discontinuity that looks exactly
    like a real regression.
    """
    torch = _torch()
    path = require_complete(step_path(run_id, step, root=root))
    # `weights_only=False`: these are optimizer, scheduler and RNG states written by this
    # process, not an untrusted download. The Hub path is where untrusted pickles would come
    # from and it never reaches here - that load is `safetensors` only, by refusal.
    return {
        "optimizer": torch.load(path / OPTIMIZER_STATE, map_location="cpu", weights_only=False),
        "scheduler": torch.load(path / SCHEDULER_STATE, map_location="cpu", weights_only=False),
        "rng": torch.load(path / RNG_STATE, map_location="cpu", weights_only=False),
    }


def clean_stale_temp_dirs(run_id: str, *, root: Path | None = None) -> list[Path]:
    """Remove in-flight write directories left by a killed writer. Returns what was removed.

    The primitive cleans up its own temp directory when it raises, but not when the process
    is killed outright - E-10's case. Called before a write rather than on a schedule, so a
    run's own next save tidies the wreck of its last one, and so nothing sweeps a directory
    that another process is currently writing into.
    """
    base = run_root(run_id, root=root)
    if not base.is_dir():
        return []
    import shutil

    removed: list[Path] = []
    for entry in base.iterdir():
        if entry.is_dir() and is_temp_dir(entry.name):
            shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry)
    if removed:
        log.info(
            "removed stale trainer-state temp directories",
            extra={
                "code": "trainer_state_temp_cleaned",
                "run_id": run_id,
                "removed": [str(p) for p in removed],
            },
        )
    return removed


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise refuse(
            "ml_runtime_unavailable",
            "torch is not installed in this process, so optimizer state cannot be saved or "
            "loaded. Everything in miForge except a training step runs without it.",
            detail=str(exc),
        ) from exc
    return torch
