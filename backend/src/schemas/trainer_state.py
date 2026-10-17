"""The trainer-state directory layout  (foundation task 3.7; R-02.15).

**The sixth shared contract, and it was found in the feature reviews rather than written
with the other five.** It has the exact property PPRD section 12's argument is about -
several features depending on a shape no task defined:

* **F1 writes it** (R-02.15), at every save point;
* **F5 reads it** (R-02.22), to continue a run from a checkpoint;
* **F11 reads it** (R-02.30(b)), to recover after an interrupted commit.

Three readers of one directory, and before this module there were three spellings of what
was in it. It is **not** a vendored JSON schema - nothing outside miForge consumes it - but
it needs one module that writes and reads it, so there is a single definition to point the
three at.

## Two things this layout is not

**There is no ``latest`` pointer.** Checkpoints have one; trainer state does not. A
"behaviour-preserving" refactor of F1's writer onto :func:`core.atomic_dir.atomic_dir` that
left the default on would add a symlink F1 never wrote, which is why the parameter is
explicit at the call site rather than inferred.

**The write is atomic, and that was a defect for weeks.** R-02.20 gained atomicity in the
standalone reframe and **R-02.15 was not touched** - so the requirement that writes the
optimizer state did not require an atomic write, while the one that writes the checkpoint
did. A half-written trainer-state directory is restored as a *valid* optimizer state and
produces exactly the unexplained loss discontinuity R-02.22 exists to prevent, failing
acceptance criterion 4. Hence :func:`is_complete`: a reader must be able to tell a finished
directory from a torn one **without trusting that the writer was atomic**, because the
reader may be running after a crash that the writer never got to handle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.errors import refuse

#: The optimizer's state. ``bitsandbytes`` 8-bit states live here too (ADR-002); the file
#: name does not change with the optimizer's dtype, because the manifest records which
#: optimizer wrote it and a name that encoded the dtype would be a second source of truth.
OPTIMIZER_STATE = "optimizer.pt"

#: The LR scheduler's state. Separate from the optimizer's because a continue may legally
#: resume the optimizer while rebuilding the schedule - a curriculum stage boundary does
#: exactly that, and a single blob would make it impossible.
SCHEDULER_STATE = "scheduler.pt"

#: RNG state for torch, numpy and python. **Without this a continue is not a continue**: a
#: resumed run that reseeds draws a different sample order from the same data and produces
#: a loss discontinuity indistinguishable from a real regression.
RNG_STATE = "rng.pt"

#: This directory's own record: the step, the optimizer identity, the run, the hashes.
#: Written **last**, which is what makes :func:`is_complete` meaningful.
STATE_MANIFEST = "trainer_state.json"

#: The path form, from BRD-01 R-01.24: ``step-`` plus the integer zero-padded to six
#: digits, so directories sort lexicographically. R-02.15 requires the trainer-state path
#: to use the same form as the weights checkpoint, and R-01.24 names both in one sentence.
STEP_DIR_PREFIX = "step-"

#: Required, in the order they are written. The manifest is last on purpose.
REQUIRED_FILES: tuple[str, ...] = (OPTIMIZER_STATE, SCHEDULER_STATE, RNG_STATE, STATE_MANIFEST)


@dataclass(frozen=True)
class TrainerStateDir:
    """One trainer-state directory, named by its step.

    ``step`` is the canonical single integer (BRD-01 R-01.24), and the path form is
    ``step-`` plus that integer **zero-padded to six digits** - ``step-000500`` - so that
    directories sort lexicographically. R-02.15 says the trainer-state path uses *the same
    form as the weights checkpoint*, and R-01.24 names both together.

    **Corrected 2026-10-14, and the error was mine.** This rendered a bare ``000500`` and
    this docstring asserted *"``step-<N>`` appears in no contract and must not be
    reintroduced"*. R-01.24 eliminated the **unpadded** ``step-<N>`` as a *path* form - that
    spelling survives only in the fully qualified display name ``<base>:<run>:step-500`` -
    and it states in as many words that checkpoint **and trainer-state** paths carry the
    ``step-`` prefix. I applied a rule about one rendered form to a different one, wrote the
    conclusion into a gate contract as settled fact, and F1 built against it exactly as its
    brief told it to. ``latest_complete`` compounded it by filtering on ``name.isdigit()``,
    which no conforming directory would ever have matched.

    *The lesson is the project's own, and it is the one written at the top of AGENT.md:
    verify that a cited requirement says what you claim before building on it. The claim was
    checkable in one grep, by the person asserting it, and it was asserted instead.*
    """

    root: Path
    step: int

    @property
    def path(self) -> Path:
        return self.root / f"{STEP_DIR_PREFIX}{self.step:06d}"

    def file(self, name: str) -> Path:
        if name not in REQUIRED_FILES:
            raise refuse(
                "trainer_state_file_unknown",
                f"{name!r} is not part of the trainer-state layout. Refusing rather than "
                f"accepting it: a fourth file that only one reader knows about is how three "
                f"readers end up with three layouts. Known: {list(REQUIRED_FILES)}.",
                name=name,
                known=list(REQUIRED_FILES),
            )
        return self.path / name


def is_complete(path: Path) -> bool:
    """Whether ``path`` holds a **complete** trainer state.

    Checks for every required file with the manifest last, so a directory that lost power
    mid-write reads as incomplete rather than as a valid state missing its RNG. A reader
    must never have to trust that the writer was atomic: the writer may have been killed
    before its own cleanup ran, and ``is_complete`` is the only thing standing between that
    and a continue that silently reseeds.
    """
    if not path.is_dir():
        return False
    return all((path / name).is_file() for name in REQUIRED_FILES)


def require_complete(path: Path) -> Path:
    """Return ``path`` when complete, else refuse by name.

    **Refuses rather than continuing from a partial state.** The failure mode this prevents
    is the expensive one: a continue from a torn directory does not crash - it runs, and
    produces a loss discontinuity that looks exactly like a real regression, so the time is
    lost twice, once to the run and once to diagnosing it.
    """
    if not path.is_dir():
        raise refuse(
            "trainer_state_absent",
            f"No trainer state at {path}. A continue needs the optimizer, scheduler and RNG "
            f"state; starting fresh instead would change the sample order and show as an "
            f"unexplained loss discontinuity.",
            path=str(path),
        )
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise refuse(
            "trainer_state_incomplete",
            f"The trainer state at {path} is missing {missing}. Refusing to continue from a "
            f"partial state: it would restore as a valid optimizer state and produce the "
            f"unexplained loss discontinuity R-02.22 exists to prevent. Roll back to the "
            f"previous checkpoint.",
            path=str(path),
            missing=missing,
            expected=list(REQUIRED_FILES),
        )
    return path


def latest_complete(root: Path) -> TrainerStateDir | None:
    """The highest-numbered **complete** trainer state under ``root``, or ``None``.

    Skips incomplete directories rather than stopping at them: a crash leaves the newest
    directory torn, and the useful answer is the newest *usable* one. Also skips in-flight
    writes via :func:`core.atomic_dir.is_temp_dir`, so a save happening right now is never
    mistaken for a candidate.
    """
    from core.atomic_dir import is_temp_dir

    if not root.is_dir():
        return None
    best: TrainerStateDir | None = None
    for entry in root.iterdir():
        if not entry.is_dir() or is_temp_dir(entry.name):
            continue
        # Match the real path form. `name.isdigit()` matched a bare `000500`, which is not
        # what any conforming writer produces - so this loop silently found nothing and a
        # continue would have started fresh while a complete state sat beside it.
        if not entry.name.startswith(STEP_DIR_PREFIX):
            continue
        suffix = entry.name[len(STEP_DIR_PREFIX) :]
        if not suffix.isdigit():
            continue
        if not is_complete(entry):
            continue
        candidate = TrainerStateDir(root=root, step=int(suffix))
        if best is None or candidate.step > best.step:
            best = candidate
    return best
