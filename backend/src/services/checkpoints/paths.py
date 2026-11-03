"""Where a checkpoint lives, and how its two step forms are rendered  (FR-005.2, FR-005.12).

**One integer, two rendered forms, rendered at the point of use.** BRD-01 R-01.24 fixes it
and ``checkpoint-manifest-v1.json``'s own ``step`` description repeats it: the step is one
canonical integer, a *path* renders it zero-padded to six digits (``step-000500``) so
directories sort lexicographically, and the fully qualified *name* renders it unpadded
(``step-500``). **Nothing anywhere carries the step as a string of its own**, which is why
both renderings are functions here rather than values stored next to the integer.

*The padding rule has already been got wrong once in this repository, in the trainer-state
layout, by applying a rule about one rendered form to the other one. Hence one module.*

## Why every walk here skips dot-prefixed entries

Acceptance criterion 8 requires that no half-written checkpoint be **discoverable** - not
merely that the write be atomic. The temp directory must be a sibling of the destination
(``rename(2)`` is atomic only within one filesystem), so it necessarily sits inside the
tree a listing walks. Without the exclusion AC8 fails while the write is still perfectly
atomic. :func:`core.atomic_dir.is_temp_dir` is the one test for it, imported rather than
re-spelled so the four discovery paths agree.
"""

from __future__ import annotations

from pathlib import Path

from core.atomic_dir import LATEST, is_temp_dir
from core.config import get_settings
from core.errors import refuse
from core.paths import contained_child, single_segment
from schemas.trainer_state import STEP_DIR_PREFIX

#: ``step-`` + six digits. The *path* form. Shared with the trainer-state layout
#: deliberately: R-02.15 requires the two to use the same form and R-01.24 names them in one
#: sentence, so the prefix is imported from the layout that already declares it rather than
#: spelled a second time here.
__all__ = [
    "checkpoint_dir",
    "iter_step_dirs",
    "parse_qualified_name",
    "qualified_name",
    "resolve_latest",
    "root_holding_run",
    "root_of",
    "run_dir",
    "step_dir_name",
    "step_of_dir_name",
]


def root_of(root: Path | None = None) -> Path:
    """The configured checkpoint root, or an override for tests."""
    return root if root is not None else get_settings().checkpoint_root


def root_holding_run(run_id: str, *, root: Path | None = None) -> Path:
    """The checkpoint root that holds ``run_id``'s checkpoints, for a READ.

    An explicit ``root`` wins (tests, and callers that already know). Otherwise the
    configured ``checkpoint_root`` first, then each of ``checkpoint_read_roots`` in order
    (D47: an attached run's checkpoints are written by miLLM under its own directory, which
    the deployment mounts here read-only). The first root with a directory for the run is
    the answer; with none, the primary root - so a "not found" names the place miForge
    itself writes.
    """
    if root is not None:
        return root
    settings = get_settings()
    for candidate in (settings.checkpoint_root, *settings.checkpoint_read_root_paths):
        if run_dir(run_id, root=candidate).is_dir():
            return candidate
    return settings.checkpoint_root


def _safe_run_id(run_id: str) -> str:
    """Refuse a ``run_id`` that would escape the checkpoint root.

    **A traversal attempt is a validation failure, not a sanitised path** (FTDD section 8).
    Sanitising would compose *some* path and operate on it, which is the silently-wrong
    answer ADR-013 forbids; refusing names what happened. The API resolves ``run_id``
    against its row as well - this is the boundary check, so a non-HTTP caller cannot get
    past it either.
    """
    if not run_id:
        raise refuse(
            "checkpoint_run_id_required",
            "A checkpoint belongs to a named run; name the run_id.",
        )
    single_segment(run_id, code="checkpoint_run_id_invalid", what="a checkpoint directory")
    return run_id


def step_dir_name(step: int) -> str:
    """``step-000500`` - the **path** form, rendered from the integer (FR-005.2)."""
    if step < 0:
        raise refuse(
            "checkpoint_step_negative",
            f"A step number must be non-negative; got {step}.",
            step=step,
        )
    return f"{STEP_DIR_PREFIX}{step:06d}"


def step_of_dir_name(name: str) -> int | None:
    """The integer behind a step directory name, or ``None`` if it is not one.

    ``None`` rather than a refusal because the callers are *walks*: a directory that is not
    a step directory is skipped, and refusing would make one stray entry hide every
    checkpoint beside it.
    """
    if not name.startswith(STEP_DIR_PREFIX):
        return None
    suffix = name[len(STEP_DIR_PREFIX) :]
    return int(suffix) if suffix.isdigit() else None


def run_dir(run_id: str, *, root: Path | None = None) -> Path:
    """``<checkpoint_root>/<run_id>/``."""
    base = root_of(root)
    return contained_child(
        base, _safe_run_id(run_id), code="checkpoint_run_id_invalid", what="a checkpoint directory"
    )


def checkpoint_dir(run_id: str, step: int, *, root: Path | None = None) -> Path:
    """``<checkpoint_root>/<run_id>/step-<NNNNNN>/`` (FR-005.2)."""
    # Contained like the run directory: the step arrives from a URL path, and a step
    # directory that is a link out of the run is refused rather than followed.
    return contained_child(
        run_dir(run_id, root=root),
        step_dir_name(step),
        code="checkpoint_step_dir_invalid",
        what="a checkpoint directory",
    )


def qualified_name(base_model: str, run_id: str, step: int) -> str:
    """``<base_model>:<run_id>:step-<step>`` with the step **unpadded** (FR-005.12).

    The schema's own description fixes this format; it is what an API response carries and
    what a ``parent`` edge names. Unpadded here and padded in paths, from the one integer.
    """
    if not base_model:
        raise refuse(
            "checkpoint_base_model_required",
            "A checkpoint's name begins with its lineage root; base_model cannot be empty.",
        )
    return f"{base_model}:{run_id}:{STEP_DIR_PREFIX}{step}"


def parse_qualified_name(name: str) -> tuple[str, str, int] | None:
    """``(base_model, run_id, step)`` from a fully qualified name, or ``None``.

    Split from the **right**, twice: a base model identity may legitimately contain a colon
    (a repo at a revision), while the last two segments are always ``<run_id>`` and
    ``step-<N>``. Splitting from the left would attribute a revision-qualified base model's
    tail to the run.

    ``None`` rather than a refusal is the whole point of this function: the lineage walk
    uses it, and a ``parent`` name it cannot parse terminates the chain as **truncated**
    rather than as the root (E-11). Conflating those two makes a partial tree look complete.
    """
    parts = name.rsplit(":", 2)
    if len(parts) != 3:
        return None
    base_model, run_id, rendered = parts
    step = step_of_dir_name(rendered)
    if not base_model or not run_id or step is None:
        return None
    return base_model, run_id, step


def iter_step_dirs(run_id: str, *, root: Path | None = None) -> list[tuple[int, Path]]:
    """Every published step directory of one run, ascending by step.

    **Skips in-flight writes and the ``latest`` pointer.** See the module docstring: AC8 is
    about discoverability, and this is one of the four paths that has to be blind to a temp
    directory for it to hold.
    """
    base = run_dir(run_id, root=root)
    if not base.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for entry in base.iterdir():
        if not entry.is_dir() or is_temp_dir(entry.name) or entry.name == LATEST:
            continue
        step = step_of_dir_name(entry.name)
        if step is None:
            continue
        found.append((step, entry))
    return sorted(found)


def resolve_latest(run_id: str, *, root: Path | None = None) -> int:
    """The step ``latest`` points at, falling back to the highest published step.

    The symlink is the answer when it exists, because that is what the writer published and
    what a copied directory carries. The fallback exists for a directory tree that arrived
    without its symlink - a ``tar`` that dropped links, or a copy made with ``cp`` rather
    than ``cp -a`` - where refusing would make a perfectly good set of checkpoints
    unreachable while every one of them sits on disk.

    Raises:
        Refusal: ``checkpoint_not_found`` when the run has no published checkpoint.
    """
    base = run_dir(run_id, root=root)
    link = base / LATEST
    if link.is_symlink() or link.exists():
        target = link.resolve()
        step = step_of_dir_name(target.name)
        if step is not None and target.is_dir():
            return step

    steps = iter_step_dirs(run_id, root=root)
    if not steps:
        raise refuse(
            "checkpoint_not_found",
            f"Run {run_id!r} has no saved checkpoint under {base}. Save one, or name a "
            f"different run; a continue needs weights to start from.",
            run_id=run_id,
            searched=str(base),
        )
    return steps[-1][0]
