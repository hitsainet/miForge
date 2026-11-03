"""The one place a directory is renamed into place  (R-02.20, R-02.15, ADR-015).

**Three features call this and none reimplements it**: F5's checkpoint writer, F1's
trainer-state writer, F8's adapter writer. ADR-005's rule applied to a guarantee rather
than to a hash - *two implementations of this guarantee are two answers.*

**This primitive is what replaces the entire torn-weights apparatus.** Before the 2026-10-13
standalone reframe, a parameter update happened inside another process's GPU memory, where
atomicity had to be built by hand out of a commit bracket, a watchdog, a ``torn`` state and
two recovery paths - and every hard correctness problem in this project lived there.
Standalone, the update lands on a filesystem, where atomicity is free: temp dir, ``fsync``,
``rename``. **A torn parameter set cannot occur.** That is the whole dividend of the
reframe, and it is collected here.

## The order IS the requirement

``fsync`` every file -> ``fsync`` the temp directory -> ``os.rename`` -> swap ``latest``.
Not "roughly this order": the outcome is **identical** under every ordering until the power
fails, which is exactly why the test asserts the observed syscall order rather than the
result. A test that checked only the result would pass against a writer that renames first.

## A clean exit is the commit

There is no ``commit()`` to call. Leaving the ``with`` block normally publishes; raising
discards. The safe path is the default, and that is deliberate - all three call sites were
written assuming it, and under the other reading each of their writers would never publish
anything.

*Scheduling note: this module is assigned to F5 in the task lists and is built here, in the
foundation, on purpose. F1 is a wave earlier than F5 and imports it, so leaving it to F5
means F1 writes a second atomicity implementation that F5 then refactors away - the drift
this file exists to prevent, scheduled in. F5's task 2.1 becomes "confirm, do not rewrite".*
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from uuid import uuid4

from core.errors import refuse

#: Temp directories are dot-prefixed so **every discovery path can exclude them** - the
#: listing API, the index upsert, the ``reindex`` walk and ``latest`` resolution.
#: Atomicity alone does not satisfy AC8: the BRD requires that no partial checkpoint be
#: *discoverable*, and the temp directory must be a sibling of the destination (below), so
#: it necessarily sits inside the walked tree. Without this prefix AC8 fails while the
#: write is still perfectly atomic.
#: Modes for a PUBLISHED artifact, set explicitly at the commit rather than left to the
#: umask. **This was a real cross-repo hazard, not tidiness.** `tmp.mkdir()` took no mode, so
#: a published checkpoint's permissions were `0o777 & ~umask`: fine at the common 022, and
#: `0o750` or `0o700` under 027 or 077 — and **miForge's backend runs as uid 10001 while
#: miLLM's runs as uid 1000** (`gitops/pilot/backend.yaml` against their
#: `chown -R 1000:1000`). So cross-reading a checkpoint worked *by umask accident*, and the
#: failure would have arrived as miLLM reporting a checkpoint it can see and cannot open,
#: on a different machine from the one that wrote it.
#:
#: It is also a requirement rather than a preference: R-02.21 says every saved checkpoint is
#: a loadable model and R-02.51 says the output is loadable by **any** inference server. An
#: artifact the serving uid cannot read satisfies neither, and the directory layout being
#: standard is no help if the bits are wrong.
#:
#: Group and other get READ, never write: another process must load an artifact and must not
#: be able to alter one. Nothing written here is secret — the attachment id is deliberately
#: never persisted to disk (migration `0017`), and the batch record carries no prompt text
#: (ADR-020) — so there is nothing these bits would expose that the artifact is not for.
ARTIFACT_DIR_MODE = 0o755
ARTIFACT_FILE_MODE = 0o644

TMP_PREFIX = ".tmp-save-"

#: The pointer to the newest complete artifact. A **relative** symlink: ADR-004 promises a
#: copied checkpoint keeps its meaning, and an absolute symlink silently resolves to the
#: *source* host's path if that path happens to exist on the destination.
LATEST = "latest"


def _same_filesystem(left: Path, right: Path) -> bool:
    """Whether two existing directories share a device.

    ``rename(2)`` is atomic only within one filesystem, and ``shutil.move`` silently
    degrades to a copy across them - so the guarantee would be void while every test
    still passed.
    """
    return left.stat().st_dev == right.stat().st_dev


def verify_root_is_atomic(root: Path) -> Path:
    """Prove **at startup** that ``root`` can be written atomically, or refuse by name.

    **FR-005.5, and it is a startup check on purpose.** :func:`atomic_dir` already refuses
    ``checkpoint_root_not_atomic`` at **save** time, which is correct and is not enough: a
    save point arrives after hours of training, so a misconfigured root is discovered at the
    moment the first checkpoint would have been written and the run has nothing to show for
    the time. *The refusal was reachable and the discovery was late.*

    The check is the real operation rather than a prediction of it: a sibling temporary
    directory is created beside ``root`` and its device compared, which is exactly what
    :func:`atomic_dir` does at every save. Asking ``os.statvfs`` or parsing ``/proc/mounts``
    would be a second implementation of the question, and the two would disagree first on a
    bind mount - which is how this deployment mounts every one of its roots.

    Args:
        root: the configured artifact root.

    Returns:
        ``root``, so a caller can use it in an expression.

    Raises:
        Refusal: ``checkpoint_root_not_atomic`` naming **both** paths, because an operator
            reading only one of them cannot tell which of the two to move; or
            ``checkpoint_root_unwritable`` when the probe cannot be created at all - which
            is a different fault with a different fix, and collapsing the two would send
            someone to repartition a disk over a `chown`.
    """
    probe = root / f".atomicity-probe-{os.getpid()}"
    try:
        # The mkdir is INSIDE the guarded block, which it was not when this was first
        # written - and the omission reproduced, within the hour, the exact defect being
        # fixed in `packs.store.packs_root` at the same moment: a check that creates, whose
        # creation escapes as a raw `OSError` that no caller handles. It surfaced as 38
        # collection errors (`PermissionError: /data/checkpoints`) the first time the suite
        # ran against a default root this user cannot write. *The lesson landed on the
        # person writing it down.*
        root.mkdir(parents=True, exist_ok=True)
        probe.mkdir()
    except OSError as exc:
        raise refuse(
            "checkpoint_root_unwritable",
            f"The artifact root {root} cannot be written to by this process "
            f"({type(exc).__name__}), so no checkpoint could be published there. Checked at "
            f"startup rather than at the first save point, which arrives hours into a run.",
            root=str(root),
            error=type(exc).__name__,
        ) from exc
    try:
        if not _same_filesystem(probe, root):
            raise refuse(
                "checkpoint_root_not_atomic",
                f"A directory created inside {root} does not share a filesystem with it, so "
                f"the temp-dir/fsync/rename publish cannot be atomic and `shutil.move` would "
                f"silently degrade to a copy - leaving a half-written checkpoint that reads "
                f"as a whole one. Put the root and its temporary area on one filesystem.",
                root=str(root),
                probe=str(probe),
            )
    finally:
        with suppress(OSError):
            probe.rmdir()
    return root


@contextmanager
def atomic_dir(
    dest: Path,
    *,
    latest_link: bool = True,
    tmp_parent: Path | None = None,
) -> Iterator[Path]:
    """Yield a temp directory that becomes ``dest`` on a clean exit of the block.

    Args:
        dest: Final directory. Must not already exist.
        latest_link: Whether to point ``dest.parent / "latest"`` at the result.
            **``False`` for the trainer-state write**, which carries no ``latest``
            pointer (R-02.15, and F1's layout): a "behaviour-preserving" refactor that
            left the default on would add a symlink F1 never wrote. Adapter and
            checkpoint directories take the default and do get one.
        tmp_parent: Where the temp directory lives. ``None`` means a **sibling of
            ``dest``**, which is the only shape production ever uses (and the only shape
            that keeps the rename atomic).

            This parameter exists so the cross-filesystem refusal below is **reachable**.
            With the parent hardcoded to ``dest.parent`` the guard compares a path with
            itself, is always true, and the refusal is dead code with a test that cannot
            be written from outside the function. Do **not** plumb it to configuration
            without re-reading the sibling requirement: a configurable temp area is how
            the cross-filesystem case would arrive for real.

    Raises:
        Refusal: ``checkpoint_root_not_atomic`` if the temp parent and the destination are
            on different filesystems; ``artifact_already_exists`` if ``dest`` is present.
    """
    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)

    source_parent = tmp_parent if tmp_parent is not None else parent
    if not source_parent.is_dir():
        raise refuse(
            "temp_parent_missing",
            f"The temporary directory's parent {source_parent} does not exist, so the "
            f"write cannot be made atomic. Refusing before writing anything.",
            tmp_parent=str(source_parent),
        )
    if not _same_filesystem(source_parent, parent):
        raise refuse(
            "checkpoint_root_not_atomic",
            f"The temporary directory ({source_parent}) and the destination ({parent}) are "
            f"on different filesystems. rename(2) is atomic only within one, and "
            f"shutil.move degrades to a copy across them - so the atomicity guarantee "
            f"would be void while every test still passed. Put them on one filesystem.",
            tmp_parent=str(source_parent),
            dest_parent=str(parent),
        )
    if dest.exists():
        raise refuse(
            "artifact_already_exists",
            f"{dest} already exists. A save never overwrites a published artifact: the "
            f"step is part of the identity, so an overwrite means two different parameter "
            f"sets claiming one name.",
            dest=str(dest),
        )

    tmp = source_parent / f"{TMP_PREFIX}{uuid4().hex}"
    # 0o700 WHILE IT IS A TEMP DIRECTORY, widened to ARTIFACT_DIR_MODE at the commit below.
    # A half-written checkpoint should not be readable by another process that might mistake
    # it for a published one; a finished one must be. `mkdir(mode=...)` is masked by the
    # umask, so the mode is set explicitly at the commit rather than trusted here.
    tmp.mkdir(mode=0o700)
    try:
        yield tmp

        # ---- the commit, in the order that IS the requirement ----
        # The same walk sets the modes, because an artifact nobody else can read does not
        # satisfy R-02.21 or R-02.51 - see ARTIFACT_DIR_MODE.
        for path in sorted(tmp.rglob("*")):
            if path.is_file():
                os.chmod(path, ARTIFACT_FILE_MODE)
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            elif path.is_dir():
                os.chmod(path, ARTIFACT_DIR_MODE)
        # The root last, so nothing is reachable through it until its contents are set.
        os.chmod(tmp, ARTIFACT_DIR_MODE)
        _fsync_dir(tmp)
        os.rename(tmp, dest)
        _fsync_dir(parent)

        if latest_link:
            _swap_latest(parent, dest.name)
    except BaseException:
        # Any exception discards. `BaseException` on purpose: cooperative cancellation
        # (ADR-007) raises a BaseException subclass at a step boundary, and a cancelled
        # save must leave nothing half-published rather than being treated as a commit.
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _fsync_dir(path: Path) -> None:
    """``fsync`` a directory, so the rename itself survives a power loss.

    Dropping this is the mutation that no outcome-based test can catch: the directory
    entry is visible immediately either way and only an untimely power loss tells the
    difference.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _swap_latest(parent: Path, target_name: str) -> None:
    """Point ``parent/latest`` at ``target_name`` atomically, with a **relative** target."""
    link = parent / LATEST
    staging = parent / f".{LATEST}.tmp-{uuid4().hex}"
    os.symlink(target_name, staging)  # relative: just the directory name
    os.replace(staging, link)
    _fsync_dir(parent)


def is_temp_dir(name: str) -> bool:
    """Whether a directory entry is an in-flight write.

    **Every discovery path must call this**: the listing API, the index upsert, the
    ``reindex`` walk and ``latest`` resolution. Exported so those four agree, rather than
    each testing the prefix itself and one of them eventually not.
    """
    return name.startswith(TMP_PREFIX) or name.startswith(f".{LATEST}.tmp-")
