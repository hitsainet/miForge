"""The safe walk of ``/out`` (FR-016.10 step 12, FR-016.19; R1B-M2, R2A-15).

The child wrote ``/out`` and the child is hostile, so nothing in it is followed or trusted:

- every directory is opened ``O_DIRECTORY | O_NOFOLLOW`` relative to its parent's descriptor
  (``openat``), so a symlinked directory, or one swapped for a symlink mid-walk, is never entered;
- every entry is ``lstat``-ed first: only a regular file with ``st_nlink == 1`` is read (a symlink,
  FIFO, socket, device or hard link is recorded, never opened), and it is opened
  ``O_NOFOLLOW | O_NONBLOCK`` and ``fstat``-ed again, so a swap between the two is caught;
- each file and the total are capped, depth and entry counts are bounded;
- a ``supervisor.json`` at the top is itself unsafe: only the supervisor writes that name, into
  the result, never from ``/out`` (R2A-15).

Every refusal becomes one ``worker_result_unsafe`` reason in the result's ``supervisor.json``; the
walk itself never raises on what it finds.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field

FILE_MAX_BYTES = 64 * 1_048_576
TOTAL_MAX_BYTES = 256 * 1_048_576
MAX_DEPTH = 4
MAX_ENTRIES = 4096
RESERVED_TOP = ("supervisor.json",)


@dataclass
class Walk:
    """What the walk kept (relative path -> bytes) and why it refused the rest."""

    files: dict[str, bytes] = field(default_factory=dict)
    unsafe: list[str] = field(default_factory=list)
    total: int = 0
    entries: int = 0


def _kind(mode: int) -> str:
    for test, name in (
        (stat.S_ISLNK, "symlink"),
        (stat.S_ISFIFO, "fifo"),
        (stat.S_ISSOCK, "socket"),
        (stat.S_ISCHR, "character device"),
        (stat.S_ISBLK, "block device"),
    ):
        if test(mode):
            return name
    return "special file"


def _read_file(dir_fd: int, name: str, rel: str, walk: Walk) -> None:
    try:
        st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except OSError as exc:
        walk.unsafe.append(f"{rel}: cannot stat ({exc.strerror})")
        return
    if not stat.S_ISREG(st.st_mode):
        walk.unsafe.append(f"{rel}: a {_kind(st.st_mode)}, not a regular file")
        return
    if st.st_nlink != 1:
        walk.unsafe.append(f"{rel}: a hard link ({st.st_nlink} links)")
        return
    if st.st_size > FILE_MAX_BYTES:
        walk.unsafe.append(f"{rel}: {st.st_size} bytes, over the {FILE_MAX_BYTES}-byte cap")
        return
    try:
        fd = os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=dir_fd
        )
    except OSError as exc:
        walk.unsafe.append(f"{rel}: cannot open ({exc.strerror})")
        return
    try:
        again = os.fstat(fd)
        if (
            not stat.S_ISREG(again.st_mode)
            or again.st_nlink != 1
            or (again.st_dev, again.st_ino) != (st.st_dev, st.st_ino)
        ):
            walk.unsafe.append(f"{rel}: changed between the check and the open")
            return
        chunks: list[bytes] = []
        size = 0
        while size <= FILE_MAX_BYTES:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        if size > FILE_MAX_BYTES:
            walk.unsafe.append(f"{rel}: grew past the {FILE_MAX_BYTES}-byte cap while read")
            return
        if walk.total + size > TOTAL_MAX_BYTES:
            walk.unsafe.append(f"{rel}: the result would pass the {TOTAL_MAX_BYTES}-byte total")
            return
        walk.total += size
        walk.files[rel] = b"".join(chunks)
    finally:
        os.close(fd)


def _walk_dir(dir_fd: int, prefix: str, depth: int, walk: Walk) -> None:
    for name in sorted(os.listdir(dir_fd)):
        walk.entries += 1
        rel = f"{prefix}{name}"
        if walk.entries > MAX_ENTRIES:
            walk.unsafe.append(f"more than {MAX_ENTRIES} entries; the rest were not read")
            return
        if not prefix and name in RESERVED_TOP:
            walk.unsafe.append(f"{rel}: a name only the supervisor writes")
            continue
        try:
            st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except OSError as exc:
            walk.unsafe.append(f"{rel}: cannot stat ({exc.strerror})")
            continue
        if stat.S_ISDIR(st.st_mode):
            if depth >= MAX_DEPTH:
                walk.unsafe.append(f"{rel}/: deeper than {MAX_DEPTH} levels")
                continue
            try:
                child = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd
                )
            except OSError as exc:
                walk.unsafe.append(f"{rel}/: cannot open ({exc.strerror})")
                continue
            try:
                _walk_dir(child, f"{rel}/", depth + 1, walk)
            finally:
                os.close(child)
        else:
            _read_file(dir_fd, name, rel, walk)


def walk(root: str) -> Walk:
    """Read ``root`` safely. ``root`` itself is the supervisor's own mount (or its local twin)."""
    result = Walk()
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        result.unsafe.append(f"/out cannot be opened ({exc.strerror})")
        return result
    try:
        _walk_dir(fd, "", 0, result)
    except OSError as exc:
        result.unsafe.append(f"the walk stopped: {exc.strerror}")
    finally:
        os.close(fd)
    return result
