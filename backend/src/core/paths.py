"""Path containment: the one check that a caller-supplied name stays under its root.

A ``run_id`` reaches the filesystem from a URL path (``/runs/{run_id}/boundaries``), from a
submitted run definition and from a task argument. Every one of those composes
``<root>/<run_id>/...`` and then reads, appends or creates. So the identifier is checked
**here, once**, rather than at each caller - a check per caller is the shape in which one
caller is forgotten.

Two layers, because each misses something the other catches:

1. **Lexical** - the name must be one path segment: no separator, not ``.`` or ``..``, no
   NUL, not empty. That refuses ``../x`` before anything touches the disk.
2. **Resolved** - ``(root / name).resolve()`` must still lie under ``root.resolve()``. That
   refuses a *symlink* planted at ``<root>/<name>`` pointing outside the root, which is a
   single legal segment and passes layer 1.

**A traversal is refused, never sanitised** (ADR-013): a sanitised path is a path the caller
did not ask for, and operating on it is the silently-wrong answer.
"""

from __future__ import annotations

import os
from pathlib import Path

from core.errors import refuse

__all__ = ["contained_child", "single_segment"]


def single_segment(name: str, *, code: str, what: str) -> str:
    """``name`` if it is exactly one path segment; otherwise refuse with ``code``.

    Args:
        name: the caller-supplied identifier.
        code: the refusal code, so each surface keeps its own documented name.
        what: what the name identifies, for the message ("a run log", "a checkpoint").
    """
    if not name:
        raise refuse(code, f"{what} must belong to a named run; the identifier is empty.")
    if (
        name in {".", ".."}
        or "\x00" in name
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
        or name != Path(name).name
    ):
        raise refuse(
            code,
            f"{name!r} is not a single path segment, so it cannot address {what}. Refusing "
            f"rather than sanitising it: a sanitised path is a path the caller did not ask for.",
            run_id=name,
        )
    return name


def contained_child(root: Path, name: str, *, code: str, what: str) -> Path:
    """``root / name``, refused unless it is one segment **and** resolves under ``root``.

    The returned path is the **resolved** one, so what the caller opens is exactly what was
    checked - no second resolution between the check and the use.

    Raises:
        Refusal: ``code`` for an empty or multi-segment name, or for one that resolves
            outside ``root`` (a symlink escape).
    """
    single_segment(name, code=code, what=what)
    # Written in the shape static analysis recognises as a path guard (CodeQL
    # py/path-injection): normalise with `os.path.realpath`, then a `startswith` check against
    # the normalised root that raises before the path is returned. Same behaviour as
    # `resolve()` + `is_relative_to`: symlinks are followed, and the root itself (an empty
    # name resolving to the root) is not a child.
    base = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(base, name))
    if not candidate.startswith(base.rstrip(os.sep) + os.sep):
        raise refuse(
            code,
            f"{name!r} resolves outside its root ({base}); a link under the root points "
            f"elsewhere. Refusing to follow it to {what}.",
            run_id=name,
            root=base,
        )
    return Path(candidate)
