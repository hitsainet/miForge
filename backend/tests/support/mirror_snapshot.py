"""Is this checkout the public mirror's snapshot?  (mirror CI, 2026-11-08)

``sync-to-clean.yml`` publishes ONE parentless commit to ``hitsainet/miForge`` and stamps it
with the ``Source-Commit: <sha>`` and ``Private-CI: passed`` trailers, the marker the mirror's
image build already requires (``docker-images.yml``, job ``private-ci``). A test that cannot run
on that snapshot (it needs history, or a file the sync strips) may skip there - and ONLY there:
the private repository's CI is the gate, and nothing may skip on it for want of something only
the mirror lacks. The absence of a file is not proof of the mirror (a deleted file looks the
same); this commit shape is.

``EXPECTED`` and :func:`make_repo` build each shape of tree from a real ``git`` so the predicate
is tested against commits, not stubs.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_COMMIT = re.compile(r"^Source-Commit: [0-9a-f]{40}$", re.M)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    except (FileNotFoundError, NotADirectoryError):
        return None


def mirror_snapshot(root: Path = REPO_ROOT) -> bool:
    """True only when ``root``'s HEAD is a parentless commit carrying both sync trailers. A
    commit of this repository has a parent and no such trailers; a shallow clone of it still
    lacks the trailers. No git, or no repository: False."""
    parents = _git(root, "rev-list", "--parents", "-n", "1", "HEAD")
    trailers = _git(root, "log", "-1", "--format=%(trailers:only,unfold)", "HEAD")
    if parents is None or trailers is None or parents.returncode or trailers.returncode:
        return False
    if len(parents.stdout.split()) != 1:
        return False
    lines = trailers.stdout.splitlines()
    return "Private-CI: passed" in lines and bool(_SOURCE_COMMIT.search(trailers.stdout))


SOURCE_SHA = "a68add3e85ae0cf25c16b885826f5fbfaeded0d5"
#: What the sync's "Build the orphan snapshot" step passes to ``git commit``.
SYNC_TRAILERS = ("--trailer", f"Source-Commit: {SOURCE_SHA}", "--trailer", "Private-CI: passed")

#: shape -> whether :func:`mirror_snapshot` must call it the mirror
EXPECTED = {
    "none": False,
    "private": False,
    "shallow-private": False,
    "orphan-plain": False,
    "orphan-source-only": False,
    "trailers-with-parent": False,
    "mirror": True,
}


def _run(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )


def make_repo(root: Path, shape: str) -> Path:
    """A tree of ``shape`` (a key of :data:`EXPECTED`): ``none`` (no repository), ``private``
    (two plain commits), ``shallow-private`` (a depth-1 clone of one), ``orphan-plain``,
    ``orphan-source-only`` (one trailer), ``trailers-with-parent``, ``mirror`` (what the sync
    publishes)."""
    root.mkdir(parents=True, exist_ok=True)
    if shape == "none":
        return root
    if shape == "shallow-private":
        source = make_repo(root.parent / f"{root.name}-source", "private")
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", source.as_uri(), str(root)],
            check=True,
            capture_output=True,
        )
        return root
    _run(root, "init", "-q")
    (root / "README.md").write_text("x\n")
    _run(root, "add", "-A")
    if shape in ("private", "trailers-with-parent"):
        _run(root, "commit", "-qm", "first")
        (root / "README.md").write_text("y\n")
        _run(root, "add", "-A")
    trailers: tuple[str, ...] = ()
    if shape in ("mirror", "trailers-with-parent"):
        trailers = SYNC_TRAILERS
    elif shape == "orphan-source-only":
        trailers = SYNC_TRAILERS[:2]
    elif shape not in ("private", "orphan-plain"):
        raise ValueError(shape)
    _run(root, "commit", "-qm", "miForge - filtered snapshot", *trailers)
    return root
