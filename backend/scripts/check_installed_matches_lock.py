"""Fail when the environment holds a distribution ``constraints.txt`` does not pin, or pins
at another version (deps sync 2026-10-21).

Run straight after the dependency install in ``backend/Dockerfile`` and in every job of
``backend-tests.yml``, so the image BUILD and the CI install fail on drift instead of a test
noticing later. ``-c constraints.txt`` stops pip choosing a different version of a package
the lock names. This check catches the other half: a package the lock does not name at all,
pulled in by a new dependency. Standard library only, because it runs before anything else
is trusted.

    python backend/scripts/check_installed_matches_lock.py constraints.txt

``test_dependencies_are_locked.py`` runs the same ``drift`` function inside the suite, and
pins that both install paths call this script.
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

#: Installed but never by the lock: the project itself (editable, or ``--no-deps``) and the
#: installer that ships with the interpreter.
NOT_LOCKED = frozenset({"miforge", "pip"})


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_lock(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*==\s*([0-9][0-9A-Za-z.+!]*)", line)
        if match is None:
            raise ValueError(f"constraints.txt: {raw!r} is not an exact pin")
        name = canon(match.group(1))
        if name in out:
            raise ValueError(f"constraints.txt pins {name} twice")
        out[name] = match.group(3)
    return out


def drift(installed: dict[str, str], lock: dict[str, str]) -> dict[str, str]:
    """Installed distributions the lock does not pin, or pins at another version.

    A local version label (``2.14.1+cpu``) is the same release as the lock's ``2.14.1``. pip
    treats ``==2.14.1`` as satisfied by it, and that is how CI keeps its CPU torch.
    """
    out = {}
    for name, version in sorted(installed.items()):
        if name in NOT_LOCKED:
            continue
        if name not in lock:
            out[name] = f"{version} installed, not in constraints.txt"
        elif version.split("+", 1)[0] != lock[name]:
            out[name] = f"{version} installed, constraints.txt pins {lock[name]}"
    return out


def installed_now() -> dict[str, str]:
    return {canon(d.metadata["Name"]): d.version for d in metadata.distributions()}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} constraints.txt", file=sys.stderr)
        return 2
    lock = read_lock(Path(argv[1]).read_text("utf-8"))
    found = drift(installed_now(), lock)
    for name, why in found.items():
        print(f"LOCK DRIFT {name}: {why}", file=sys.stderr)
    if found:
        print(
            f"{len(found)} distribution(s) differ from {argv[1]}. Regenerate the lock (its "
            "header says how) or fix the install line.",
            file=sys.stderr,
        )
        return 1
    print(f"installed set matches {argv[1]} ({len(lock)} pins)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
