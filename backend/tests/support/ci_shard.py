"""Split the test tree across parallel CI jobs, by FILE, balanced on recorded durations.

The private repository's runners are small, so one job running the whole tree is the slowest
step between a push and a deployed image. `backend-tests.yml` runs the tree as a matrix of
shards instead, each job with its own service containers, and a final job that is green only
when every shard is.

How a shard is chosen
---------------------
`CI_SHARD=<i>/<n>` (1-based) in the environment turns this on; with it unset nothing is
ignored, so a local `pytest` and every other job run the whole tree exactly as before.

Every test file under the test root is assigned to exactly one shard by a deterministic
longest-processing-time packing over `ci_shard_durations.json` (seconds per file, measured).
A file the durations do not name (a new file) weighs the median, so it is still assigned and
still runs - nothing is dropped because the durations are stale; they only get less balanced.
`pytest_ignore_collect` then skips, without importing, every test file that belongs to
another shard. Each shard computes the same assignment from the same tree, so the shards are
disjoint and their union is the tree; `tests/.../test_ci_shards.py` proves both against
pytest's own collection.

Refresh the durations after the suite changes shape (optional; only balance depends on it):

    pytest <the CI arguments> -o junit_family=xunit1 --junitxml=/tmp/j.xml
    python <this file> refresh /tmp/j.xml
"""

from __future__ import annotations

import fnmatch
import json
import os
import statistics
import sys
import zlib
from pathlib import Path

ENV = "CI_SHARD"
DURATIONS = Path(__file__).with_name("ci_shard_durations.json")
_SKIP_DIRS = {"__pycache__", "node_modules", ".pytest_cache", ".venv", "venv", ".git"}


def parse(spec: str | None) -> tuple[int, int] | None:
    """`"2/5"` -> (2, 5). Unset or empty -> None. Anything else is an error, not a no-op:
    a typo in the matrix must not silently run the whole tree in every shard."""
    if not spec:
        return None
    try:
        i_s, n_s = spec.split("/")
        i, n = int(i_s), int(n_s)
    except ValueError as exc:
        raise ValueError(f"{ENV}={spec!r}: expected '<i>/<n>'") from exc
    if not (n >= 1 and 1 <= i <= n):
        raise ValueError(f"{ENV}={spec!r}: need 1 <= i <= n")
    return i, n


def collectable_files(root: Path, patterns: list[str]) -> list[str]:
    """Every file under `root` whose name matches pytest's `python_files`, as sorted
    root-relative POSIX paths."""
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in filenames:
            if name.endswith(".py") and any(fnmatch.fnmatch(name, p) for p in patterns):
                out.append(Path(dirpath, name).relative_to(root).as_posix())
    return sorted(out)


def load_durations(path: Path = DURATIONS) -> dict[str, float]:
    try:
        return {str(k): float(v) for k, v in json.loads(path.read_text()).items()}
    except FileNotFoundError:
        return {}


def assign(files: list[str], n: int, durations: dict[str, float]) -> dict[str, int]:
    """Map each file to a shard in 1..n. Deterministic: the same inputs give the same map."""
    known = [durations[f] for f in files if f in durations]
    default = statistics.median(known) if known else 1.0
    weight = {f: durations.get(f, default) for f in files}
    load = [0.0] * n
    out: dict[str, int] = {}
    for f in sorted(files, key=lambda f: (-weight[f], f)):
        k = min(range(n), key=lambda s: (load[s], s))
        load[k] += weight[f]
        out[f] = k + 1
    return out


def shard_of(rel: str, n: int, assignment: dict[str, int]) -> int:
    """The shard a file runs in. A path outside the assignment (it should not happen: the
    walk sees what pytest sees) still lands in exactly one shard, by a stable hash."""
    if rel in assignment:
        return assignment[rel]
    return zlib.crc32(rel.encode()) % n + 1


def ignore(collection_path: Path, config, root: Path) -> bool | None:
    """The body of `pytest_ignore_collect` for the conftest at `root`. Returns True to skip a
    test file that belongs to another shard, None otherwise (never False: False would FORCE
    collection of something pytest would have skipped)."""
    parsed = parse(os.environ.get(ENV))
    if parsed is None:
        return None
    i, n = parsed
    path = Path(collection_path)
    if path.suffix != ".py":
        return None
    patterns = list(config.getini("python_files"))
    if not any(fnmatch.fnmatch(path.name, p) for p in patterns):
        return None
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None
    cache = config.stash.setdefault(_KEY, {})
    if n not in cache:
        cache[n] = assign(collectable_files(root, patterns), n, load_durations())
    return True if shard_of(rel, n, cache[n]) != i else None


try:  # pytest's typed stash; imported lazily so `refresh` needs no pytest
    import pytest as _pytest

    _KEY = _pytest.StashKey[dict]()
except ImportError:  # pragma: no cover - only the CLI path runs without pytest
    _KEY = None  # type: ignore[assignment]


def _refresh(junit_paths: list[str]) -> None:
    import xml.etree.ElementTree as ET

    root = Path(__file__).resolve().parent.parent  # the test root this file lives under
    per_file: dict[str, float] = {}
    for jp in junit_paths:
        for case in ET.parse(jp).getroot().iter("testcase"):
            f = case.get("file")
            if not f:
                raise SystemExit("junit has no file= attribute; run with -o junit_family=xunit1")
            p = Path(f)
            p = p if p.is_absolute() else Path.cwd() / p
            try:
                rel = p.resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            per_file[rel] = per_file.get(rel, 0.0) + float(case.get("time") or 0.0)
    DURATIONS.write_text(
        json.dumps({k: round(v, 2) for k, v in sorted(per_file.items())}, indent=1) + "\n"
    )
    print(f"wrote {len(per_file)} files, {sum(per_file.values()):.0f}s total, to {DURATIONS}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "refresh":
        _refresh(sys.argv[2:])
    else:
        raise SystemExit("usage: ci_shard.py refresh <junit.xml>...")
