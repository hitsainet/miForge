"""The worker's own exit and wipe path is unchanged by the fresh pod per session (operator
2026-11-26: option B ADDS a pod deletion from the backend; the supervisor still kills and reaps the
session, wipes its roots and, as PID 1, IPC, truncates the termination message and ``os._exit``s,
exactly as P49 and P51 made it).

Each function on that path is pinned by the sha256 of its AST (``ast.dump``: code and docstring,
not comments or layout), read from the file the worker image ships. A change to any of them turns
this red; it then needs its own review of the containment it touches, and a new digest here with
the reason. A function that disappears is red too (never a silent pass).
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[3] / "src" / "wsp_worker"

#: (file, qualified name) -> sha256 of ``ast.dump(node)``, at miForge ``c8230db``.
PINNED = {
    ("supervisor.py", "Supervisor.exit_now"): (
        "419efa0cb32de8e3a342a4e6bbacecdf3031730af6385f92348f94a2b27687de"
    ),
    ("supervisor.py", "Supervisor.kill_session"): (
        "96581b20ecf22505352184527d76c7e9c594cbcc5c9b5fcbd4d2bfeb5d1c61db"
    ),
    ("supervisor.py", "Supervisor._reap"): (
        "f31dfa498e60fbb7aaca51708f986e57381b18b055c1b761876e4ef7707d8f9f"
    ),
    ("supervisor.py", "Supervisor._due"): (
        "8a94b53ba1c9569cfdc496ac41d2b5dc05d5d30033ed4623dbfea634d6904f68"
    ),
    ("supervisor.py", "Supervisor.watchdog"): (
        "afb9beaf94d56a6ac2ec298737081d3b3d30f50dcbf7bc9b99bf1b10efaa5d3b"
    ),
    ("supervisor.py", "Handler._ack"): (
        "2937f0cb521f9f0e77fc6dbde0fdf107339d7f9a23071890ddf3ec3a03ac6b76"
    ),
    ("supervisor.py", "Handler._recycle"): (
        "4609550884fcc416113dd51d9c76262c36a6805e9428d1609f4bc5f709af795f"
    ),
    ("supervisor.py", "wipe_roots"): (
        "9bbde42c9916a99347ba22023c035b90451ddd7bf927614399ee9fbf44487c1d"
    ),
    ("supervisor.py", "truncate"): (
        "60c3e33ff4f0f3d6300d92f40c900a7dc718daf6778c650bd0ec153fa50215c6"
    ),
    ("supervisor.py", "run"): "c585961f7e93b640255fc1dd85e35dd6d37e4790d3c7244b30d9dfbc12bf21d1",
    ("supervisor.py", "main"): "028268948e14a95730a8904adba1529041936b0ed3e045d1fc845a7d0badbad8",
    ("ipc_wipe.py", "wipe_all"): (
        "77c801d2ac5236d3d73b27128b50d88169a7bef0cefc2c26a115b4315f83ad4d"
    ),
}


def _definitions(path: Path) -> dict[str, ast.AST]:
    found: dict[str, ast.AST] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = prefix + child.name
                found[name] = child
                if isinstance(child, ast.ClassDef):
                    walk(child, name + ".")

    walk(ast.parse(path.read_text(encoding="utf-8")), "")
    return found


def _digest(node: ast.AST) -> str:
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


@pytest.mark.parametrize(("filename", "name"), sorted(PINNED))
def test_the_exit_and_wipe_path_is_byte_for_byte_the_same_code(filename: str, name: str) -> None:
    definitions = _definitions(WORKER / filename)
    assert name in definitions, f"{filename}: {name} is gone"
    assert _digest(definitions[name]) == PINNED[(filename, name)], (
        f"{filename}: {name} changed. The fresh pod per session must not change the worker's own "
        "exit and wipe; review the containment it touches, then update this digest with the reason."
    )


def test_the_exit_still_wipes_then_exits_ok() -> None:
    """A readable statement of what the digest holds, so a reader need not decode it."""
    source = ast.unparse(_definitions(WORKER / "supervisor.py")["Supervisor.exit_now"])
    order = [
        "self.kill_session()",
        "self._reap(",
        "wipe_roots(",
        "ipc_wipe.wipe_all()",
        "truncate(",
        "os._exit(spec.EXIT_OK)",
    ]
    positions = [source.index(step) for step in order]
    assert positions == sorted(positions)
