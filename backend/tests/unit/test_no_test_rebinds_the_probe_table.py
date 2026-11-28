"""No test assigns into ``api.health.DEPENDENCY_PROBES`` directly (2026-11-08, main red).

Two client fixtures did ``DEPENDENCY_PROBES[name] = lambda: ok``: the stub outlived the test
and every later test in the same process saw a Postgres probe that always says ok. CI runs a
shard in ONE worker, so ``test_health_bounded_on_a_hung_database.py`` read a hung database as
healthy there while every local ``-n 8`` run passed. ``monkeypatch.setitem`` restores; an
assignment does not. Scanned by AST over every test module.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]


def rebinding_sites(source: str) -> list[int]:
    lines = []
    for node in ast.walk(ast.parse(source)):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Subscript):
                base = target.value
                name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
                if name == "DEPENDENCY_PROBES":
                    lines.append(node.lineno)
    return lines


def test_no_test_rebinds_the_probe_table() -> None:
    modules = sorted(TESTS.rglob("*.py"))
    assert len(modules) > 100
    found = [
        f"{p.relative_to(TESTS)}:{line}"
        for p in modules
        for line in rebinding_sites(p.read_text("utf-8"))
    ]
    assert not found, (
        "assign probes with monkeypatch.setitem(health_module.DEPENDENCY_PROBES, ...), "
        "never directly - an assignment outlives the test: " + ", ".join(found)
    )


def test_the_scanner_sees_an_assignment() -> None:
    assert rebinding_sites("health_module.DEPENDENCY_PROBES['postgres'] = lambda: {}\n") == [1]
    assert rebinding_sites("DEPENDENCY_PROBES[name] = f\n") == [1]
    assert rebinding_sites("monkeypatch.setitem(health_module.DEPENDENCY_PROBES, 'x', f)\n") == []
