"""No test finds a sibling checkout except through ``tests/support/siblings.py`` (FR-015.8).

Five private implementations of "where is miLLM" disagreed in three ways, and each
disagreement was a defect (FTDD §10.6). One helper fixes them only while nothing re-grows
its own copy, so this guard walks every test module's syntax tree and refuses:

- a path join whose right operand is a sibling folder name (``REPO_ROOT.parent / "miLLM"``);
- a read of a sibling-location variable or a require switch through ``os.environ.get``,
  ``os.environ[...]`` or ``os.getenv``.

It reads the AST, not the text, so a comment or docstring naming a folder is not a finding,
and the scanner is a pure function proven on planted sources by its own self-test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.support import siblings

TESTS_ROOT = Path(__file__).resolve().parents[1]
HELPER = TESTS_ROOT / "support" / "siblings.py"
FOLDERS = frozenset({"miLLM", "miServer", "miDataworks", "miStudio", "miWorkbench"})
VARIABLES = frozenset(
    (
        *siblings.MISERVER.env,
        *siblings.RETIRED_ENV,
        *siblings.MIDATAWORKS.env,
        *siblings.REQUIRE_SWITCHES,
    )
)


def _is_environ(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _constant(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def inline_lookups(source: str) -> list[tuple[int, str]]:
    """Every (line, description) in ``source`` that finds a sibling inline."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            folder = _constant(node.right)
            if folder in FOLDERS:
                found.append((node.lineno, f'path join with "{folder}"'))
        elif isinstance(node, ast.Call):
            func = node.func
            first = _constant(node.args[0]) if node.args else None
            environ_get = (
                isinstance(func, ast.Attribute) and func.attr == "get" and _is_environ(func.value)
            )
            getenv = (
                isinstance(func, ast.Attribute)
                and func.attr == "getenv"
                and isinstance(func.value, ast.Name)
                and func.value.id == "os"
            )
            if (environ_get or getenv) and first in VARIABLES:
                found.append((node.lineno, f"environment read of {first}"))
        elif isinstance(node, ast.Subscript) and _is_environ(node.value):
            key = _constant(node.slice)
            if key in VARIABLES:
                found.append((node.lineno, f"environment read of {key}"))
    return found


def test_no_inline_lookup() -> None:
    modules = sorted(p for p in TESTS_ROOT.rglob("*.py") if p != HELPER)
    assert len(modules) > 100, f"discovery found only {len(modules)} test modules"
    findings = [
        f"{path.relative_to(TESTS_ROOT.parent.parent)}:{line}: {what}"
        for path in modules
        for line, what in inline_lookups(path.read_text("utf-8"))
    ]
    assert not findings, (
        "find siblings through tests/support/siblings.py (locate/absent), not inline:\n"
        + "\n".join(findings)
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('root = REPO_ROOT.parent / "miLLM"\n', 1),
        ('root = base / "miServer" / "docs"\n', 1),
        ('import os\nx = os.environ.get("MIFORGE_MILLM_ROOT")\n', 1),
        ('import os\nx = os.environ["MILLM_REPO"]\n', 1),
        ('import os\nx = os.getenv("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS")\n', 1),
        ('import os\nx = os.environ.get("MIFORGE_MIDATAWORKS_ROOT")\n', 1),
        # Not findings: a comment, a docstring, an unrelated variable, a non-sibling folder.
        ('"""Finds ../miLLM beside us."""\n# REPO_ROOT.parent / "miLLM"\n', 0),
        ('import os\nx = os.environ.get("MILLM_BASE_URL")\n', 0),
        ('root = base / "docs"\n', 0),
    ],
)
def test_scanner_self_test(source: str, expected: int) -> None:
    assert len(inline_lookups(source)) == expected
