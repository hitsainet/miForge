"""G8 (FR-012.2, MR-2; ADR-022): nothing under ``mcp_server/`` imports the backend.

The MCP server is its own Deployment with no database, Redis or GPU (``gitops/pilot/mcp.yaml``).
A module that imported ``services`` or a session factory would raise on its first real call
there - however green its tests in a process that has a database. Every module of the package
is scanned (a helper a tool imports runs in the same pod), by AST, with a self-test that plants
each import form and requires it caught - so a scanner that sees nothing cannot pass.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MCP_DIR = Path(__file__).resolve().parents[2] / "src" / "mcp_server"

#: The backend's application packages, the ORM, and the fenced hot-update package.
FORBIDDEN_ROOTS = frozenset(
    {
        "api",
        "core",
        "db",
        "models",
        "services",
        "training",
        "workers",
        "schemas",
        "packs",
        "clients",
        "providers",
        "sqlalchemy",
    }
)


def forbidden_imports(source: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] in FORBIDDEN_ROOTS]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] in FORBIDDEN_ROOTS:
                found.append(node.module)
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in {"import_module", "__import__"} and node.args:
                arg = node.args[0]
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value.split(".")[0] in FORBIDDEN_ROOTS
                ):
                    found.append(arg.value)
    return found


def test_the_scan_reads_the_whole_package() -> None:
    files = list(MCP_DIR.rglob("*.py"))
    assert len(files) >= 18, f"found only {len(files)} modules under {MCP_DIR}"


@pytest.mark.parametrize(
    "path", sorted(MCP_DIR.rglob("*.py")), ids=lambda p: str(p.relative_to(MCP_DIR))
)
def test_no_mcp_module_imports_the_backend(path: Path) -> None:
    assert forbidden_imports(path.read_text()) == [], path


@pytest.mark.parametrize(
    "planted",
    [
        "import services.run_launch",
        "from db.session import session_scope",
        "from sqlalchemy.orm import Session",
        "import training.hotupdate",
        "from core import config",
        "importlib.import_module('models.approval')",
        "__import__('workers.celery_app')",
    ],
)
def test_the_scanner_catches_each_planted_form(planted: str) -> None:
    assert forbidden_imports(planted), planted


def test_a_relative_import_inside_the_package_is_not_a_backend_import() -> None:
    assert forbidden_imports("from .client import ForgeClient\nfrom ..context import READ") == []
