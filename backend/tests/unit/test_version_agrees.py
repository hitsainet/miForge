"""miForge has ONE version: ``pyproject.toml`` ``[project].version``.

Operator rule (2026-10-17): miForge is 0.1.0, like miDataworks, until feature MVP, when it moves to
0.5.0 (the "functional pre-release"). Every snapshot used to be titled "miForge unknown" because the
sync workflow read a ``VERSION`` file that never existed. **MVP declared by the operator 2026-11-28
(decision O14): 0.5.0.** The API's OpenAPI ``info.version`` (``api/app.py``) and the frontend's
lock file now agree with it too: until then ``app.py`` carried its own literal, pinned by
nothing.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _pyproject_version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def test_frontend_version_equals_the_project_version():
    package = json.loads((ROOT / "frontend" / "package.json").read_text())
    assert package["version"] == _pyproject_version()


def test_the_version_is_a_plain_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", _pyproject_version())


def test_the_snapshot_title_reads_the_project_version():
    workflow = (ROOT / ".github" / "workflows" / "sync-to-clean.yml").read_text()
    assert "cat VERSION" not in workflow
    assert 'tomllib.load(open("pyproject.toml","rb"))["project"]["version"]' in workflow
    assert 'git commit -m "miForge ${VERSION}' in workflow


def test_the_version_is_the_operators_mvp_version():
    """Operator decision O14 (2026-11-28): feature MVP, the functional pre-release."""
    assert _pyproject_version() == "0.5.0"


def test_the_frontend_lock_file_carries_the_project_version():
    lock = json.loads((ROOT / "frontend" / "package-lock.json").read_text())
    assert lock["version"] == _pyproject_version()
    assert lock["packages"][""]["version"] == _pyproject_version()


def test_the_api_reports_the_project_version():
    """``api/app.py``'s ``FastAPI(version=...)`` literal is ``info.version`` of the OpenAPI
    document; read by AST, so the check needs no database."""
    tree = ast.parse((ROOT / "backend" / "src" / "api" / "app.py").read_text())
    found = [
        kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "FastAPI"
        for kw in node.keywords
        if kw.arg == "version" and isinstance(kw.value, ast.Constant)
    ]
    assert found == [_pyproject_version()]
