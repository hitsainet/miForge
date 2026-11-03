"""miForge has ONE version: ``pyproject.toml`` ``[project].version``.

Operator rule (2026-10-17): miForge is 0.1.0, like miDataworks, until feature MVP, when it moves to
0.5.0 (the "functional pre-release"). Every snapshot used to be titled "miForge unknown" because the
sync workflow read a ``VERSION`` file that never existed.
"""

from __future__ import annotations

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
