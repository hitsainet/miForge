"""miForge runs on ONE Python version, and every place that names it agrees.

Operator decision 2026-10-21 (suite memory ``python-313-upgrade-decision``): the suite moves
from 3.11 to 3.13, miForge first. The version is written in six places that nothing tied
together:

* ``backend/Dockerfile``'s ``FROM python:X.Y-slim`` - what production runs;
* every ``actions/setup-python`` step in ``.github/workflows/`` - what CI tests and lints on;
* ``requires-python``'s floor - what an install promises;
* Black's and Ruff's ``target-version`` and MyPy's ``python_version`` - what the source is
  checked as.

Drift between them is the failure suite memory ``dev-venv-vs-image-version-drift`` records:
a suite green on one version, an image shipping another. It also has a sharper local form
here: with MyPy's target below 3.12 and CI on a newer interpreter, MyPy ABORTS parsing
numpy's PEP 695 stubs before reading a first-party file (pyproject.toml's MyPy section).

Every value is read from the file it governs; nothing here restates "3.13" except the
parser self-checks, so the next upgrade edits the six files and this test stays as is.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[3]
PYPROJECT = REPO / "pyproject.toml"
DOCKERFILE = REPO / "backend" / "Dockerfile"
WORKFLOWS = REPO / ".github" / "workflows"

_FROM_PYTHON = re.compile(r"^FROM\s+(?:--platform=\S+\s+)?python:(\d+)\.(\d+)[^\s]*", re.M)


def _dockerfile_versions(text: str) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in _FROM_PYTHON.findall(text)]


def _setup_python_steps(workflow: dict[str, Any]) -> list[tuple[str, Any]]:
    """Every ``actions/setup-python`` step, as (job id, its ``python-version`` or None)."""
    found: list[tuple[str, Any]] = []
    for job_id, job in (workflow.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            if str(step.get("uses", "")).startswith("actions/setup-python"):
                found.append((job_id, (step.get("with") or {}).get("python-version")))
    return found


def _mm(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)\.(\d+)", str(text).strip())
    assert m, f"not a MAJOR.MINOR version: {text!r}"
    return int(m.group(1)), int(m.group(2))


def _the_shipped_version() -> tuple[int, int]:
    versions = _dockerfile_versions(DOCKERFILE.read_text())
    assert versions, "backend/Dockerfile has no `FROM python:X.Y` line - the parser is blind"
    assert len(set(versions)) == 1, f"backend/Dockerfile names several Pythons: {versions}"
    return versions[0]


# ── the parsers, against inputs whose answers are known ───────────────────────────────


class TestTheParsersSeeWhatTheyClaimTo:
    def test_a_from_line_is_read_and_a_comment_is_not(self) -> None:
        text = "# FROM python:3.9-slim\nFROM python:3.13-slim AS base\nFROM base\n"
        assert _dockerfile_versions(text) == [(3, 13)]

    def test_a_setup_python_step_without_a_version_is_reported_not_skipped(self) -> None:
        wf = {"jobs": {"a": {"steps": [{"uses": "actions/setup-python@v5"}]}}}
        assert _setup_python_steps(wf) == [("a", None)]

    def test_a_quoted_and_an_unquoted_version_read_the_same(self) -> None:
        # Unquoted `3.10` is the YAML float 3.1 - the reason workflows quote it.
        wf = yaml.safe_load(
            "jobs:\n a:\n  steps:\n  - uses: actions/setup-python@v5\n"
            "    with:\n     python-version: '3.13'\n"
        )
        assert [_mm(v) for _, v in _setup_python_steps(wf)] == [(3, 13)]


# ── the six places agree ───────────────────────────────────────────────────────────────


class TestEveryPlaceNamesTheShippedVersion:
    def test_every_ci_python_is_the_image_python(self) -> None:
        shipped = _the_shipped_version()
        steps: list[tuple[str, str, Any]] = []
        for path in sorted(WORKFLOWS.glob("*.yml")):
            for job, version in _setup_python_steps(yaml.safe_load(path.read_text())):
                steps.append((path.name, job, version))
        # backend-tests.yml alone has three (lint, SQLite, PostgreSQL). Fewer means the
        # walk stopped seeing them, and an empty comparison below would pass.
        assert len([s for s in steps if s[0] == "backend-tests.yml"]) >= 3, steps
        wrong = [s for s in steps if s[2] is None or _mm(s[2]) != shipped]
        assert not wrong, f"image runs {shipped}; these CI steps do not: {wrong}"

    def test_requires_python_floor_is_the_image_python(self) -> None:
        spec = tomllib.loads(PYPROJECT.read_text())["project"]["requires-python"]
        m = re.fullmatch(r">=\s*(\d+\.\d+)", spec.strip())
        assert m, f"requires-python should be a bare floor, got {spec!r}"
        assert _mm(m.group(1)) == _the_shipped_version()

    def test_black_ruff_and_mypy_target_the_image_python(self) -> None:
        tools = tomllib.loads(PYPROJECT.read_text())["tool"]
        major, minor = _the_shipped_version()
        assert tools["black"]["target-version"] == [f"py{major}{minor}"]
        assert tools["ruff"]["target-version"] == f"py{major}{minor}"
        assert _mm(tools["mypy"]["python_version"]) == (major, minor)

    def test_the_interpreter_running_this_suite_meets_the_floor(self) -> None:
        """CI installs the shipped version, so this is equality there; a newer local venv
        is allowed, an older one is not a miForge environment."""
        assert sys.version_info[:2] >= _the_shipped_version()
