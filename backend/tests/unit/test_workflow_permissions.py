"""Every GitHub Actions job runs with an explicit, least-privilege GITHUB_TOKEN.

CodeQL ``actions/missing-workflow-permissions`` (alerts 1-2): a job with no ``permissions``
block, in a workflow with none at the top, gets the repository's default token, which may be
read-write on everything. Each job must therefore be covered by its own block or a top-level
one, and a top-level block must not grant write. Read from the workflow files themselves, so a
new workflow or job is checked the day it lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[3] / ".github" / "workflows"


def _workflows() -> list[Path]:
    found = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    assert found, f"no workflows found under {WORKFLOWS} - the check would pass vacuously"
    return found


def _load(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(path.read_text("utf-8"))
    return loaded


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_every_job_has_an_explicit_token_scope(path: Path) -> None:
    doc = _load(path)
    top = doc.get("permissions")
    uncovered = [
        name
        for name, job in (doc.get("jobs") or {}).items()
        if top is None and (job or {}).get("permissions") is None
    ]
    assert not uncovered, f"{path.name}: jobs with the default token: {uncovered}"


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_a_top_level_block_never_grants_everything(path: Path) -> None:
    top = _load(path).get("permissions")
    assert top not in ("write-all", "read-all"), f"{path.name}: top-level {top!r}"


@pytest.mark.parametrize("name", ["docs-check.yml", "docker-images.yml"])
def test_the_two_flagged_workflows_default_to_read_only(name: str) -> None:
    assert _load(WORKFLOWS / name).get("permissions") == {"contents": "read"}
