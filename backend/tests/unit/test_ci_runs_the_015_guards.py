"""CI runs the 015 guards on every push, and the mirror check refuses the full export (review 6).

miLLM's backend tests ran only for code-path changes and its publish gate passed any commit that
touched none of them, so a docs-, k8s- or workflow-only push skipped the very tests that guard
those files - one such push would have published the private census export. In miForge the
workflows the publish gate requires (``sync-to-clean.yml`` ``REQUIRED_WORKFLOWS``) carry no path
filter; this pins that, or, should a filter ever be added, that it covers every path 015's
guards read. And the mirror verification step lists the full export among what must be absent.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
SYNC = WORKFLOWS / "sync-to-clean.yml"
#: A representative file for each path 015's guards read.
GUARDED_PATHS = (
    "docs/legacy-identifiers.md",
    "docs/legacy-identifiers.public.md",
    "gitops/pilot/storage.yaml",
    "gitops/pilot/config.yaml",
    "k8s/base/x.yaml",
    ".github/workflows/sync-to-clean.yml",
    "manual/docusaurus.config.ts",
    "manual/publish-rules.json",
    "brand-guard.json",
)


def _on(doc: dict[str, Any]) -> dict[str, Any]:
    return doc.get("on") or doc.get(True) or {}  # PyYAML reads a bare `on:` key as True


def required_workflows() -> list[str]:
    gate = yaml.safe_load(SYNC.read_text("utf-8"))["jobs"]["gate"]
    return [w.strip() for w in gate["env"]["REQUIRED_WORKFLOWS"].splitlines() if w.strip()]


def untested_paths(doc: dict[str, Any]) -> list[str]:
    """The guarded paths a push could change without this workflow running."""
    push = _on(doc).get("push") or {}
    if not isinstance(push, dict):
        return []
    include, exclude = push.get("paths"), push.get("paths-ignore")
    missed = []
    for path in GUARDED_PATHS:
        not_included = bool(include) and not _selected(path, include)
        excluded = bool(exclude) and _selected(path, exclude)
        if not_included or excluded:
            missed.append(path)
    return missed


def _selected(path: str, patterns: list[str]) -> bool:
    """GitHub's filter semantics: patterns apply in order, a ``!`` pattern un-selects, and
    the last pattern that matches decides (review round 2: ``['**', '!docs/**']`` excludes)."""
    chosen = False
    for pattern in patterns:
        negated = pattern.startswith("!")
        if fnmatch.fnmatch(path, pattern[1:] if negated else pattern):
            chosen = not negated
    return chosen


def test_the_gate_requires_the_test_workflows() -> None:
    required = required_workflows()
    assert {"backend-tests.yml", "frontend-tests.yml"} <= set(required), required


@pytest.mark.parametrize("workflow", required_workflows())  # derived, never a hand-kept list
def test_required_workflows_run_on_every_guarded_path(workflow: str) -> None:
    doc = yaml.safe_load((WORKFLOWS / workflow).read_text("utf-8"))
    missed = untested_paths(doc)
    assert not missed, f"{workflow} would not run for a push touching {missed}"


def test_the_filter_check_sees_a_filter() -> None:
    """Self-test: a code-only filter like miLLM's is reported, a covering one is not."""
    narrow = {"on": {"push": {"branches": ["main"], "paths": ["backend/**", "pyproject.toml"]}}}
    assert "docs/legacy-identifiers.md" in untested_paths(narrow)
    wide = {"on": {"push": {"paths": ["**"]}}}
    assert untested_paths(wide) == []
    ignoring = {"on": {"push": {"paths-ignore": ["docs/**"]}}}
    assert "docs/legacy-identifiers.md" in untested_paths(ignoring)
    negated = {"on": {"push": {"paths": ["**", "!docs/**"]}}}
    assert "docs/legacy-identifiers.md" in untested_paths(negated)
    reinstated = {"on": {"push": {"paths": ["**", "!docs/**", "docs/legacy-identifiers*.md"]}}}
    assert "docs/legacy-identifiers.md" not in untested_paths(reinstated)


def test_the_mirror_check_refuses_the_full_export() -> None:
    steps = yaml.safe_load(SYNC.read_text("utf-8"))["jobs"]["sync"]["steps"]
    verify = next(s for s in steps if s.get("name") == "Verify the published mirror")
    loops = [ln for ln in verify["run"].splitlines() if ln.strip().startswith("for p in 0xcc")]
    assert loops, "the leak loop of the mirror verification step was not found"
    assert "docs/legacy-identifiers.md" in loops[0].replace(";", " ").split(), loops[0]
    assert "docs/legacy-identifiers.public.md" not in loops[0]
