"""The manual's wiring: built and checked on every change, published from the mirror only.

Each assertion is about a line whose removal would leave the manual broken or stale WITHOUT any
other signal - the failure mode that goes quietly dark:

* the sync's dispatch of ``deploy-manual.yml`` (without it the mirror's ``paths:`` filter never
  fires over an orphan force-push, and the published manual silently stops updating - miStudio's
  sat eighteen days stale that way);
* the deploy's repository gate (without it the private repo tries to publish too);
* the CI build check's triggers (a ``paths:`` filter or a missing ``pull_request`` means a broken
  page merges green);
* the ``prebuild``/``postbuild`` hooks (without them the MCP tool page is missing, or nothing
  checks the built site for internal content);
* the three ``throw`` settings that make a broken link a failed build;
* ``npm test`` before the build (the publish check's own tests).

Read from the files the runner and npm actually read, parsed as YAML and JSON.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
MANUAL = REPO_ROOT / "manual"
MIRROR = "hitsainet/miForge"


def _workflow(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((WORKFLOWS / name).read_text("utf-8"))
    return loaded


def _triggers(doc: dict[str, Any]) -> dict[str, Any]:
    # PyYAML reads the bare key `on` as the boolean True.
    triggers = doc.get("on", doc.get(True))
    assert isinstance(triggers, dict), f"no trigger block: {triggers!r}"
    return triggers


def _run_lines(job: dict[str, Any]) -> list[str]:
    return [step.get("run", "") for step in job["steps"] if isinstance(step, dict)]


# ── deploy-manual.yml: publish from the mirror, and only from it ─────────────────────────


def test_deploy_manual_runs_only_on_the_public_mirror() -> None:
    jobs = _workflow("deploy-manual.yml")["jobs"]
    assert set(jobs) == {"build", "deploy"}
    for name, job in jobs.items():
        assert job.get("if") == f"github.repository == '{MIRROR}'", f"{name}: {job.get('if')!r}"


def test_deploy_manual_can_be_dispatched_and_builds_what_it_uploads() -> None:
    doc = _workflow("deploy-manual.yml")
    assert "workflow_dispatch" in _triggers(doc)
    build = doc["jobs"]["build"]
    assert "npm ci" in _run_lines(build)
    assert "npm run build" in _run_lines(build)
    upload = [
        s
        for s in build["steps"]
        if str(s.get("uses", "")).startswith("actions/upload-pages-artifact@")
    ]
    assert len(upload) == 1 and upload[0]["with"]["path"] == "manual/build"
    deploy = doc["jobs"]["deploy"]
    assert deploy["needs"] == "build"
    assert [s["uses"] for s in deploy["steps"]] == ["actions/deploy-pages@v4"]


def test_only_the_deploy_job_can_write_pages() -> None:
    doc = _workflow("deploy-manual.yml")
    assert doc["permissions"] == {"contents": "read"}
    assert doc["jobs"]["build"]["permissions"] == {"contents": "read"}
    assert doc["jobs"]["deploy"]["permissions"] == {"pages": "write", "id-token": "write"}


# ── manual-ci.yml: the private repo's build check ────────────────────────────────────────


def test_the_manual_build_check_runs_on_every_push_and_pull_request() -> None:
    doc = _workflow("manual-ci.yml")
    triggers = _triggers(doc)
    for event in ("push", "pull_request"):
        assert event in triggers, f"manual-ci does not run on {event}"
        assert "paths" not in (triggers[event] or {}), f"a paths filter on {event} skips runs"
    job = doc["jobs"]["build"]
    # Not gated to one repository: it must run on the private primary, where PRs are.
    assert "if" not in job
    assert job["defaults"]["run"]["working-directory"] == "manual"
    runs = _run_lines(job)
    assert runs.index("npm ci") < runs.index("npm run build")
    # The publish check's own tests (CodeQL js/bad-tag-filter): without this step a scanner that
    # silently scans less still passes every build.
    assert runs.index("npm ci") < runs.index("npm test") < runs.index("npm run build")


def test_npm_test_runs_the_publish_check_tests() -> None:
    scripts = json.loads((MANUAL / "package.json").read_text("utf-8"))["scripts"]
    assert scripts["test"] == "node --test scripts/check-published.test.mjs"
    assert (MANUAL / "scripts" / "check-published.test.mjs").is_file()


# ── sync-to-clean.yml: publish manual/, then dispatch the deploy ─────────────────────────


def _sync_steps() -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = _workflow("sync-to-clean.yml")["jobs"]["sync"]["steps"]
    return steps


def test_the_sync_dispatches_the_mirror_manual_deploy_after_verifying() -> None:
    steps = _sync_steps()
    names = [s.get("name", "") for s in steps]
    dispatch = [
        i for i, s in enumerate(steps) if "deploy-manual.yml/dispatches" in s.get("run", "")
    ]
    assert len(dispatch) == 1, "exactly one step must dispatch deploy-manual on the mirror"
    verify = names.index("Verify the published mirror")
    assert dispatch[0] > verify, "the dispatch must not be able to stop the verification"
    run = steps[dispatch[0]]["run"]
    assert '"ref":"main"' in run
    assert "${MIRROR}" in run
    # A refused dispatch is red, not a silently stale manual.
    assert re.search(r'if \[ "\$CODE" = "204" \]', run) and "exit 1" in run


def test_the_sync_publishes_manual_and_the_contract_it_builds_from() -> None:
    steps = _sync_steps()
    remove = next(s["run"] for s in steps if s.get("name") == "Remove excluded files and folders")
    assert not re.search(r"\brm\b[^\n]*\bmanual\b", remove), "the sync must not strip manual/"
    assert "! -name mcp-contract.md" in remove
    verify = next(s["run"] for s in steps if s.get("name") == "Verify the published mirror")
    for path in ("manual/docusaurus.config.ts", "docs/mcp-contract.md"):
        assert path in verify, f"Verify does not require {path} on the mirror"


# ── package.json and the site config: the three checks a build makes ─────────────────────


def test_the_build_generates_the_tool_page_and_checks_what_it_built() -> None:
    scripts = json.loads((MANUAL / "package.json").read_text("utf-8"))["scripts"]
    assert scripts["build"] == "docusaurus build"
    assert scripts["prebuild"] == "node scripts/sync-mcp-contract.mjs"
    assert scripts["postbuild"] == "node scripts/check-published.mjs"
    for script in ("sync-mcp-contract.mjs", "check-published.mjs"):
        assert (MANUAL / "scripts" / script).is_file()


def test_a_broken_link_anchor_or_markdown_link_fails_the_build() -> None:
    config = (MANUAL / "docusaurus.config.ts").read_text("utf-8")
    for setting in ("onBrokenLinks", "onBrokenAnchors", "onBrokenMarkdownLinks"):
        assert re.search(rf"^\s*{setting}: 'throw',$", config, flags=re.MULTILINE), setting
