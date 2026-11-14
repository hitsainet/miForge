"""Review round 2 of 015 P0: the config-rev push base is wired and never narrows to nothing.

Round 1's own test covered ``stale_commits`` and ``push_base`` with ``CONFIG_REV_BASE`` set or all
zeros, in a repository with no ``origin/main``. Left unpinned:

* the workflow side - every ``backend-tests.yml`` job that runs pytest must pass
  ``github.event.before`` and check out the full history, or CI silently falls back to HEAD~1;
* the merge-base fallback must never return HEAD itself (``HEAD..HEAD`` checks nothing);
* an unknown ``CONFIG_REV_BASE`` (a force push: the old tip is not fetched) must not be used,
  because ``git rev-list <unknown>..HEAD`` prints nothing and ``stale_commits`` would pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.unit.test_gitops_config_rev import (
    CONFIG_V1,
    CONFIG_V2,
    REPO_ROOT,
    GitError,
    _commit,
    _deploy,
    _git,
    push_base,
    stale_commits,
)

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "backend-tests.yml"


def _pytest_jobs() -> dict[str, dict[str, Any]]:
    jobs = yaml.safe_load(WORKFLOW.read_text("utf-8"))["jobs"]
    return {
        name: job
        for name, job in jobs.items()
        if any("pytest" in str(step.get("run", "")) for step in job.get("steps", []))
    }


def test_every_pytest_job_passes_the_push_base_and_full_history() -> None:
    jobs = _pytest_jobs()
    assert len(jobs) >= 2, f"expected the SQLite and PostgreSQL jobs, found {sorted(jobs)}"
    for name, job in jobs.items():
        assert job.get("env", {}).get("CONFIG_REV_BASE") == "${{ github.event.before }}", name
        checkout = next(
            s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout")
        )
        assert (checkout.get("with") or {}).get(
            "fetch-depth"
        ) == 0, f"{name}: the config-rev guard needs the push's whole history (fetch-depth: 0)"


def _repo(tmp_path: Path) -> tuple[str, str]:
    """base -> config change with no bump -> unrelated commit; returns (base, head)."""
    _git(tmp_path, "init", "-q")
    pilot = tmp_path / "gitops" / "pilot"
    pilot.mkdir(parents=True)
    (pilot / "config.yaml").write_text(CONFIG_V1.replace("name: c", "name: miforge-config"))
    (pilot / "backend.yaml").write_text(_deploy("a").replace("name: c}", "name: miforge-config}"))
    base = _commit(tmp_path, "base")
    (pilot / "config.yaml").write_text(CONFIG_V2.replace("name: c", "name: miforge-config"))
    _commit(tmp_path, "config change, no bump")
    (pilot / "other.yaml").write_text("kind: Service\nmetadata: {name: s}\n")
    head = _commit(tmp_path, "unrelated")
    return base, head


def test_the_merge_base_fallback_is_never_head(tmp_path: Path) -> None:
    """On a push to main ``origin/main`` IS HEAD; the fallback must step to HEAD~1."""
    _, head = _repo(tmp_path)
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", head)
    base = push_base(tmp_path, {})
    assert base is not None and base != head, "push_base chose HEAD: HEAD..HEAD checks nothing"
    assert base == _git(tmp_path, "rev-parse", "HEAD~1").stdout.strip()


def test_the_merge_base_is_used_on_a_branch(tmp_path: Path) -> None:
    """A branch two commits ahead of origin/main checks both of them."""
    base, _ = _repo(tmp_path)
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", base)
    assert push_base(tmp_path, {}) == base
    assert len(stale_commits(tmp_path, push_base(tmp_path, {}) or "")) == 1


def test_an_unknown_before_is_not_used(tmp_path: Path) -> None:
    """A force push's old tip is not in the clone (and here cannot be fetched): the guard must
    FAIL, not silently narrow to HEAD~1 (review round 2, item 5)."""
    base, _ = _repo(tmp_path)
    missing = "1234567" * 5 + "89abc"
    with pytest.raises(GitError, match="could not be fetched"):
        push_base(tmp_path, {"CONFIG_REV_BASE": missing})
    assert push_base(tmp_path, {"CONFIG_REV_BASE": base}) == base
    assert push_base(tmp_path, {"CONFIG_REV_BASE": "0" * 40}) is not None  # a new branch


def test_a_failing_git_call_fails_closed(tmp_path: Path) -> None:
    """``rev-list`` on a bad range must raise, never return "no commits to check"."""
    _repo(tmp_path)
    with pytest.raises(GitError):
        stale_commits(tmp_path, "no-such-ref")
