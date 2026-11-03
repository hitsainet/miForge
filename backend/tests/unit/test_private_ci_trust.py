"""The mirror's image build trusts THIS repository's CI (operator directive, 2026-10-20).

Before: every push ran the suite here, the sync published a snapshot straight away, and the
public mirror's image build waited for the mirror to run the same suite again (or, in some
repos, for nothing at all). After:

1. PRIVATE: `sync-to-clean.yml`'s `gate` job runs `.github/scripts/private_ci_gate.py`, which
   refuses unless every required CI workflow passed for the exact commit (Backend Tests
   including its `all-shards` aggregator). The `sync` job `needs` the gate, checks out the
   commit the gate passed, and stamps the snapshot with `Source-Commit:` and
   `Private-CI: passed` trailers.
2. MIRROR: `docker-images.yml`'s `build` needs a `private-ci` job that requires those
   trailers on the snapshot, and nothing in it waits on the mirror's own test runs.
3. ORDER: the sync job is serialised, and a branch sync never moves the mirror back to an
   older commit than the one it already holds.

Every assertion reads the parsed YAML or RUNS the step's script; none greps the text.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

yaml = pytest.importorskip("yaml")

# ── per-repository facts ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
PRIVATE = "Onegaishimas/miForge"
# Workflows that run as CI here but deliberately do not gate the mirror, with the reason.
EXEMPT: dict[str, str] = {
    "docs-check.yml": (
        "validates the private 0xcc/ document chain, none of which ships; "
        "it never gated an image"
    ),
}
# ─────────────────────────────────────────────────────────────────────────────────────

WORKFLOWS = REPO_ROOT / ".github" / "workflows"
GATE_SCRIPT = REPO_ROOT / ".github" / "scripts" / "private_ci_gate.py"


def _load(name: str) -> dict:
    doc = yaml.safe_load((WORKFLOWS / name).read_text())
    doc["on"] = doc.pop(True, doc.get("on"))  # PyYAML reads the bare key `on` as boolean True
    return doc


def _runs_here(job: dict) -> bool:
    """False for a job that can only run on the public mirror."""
    cond = str(job.get("if", ""))
    return not (
        "github.repository" in cond and "hitsainet/" in cond and "Onegaishimas/" not in cond
    )


def _private_ci_workflows() -> set[str]:
    """Every workflow that is CI for main here: runs on a push to main AND on pull requests,
    with at least one job that runs in this repository."""
    out = set()
    for path in sorted(WORKFLOWS.glob("*.yml")):
        doc = _load(path.name)
        on = doc["on"] if isinstance(doc["on"], dict) else dict.fromkeys(doc["on"])
        if "pull_request" not in on or "push" not in on:
            continue
        branches = (on["push"] or {}).get("branches")
        if branches is not None and "main" not in branches:
            continue
        if any(_runs_here(j) for j in doc["jobs"].values()):
            out.add(path.name)
    return out - set(EXEMPT)


def _step(job: dict, name: str) -> dict:
    return next(s for s in job["steps"] if s.get("name") == name)


def _bash(script: str, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    full = {**os.environ, **env}
    return subprocess.run(
        ["bash", "-e", "-c", script], cwd=cwd, env=full, capture_output=True, text=True, check=False
    )


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


# ── 1. the private gate ──────────────────────────────────────────────────────────────


def test_the_gate_requires_every_ci_workflow_that_runs_here():
    sync = _load("sync-to-clean.yml")
    gate = sync["jobs"]["gate"]
    required = {w.strip() for w in gate["env"]["REQUIRED_WORKFLOWS"].split() if w.strip()}
    assert required == _private_ci_workflows(), (
        "REQUIRED_WORKFLOWS must name every CI workflow that runs on main here (or EXEMPT it "
        "with a reason): a CI workflow left out can be red while its code reaches the mirror"
    )
    assert "backend-tests.yml" in required
    assert gate["permissions"].get("actions") == "read", "the gate lists runs and jobs"
    run = _step(gate, "Require private CI to have passed for this commit")
    assert run["run"].strip() == "python .github/scripts/private_ci_gate.py"
    assert run["env"]["SHA"] == "${{ github.sha }}"
    assert gate["outputs"]["sha"] == "${{ steps.gate.outputs.sha }}" and run["id"] == "gate"


def test_nothing_reaches_the_mirror_except_through_the_gate():
    jobs = _load("sync-to-clean.yml")["jobs"]
    pushers = {k for k, j in jobs.items() if "HITSAINET_SYNC" in json.dumps(j)}
    assert pushers == {"sync"}, f"only the gated job may hold the mirror token: {pushers}"
    needs = jobs["sync"]["needs"]
    assert needs == "gate" or "gate" in needs
    assert "always()" not in str(
        jobs["sync"].get("if", "")
    ), "always() would run the sync over a failed gate"


def test_the_sync_publishes_the_commit_the_gate_passed():
    sync = _load("sync-to-clean.yml")["jobs"]["sync"]
    checkout = next(
        s for s in sync["steps"] if str(s.get("uses", "")).startswith("actions/checkout")
    )
    assert checkout["with"]["ref"] == "${{ needs.gate.outputs.sha }}"
    snap = _step(sync, "Build the orphan snapshot")
    assert snap["env"]["SOURCE_SHA"] == "${{ needs.gate.outputs.sha }}"
    for step in sync["steps"]:
        assert "GITHUB_SHA" not in step.get("run", ""), (
            f"{step.get('name')}: GITHUB_SHA names the triggering commit, "
            "not necessarily the gated one"
        )


def test_the_sync_is_serialised_and_never_moves_the_mirror_backwards():
    sync = _load("sync-to-clean.yml")["jobs"]["sync"]
    conc = sync["concurrency"]
    assert conc["group"] and conc["cancel-in-progress"] is False
    names = [s.get("name") for s in sync["steps"]]
    order = names.index("Never move the mirror backwards")
    assert sync["steps"][order]["id"] == "order"
    for step in sync["steps"][order + 1 :]:
        assert "steps.order.outputs.publish != 'false'" in str(step.get("if", "")), (
            f"{step.get('name')} runs even when the ordering check said "
            "an older commit must not publish"
        )


# ── 2. the snapshot's trailers and the mirror's trust check, end to end ─────────────


@pytest.fixture
def source_repo(tmp_path):
    repo = tmp_path / "src"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "VERSION").write_text("0.5.0\n")
    (repo / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.5.0"\n')
    (repo / "README.md").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "source")
    return repo


def _mirror_check() -> str:
    job = _load("docker-images.yml")["jobs"]["private-ci"]
    return _step(job, "Require the private repo's CI verdict")["run"]


def test_a_snapshot_the_sync_builds_passes_the_mirror_check(source_repo):
    source_sha = _git(source_repo, "rev-parse", "HEAD")
    snap = _step(_load("sync-to-clean.yml")["jobs"]["sync"], "Build the orphan snapshot")
    built = _bash(snap["run"], source_repo, {"SOURCE_SHA": source_sha})
    assert built.returncode == 0, built.stdout + built.stderr
    head = _git(source_repo, "rev-parse", "HEAD")
    trailers = _git(source_repo, "log", "-1", "--format=%(trailers:only,unfold)")
    assert f"Source-Commit: {source_sha}" in trailers and "Private-CI: passed" in trailers

    ok = _bash(_mirror_check(), source_repo, {"GITHUB_SHA": head})
    assert ok.returncode == 0, ok.stdout + ok.stderr


@pytest.mark.parametrize(
    "message",
    [
        "snapshot with no trailers",
        "snapshot\n\nSource-Commit: " + "a" * 40 + "\nPrivate-CI: failed",
        "snapshot\n\nPrivate-CI: passed",
        "snapshot\n\nSource-Commit: abc1234\nPrivate-CI: passed",
    ],
)
def test_the_mirror_refuses_a_commit_without_the_verdict(source_repo, message):
    _git(
        source_repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        message,
    )
    head = _git(source_repo, "rev-parse", "HEAD")
    res = _bash(_mirror_check(), source_repo, {"GITHUB_SHA": head})
    assert res.returncode != 0, f"accepted {message!r}: {res.stdout}"


def test_the_mirror_build_trusts_the_trailer_not_its_own_test_runs():
    doc = _load("docker-images.yml")
    jobs = doc["jobs"]
    trust = jobs["private-ci"]
    assert not _runs_here(trust), "the trust check belongs to the mirror"
    build = jobs["build"]
    needs = build["needs"] if isinstance(build["needs"], list) else [build["needs"]]
    assert "private-ci" in needs
    assert "always()" not in str(
        build.get("if", "")
    ), "always() would build over a failed trust check"
    for name, job in jobs.items():
        perms = job.get("permissions") or {}
        assert (
            perms.get("actions") is None
        ), f"{name} can read workflow runs - nothing here should wait on tests"
        for step in job.get("steps", []):
            run = step.get("run", "")
            assert (
                "/actions/runs" not in run and "Backend Tests" not in run
            ), f"{name}/{step.get('name')} still waits on the mirror's own test runs"


# ── 3. the shards and their aggregator ───────────────────────────────────────────────


def _shard_jobs(doc: dict) -> dict[str, dict]:
    return {
        k: j
        for k, j in doc["jobs"].items()
        if "shard" in ((j.get("strategy") or {}).get("matrix") or {})
    }


def test_every_shard_job_runs_one_slice_of_n_and_the_aggregator_needs_them_all():
    doc = _load("backend-tests.yml")
    shards = _shard_jobs(doc)
    assert shards, "Backend Tests is not sharded"
    for name, job in shards.items():
        values = job["strategy"]["matrix"]["shard"]
        n = len(values)
        assert values == list(range(1, n + 1)), f"{name}: shards must be 1..{n}, got {values}"
        assert (
            job["env"]["CI_SHARD"] == f"${{{{ matrix.shard }}}}/{n}"
        ), f"{name}: CI_SHARD disagrees with the matrix"
        assert job["strategy"].get("fail-fast") is False
    agg = doc["jobs"]["all-shards"]
    needs = agg["needs"] if isinstance(agg["needs"], list) else [agg["needs"]]
    assert set(needs) == set(shards), "the aggregator must need every shard job"
    assert "always()" in str(
        agg.get("if", "")
    ), "a skipped aggregator reads as neither pass nor fail"


@pytest.mark.parametrize(
    "results, ok",
    [
        ({"a": "success", "b": "success"}, True),
        ({"a": "success", "b": "failure"}, False),
        ({"a": "success", "b": "cancelled"}, False),
        ({"a": "skipped"}, False),
        ({}, False),
    ],
)
def test_the_aggregator_passes_only_when_every_shard_passed(tmp_path, results, ok):
    agg = _load("backend-tests.yml")["jobs"]["all-shards"]
    step = _step(agg, "Require every shard to have passed")
    needs = json.dumps({k: {"result": v, "outputs": {}} for k, v in results.items()})
    res = _bash(step["run"], tmp_path, {"RESULTS": needs})
    assert (res.returncode == 0) is ok, res.stdout + res.stderr


# ── 4. the gate script's decisions ───────────────────────────────────────────────────


def _gate_module():
    spec = importlib.util.spec_from_file_location("private_ci_gate", GATE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Api:
    """A fake of the three GitHub endpoints the gate reads."""

    def __init__(self, by_sha=None, history=None, jobs=None):
        self.by_sha = by_sha or []  # list of successive answers for head_sha=<sha>
        self.history = history or []
        self.jobs = jobs or {}
        self.calls = 0

    def __call__(self, path: str):
        if "/jobs?" in path:
            run_id = int(path.split("/runs/")[1].split("/")[0])
            return {"jobs": self.jobs.get(run_id, [])}
        if "head_sha=" in path:
            answer = self.by_sha[min(self.calls, len(self.by_sha) - 1)] if self.by_sha else []
            self.calls += 1
            return {"workflow_runs": answer}
        return {"workflow_runs": self.history}


def _run(i, sha, status="completed", conclusion="success", event="push"):
    return {
        "id": i,
        "head_sha": sha,
        "status": status,
        "conclusion": conclusion,
        "event": event,
        "created_at": f"2026-10-20T00:00:{i:02d}Z",
        "run_attempt": 1,
    }


@pytest.fixture
def history_repo(tmp_path):
    """main: A (tests passed) -> B (docs only) -> C (touches the tested path)."""
    repo = tmp_path / "repo"
    (repo / ".github" / "workflows").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / ".github" / "workflows" / "filtered.yml").write_text(
        "name: F\non:\n  push:\n    branches: [main]\n    paths: ['src/**']\n"
        "jobs:\n  t: {runs-on: x, steps: []}\n"
    )
    (repo / ".github" / "workflows" / "sharded.yml").write_text(
        "name: S\non:\n  push:\n    branches: [main]\njobs:\n  t: {runs-on: x, steps: []}\n"
        "  all-shards:\n    name: all shards\n    needs: [t]\n    runs-on: x\n    steps: []\n"
    )
    (repo / "src").mkdir()
    shas = []
    for name, path in (("A", "src/a.py"), ("B", "docs.md"), ("C", "src/c.py")):
        (repo / path).write_text(name)
        _git(repo, "add", "-A")
        _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", name)
        shas.append(_git(repo, "rev-parse", "HEAD"))
    return SimpleNamespace(repo=repo, a=shas[0], b=shas[1], c=shas[2])


def _gate(mod, repo, sha, api):
    return mod.Gate(
        "o/r",
        sha,
        repo,
        get=api,
        sleep=lambda s: None,
        clock=iter(range(0, 10**6, 60)).__next__,
        timeout=600,
        grace=120,
        interval=30,
    )


def test_gate_passes_a_green_run_for_the_commit(history_repo):
    mod = _gate_module()
    h = history_repo
    ok, msg = _gate(mod, h.repo, h.c, _Api(by_sha=[[_run(1, h.c)]])).check("filtered.yml")
    assert ok, msg


def test_gate_waits_for_a_running_run_then_passes(history_repo):
    mod = _gate_module()
    h = history_repo
    api = _Api(
        by_sha=[
            [_run(1, h.c, "in_progress", None)],
            [_run(1, h.c, "in_progress", None)],
            [_run(1, h.c)],
        ]
    )
    ok, msg = _gate(mod, h.repo, h.c, api).check("filtered.yml")
    assert ok and api.calls == 3, msg


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out"])
def test_gate_refuses_a_red_run_for_the_commit(history_repo, conclusion):
    mod = _gate_module()
    h = history_repo
    ok, msg = _gate(mod, h.repo, h.c, _Api(by_sha=[[_run(1, h.c, conclusion=conclusion)]])).check(
        "filtered.yml"
    )
    assert not ok and conclusion in msg


def test_gate_ignores_a_pull_request_run(history_repo):
    mod = _gate_module()
    h = history_repo
    ok, _ = _gate(mod, h.repo, h.c, _Api(by_sha=[[_run(1, h.c, event="pull_request")]])).check(
        "filtered.yml"
    )
    assert not ok


def test_gate_requires_the_all_shards_job(history_repo):
    mod = _gate_module()
    h = history_repo
    run = _run(7, h.c)
    no_agg = _Api(by_sha=[[run]], jobs={7: [{"name": "t", "conclusion": "success"}]})
    assert not _gate(mod, h.repo, h.c, no_agg).check("sharded.yml")[0]
    red_agg = _Api(by_sha=[[run]], jobs={7: [{"name": "all shards", "conclusion": "failure"}]})
    assert not _gate(mod, h.repo, h.c, red_agg).check("sharded.yml")[0]
    green = _Api(by_sha=[[run]], jobs={7: [{"name": "all shards", "conclusion": "success"}]})
    assert _gate(mod, h.repo, h.c, green).check("sharded.yml")[0]


def test_gate_falls_back_only_when_the_tested_paths_are_unchanged(history_repo):
    mod = _gate_module()
    h = history_repo
    # B touched only docs: no run of the filtered workflow, and A passed with identical src/.
    ok, msg = _gate(mod, h.repo, h.b, _Api(history=[_run(1, h.a)])).check("filtered.yml")
    assert ok and h.a in msg, msg
    # C changed src/ since A: A's verdict does not cover it.
    ok, _ = _gate(mod, h.repo, h.c, _Api(history=[_run(1, h.a)])).check("filtered.yml")
    assert not ok
    # A newer commit that is NOT an ancestor never vouches for an older one - even when, as
    # B here, its tested paths are identical: only the ancestry rule refuses this one.
    ok, _ = _gate(mod, h.repo, h.a, _Api(history=[_run(2, h.b)])).check("filtered.yml")
    assert not ok


def test_gate_refuses_a_missing_run_for_an_unfiltered_workflow(history_repo):
    mod = _gate_module()
    h = history_repo
    api = _Api(history=[_run(1, h.b)], jobs={1: [{"name": "all shards", "conclusion": "success"}]})
    ok, msg = _gate(mod, h.repo, h.c, api).check("sharded.yml")
    assert not ok and "no path filter" in msg


def test_gate_refuses_an_empty_requirement(monkeypatch):
    mod = _gate_module()
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("SHA", "a" * 40)
    monkeypatch.setenv("REQUIRED_WORKFLOWS", "  \n")
    assert mod.main() == 1
