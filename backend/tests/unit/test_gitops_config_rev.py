"""A ConfigMap change restarts its consumers: the ``rebrand.hitsai/config-rev`` bump (CF-27).

Pods read ``miforge-config`` only at start (``envFrom``), and a push that changes only
``gitops/`` builds no image, so a changed value would sit in the cluster while every process
kept the old one - silently. The commit that changes ``gitops/pilot/config.yaml``'s ``data``
must therefore also change the pod-template annotation ``rebrand.hitsai/config-rev`` on every
Deployment that reads the ConfigMap (consumers DERIVED from ``envFrom``/``configMapKeyRef``,
never listed). EVERY commit of the push is compared with its own parent, from the push's base
(``CONFIG_REV_BASE`` = ``github.event.before`` in CI, else the merge-base with ``origin/main``,
else ``HEAD~1``); CI checks out full history (``fetch-depth: 0``). A base CI names but the clone
cannot resolve (a force push's old tip) is fetched, and if that fails the test FAILS: it never
silently narrows to fewer commits. Every git call checks its exit status (fails closed).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.support import legacy_identifiers as li
from tests.support.mirror_snapshot import make_repo, mirror_snapshot

REPO_ROOT = li.REPO_ROOT
CONFIG = "gitops/pilot/config.yaml"
ANNOTATION = "rebrand.hitsai/config-rev"


def _docs(text: str) -> list[dict[str, Any]]:
    return [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]


def _config_data(docs: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for doc in docs:
        if doc.get("kind") == "ConfigMap" and doc["metadata"]["name"] == name:
            return dict(doc.get("data") or {})
    raise AssertionError(f"no ConfigMap {name}")


def _reads(node: Any, name: str) -> bool:
    if isinstance(node, dict):
        ref = node.get("configMapRef") or node.get("configMapKeyRef")
        if isinstance(ref, dict) and ref.get("name") == name:
            return True
        return any(_reads(v, name) for v in node.values())
    if isinstance(node, list):
        return any(_reads(v, name) for v in node)
    return False


def consumers(docs: list[dict[str, Any]], name: str) -> dict[str, str | None]:
    """Deployment name -> its config-rev annotation, for every Deployment reading ``name``."""
    out: dict[str, str | None] = {}
    for doc in docs:
        if doc.get("kind") in ("Deployment", "StatefulSet") and _reads(doc, name):
            template = doc["spec"]["template"].get("metadata") or {}
            out[doc["metadata"]["name"]] = (template.get("annotations") or {}).get(ANNOTATION)
    return out


def stale_consumers(
    old_config: str, new_config: str, old_manifests: str, new_manifests: str, name: str
) -> list[str]:
    """Consumers whose annotation did not change although the ConfigMap's data did."""
    if _config_data(_docs(old_config), name) == _config_data(_docs(new_config), name):
        return []
    before = consumers(_docs(old_manifests), name)
    after = consumers(_docs(new_manifests), name)
    assert after, f"nothing reads {name}; the derivation found no consumer"
    return sorted(d for d, rev in after.items() if rev is None or rev == before.get(d))


class GitError(RuntimeError):
    """A git call the guard depends on failed: the verdict would be built on nothing."""


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run git; with ``check`` (the default) a non-zero exit RAISES, so an empty answer from a
    failed command can never read as "no commits" or "no manifests" (fail closed)."""
    done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if check and done.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed ({done.returncode}): {done.stderr.strip()}")
    return done


def _show(root: Path, commit: str, path: str) -> str | None:
    """A file at a commit, or ``None`` when that commit does not have it (asked, not assumed)."""
    if _git(root, "cat-file", "-e", f"{commit}:{path}", check=False).returncode != 0:
        _git(root, "rev-parse", "--verify", "-q", f"{commit}^{{commit}}")  # the commit exists
        return None
    return _git(root, "show", f"{commit}:{path}").stdout


def _manifest_paths(root: Path, commit: str) -> list[str]:
    out = _git(root, "ls-tree", "-r", "--name-only", commit, "--", "gitops/pilot/").stdout
    return [p for p in out.splitlines() if p.endswith(".yaml") and p != CONFIG]


def _resolvable(root: Path, commit: str) -> bool:
    return _git(root, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode == 0


def push_base(root: Path, env: dict[str, str]) -> str | None:
    """Where this push starts: CI's ``github.event.before`` (``CONFIG_REV_BASE``); else the
    merge-base with ``origin/main`` when it is not HEAD itself; else ``HEAD~1``; ``None`` when
    HEAD has no parent (the mirror's one-commit snapshot).

    A ``CONFIG_REV_BASE`` the clone cannot resolve (a force push's old tip) is FETCHED; if it
    still cannot be resolved this raises - falling back to ``HEAD~1`` would silently check
    fewer commits than the push carries. An all-zero ``before`` is a new branch: no old tip
    exists, and the fallbacks apply.

    A parentless HEAD has no parent to compare with, whatever ``CONFIG_REV_BASE`` says: ``None``.
    That is the mirror's snapshot, where CI's ``github.event.before`` is the PREVIOUS snapshot -
    resolvable, yet not an ancestor (the mirror run of 2026-11-08 died in ``HEAD~1`` on it). The
    caller decides whether ``None`` may skip (the mirror snapshot) or must fail (anywhere else)."""
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    if _git(root, "rev-parse", "--verify", "-q", "HEAD~1", check=False).returncode != 0:
        return None
    before = env.get("CONFIG_REV_BASE", "").strip()
    if before and set(before) != {"0"}:
        if not _resolvable(root, before):
            _git(root, "fetch", "--quiet", "--no-tags", "origin", before, check=False)
        if not _resolvable(root, before):
            raise GitError(
                f"CONFIG_REV_BASE {before} is not in this clone and could not be fetched "
                f"(a force push?). Refusing to narrow the config-rev check to fewer commits."
            )
        return before
    merge_base = _git(root, "merge-base", "HEAD", "origin/main", check=False)
    if merge_base.returncode == 0 and merge_base.stdout.strip() not in ("", head):
        return merge_base.stdout.strip()
    return _git(root, "rev-parse", "--verify", "-q", "HEAD~1").stdout.strip()


def stale_commits(root: Path, base: str) -> list[str]:
    """Every commit in ``base..HEAD`` that changed the ConfigMap's data without changing each
    consumer's annotation, compared with its own parent - so an early commit of a multi-commit
    push cannot hide behind a later bump."""
    commits = _git(root, "rev-list", "--reverse", f"{base}..HEAD").stdout.split()
    problems = []
    for commit in commits:
        old_config = _show(root, f"{commit}~1", CONFIG)
        new_config = _show(root, commit, CONFIG)
        if old_config is None or new_config is None:
            continue
        old = "\n---\n".join(
            _show(root, f"{commit}~1", p) or "" for p in _manifest_paths(root, f"{commit}~1")
        )
        new = "\n---\n".join(_show(root, commit, p) or "" for p in _manifest_paths(root, commit))
        stale = stale_consumers(old_config, new_config, old, new, "miforge-config")
        if stale:
            problems.append(f"{commit[:9]}: {stale}")
    return problems


def push_problems(root: Path, env: dict[str, str]) -> list[str] | None:
    """The verdict the real-repo test reads: ``None`` when there is no base, else the stale
    commits since it. One function, so the wiring is exercised on a scratch repository too
    (review round 2: replacing ``base`` by ``"HEAD"`` in the test body stayed green)."""
    base = push_base(root, env)
    return None if base is None else stale_commits(root, base)


def test_configmap_change_bumps_consumers() -> None:
    import os

    problems = push_problems(REPO_ROOT, dict(os.environ))
    if problems is None:
        if mirror_snapshot(REPO_ROOT):
            pytest.skip(
                "MIRROR: this checkout is the public mirror's parentless snapshot (Source-Commit / "
                "Private-CI trailers): no parent to compare; the private repo's CI ran this check"
            )
        pytest.fail("no base commit: check out the push's history (fetch-depth: 0)")
    new_manifests = "\n---\n".join(
        (REPO_ROOT / p).read_text("utf-8") for p in _manifest_paths(REPO_ROOT, "HEAD")
    )
    assert consumers(_docs(new_manifests), "miforge-config"), "no consumer of miforge-config found"
    assert not problems, (
        f"{CONFIG} changed in these commits without bumping {ANNOTATION} on its consumers: "
        f"{problems}. Bump the annotation in the same commit, or the running pods keep the old "
        f"values (CF-27)."
    )


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message)
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def test_every_commit_of_a_push_is_checked(tmp_path: Path) -> None:
    """A config change without a bump in the FIRST commit of a two-commit push is found
    (HEAD~1 alone would compare only the second, unrelated commit)."""
    _git(tmp_path, "init", "-q")
    pilot = tmp_path / "gitops" / "pilot"
    pilot.mkdir(parents=True)
    (pilot / "config.yaml").write_text(CONFIG_V1.replace("name: c", "name: miforge-config"))
    (pilot / "backend.yaml").write_text(_deploy("a").replace("name: c}", "name: miforge-config}"))
    base = _commit(tmp_path, "base")
    (pilot / "config.yaml").write_text(CONFIG_V2.replace("name: c", "name: miforge-config"))
    _commit(tmp_path, "config change, no bump")
    (pilot / "other.yaml").write_text("kind: Service\nmetadata: {name: s}\n")
    _commit(tmp_path, "unrelated")
    assert len(stale_commits(tmp_path, base)) == 1
    assert stale_commits(tmp_path, f"{base}~0") == stale_commits(tmp_path, base)
    assert push_base(tmp_path, {"CONFIG_REV_BASE": base}) == base
    assert push_base(tmp_path, {"CONFIG_REV_BASE": "0" * 40}) != base  # a new branch: fallback
    head_parent = _git(tmp_path, "rev-parse", "HEAD~1").stdout.strip()
    assert stale_commits(tmp_path, head_parent) == [], "HEAD~1 alone misses the first commit"
    # The verdict the real-repo test reads, through the same wiring (review round 2).
    problems = push_problems(tmp_path, {"CONFIG_REV_BASE": base})
    assert problems is not None and len(problems) == 1, problems


CONFIG_V1 = "kind: ConfigMap\nmetadata: {name: c}\ndata: {A: '1'}\n"
CONFIG_V2 = "kind: ConfigMap\nmetadata: {name: c}\ndata: {A: '2'}\n"


def _deploy(rev: str | None, reads: bool = True) -> str:
    annotations = f"annotations: {{{ANNOTATION}: '{rev}'}}" if rev else "annotations: {}"
    env = "envFrom: [{configMapRef: {name: c}}]" if reads else "env: []"
    return (
        f"kind: Deployment\nmetadata: {{name: d}}\nspec:\n  template:\n    metadata: "
        f"{{{annotations}}}\n    spec:\n      containers: [{{name: x, {env}}}]\n"
    )


def test_self_test_on_fixture_pair() -> None:
    assert stale_consumers(CONFIG_V1, CONFIG_V1, _deploy("a"), _deploy("a"), "c") == []
    assert stale_consumers(CONFIG_V1, CONFIG_V2, _deploy("a"), _deploy("a"), "c") == ["d"]
    assert stale_consumers(CONFIG_V1, CONFIG_V2, _deploy(None), _deploy(None), "c") == ["d"]
    assert stale_consumers(CONFIG_V1, CONFIG_V2, _deploy("a"), _deploy("b"), "c") == []
    with pytest.raises(AssertionError):
        stale_consumers(CONFIG_V1, CONFIG_V2, _deploy("a", False), _deploy("a", False), "c")
    assert Path(REPO_ROOT / CONFIG).is_file()


def test_a_parentless_head_has_no_base_whatever_ci_says(tmp_path: Path) -> None:
    """The mirror's snapshot: CI's CONFIG_REV_BASE names the PREVIOUS snapshot, which the clone
    can resolve but which is no ancestor. That is "no base" (None), never a crash in ``HEAD~1``
    (the mirror's Backend Tests, 2026-11-08) and never a vacuous empty verdict."""
    previous = make_repo(tmp_path / "previous", "mirror")
    head = make_repo(tmp_path / "head", "mirror")
    _git(head, "fetch", "-q", str(previous), "HEAD:refs/prev")
    before = _git(head, "rev-parse", "refs/prev").stdout.strip()
    _git(head, "update-ref", "-d", "refs/prev")
    assert _resolvable(head, before)
    assert push_base(head, {"CONFIG_REV_BASE": before}) is None
    assert push_problems(head, {"CONFIG_REV_BASE": before}) is None
    assert push_problems(head, {}) is None


@pytest.mark.parametrize("shape", ["none", "shallow-private", "orphan-plain", "mirror"])
def test_no_base_skips_only_on_the_mirror_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """With no base, the real-repo test skips on the mirror snapshot and FAILS on any other tree:
    a private clone without history (fetch-depth 1), a parentless commit without the sync's
    trailers, or no repository at all must never skip the config-rev check."""
    tree = make_repo(tmp_path / "t", shape)
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tree)
    monkeypatch.delenv("CONFIG_REV_BASE", raising=False)
    with pytest.raises((pytest.skip.Exception, pytest.fail.Exception, GitError)) as caught:
        test_configmap_change_bumps_consumers()
    skipped = caught.type is pytest.skip.Exception
    assert skipped is (shape == "mirror"), (shape, str(caught.value))
