"""The mirror publishes the WSP worker image's ``prod`` target, never ``test`` (feature 016; P62,
R2A-19; FTASKS 8.4; FTDD §11).

``backend/Dockerfile.wsp-worker`` has two targets and the LAST one is ``test`` (pytest, the
gateway's allow-list). A build with no ``target`` therefore ships pytest. These tests run the
workflow's own shell blocks rather than reading their text (a regex over a workflow matches
nothing on an unexpected layout and asserts nothing):

- "Resolve image build parameters" with ``matrix.name = wsp-worker`` writes ``target=prod``, the
  worker Dockerfile, the root context and the image name;
- the build step hands ``steps.img.outputs.target`` to ``docker/build-push-action``;
- "Detect changed images" picks ``wsp-worker`` for a change to the worker package, its Dockerfile
  or its lock, lists it in a full rebuild, and does not pick it for a frontend-only change.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github" / "workflows" / "docker-images.yml"
DOCKERFILE = REPO / "backend" / "Dockerfile.wsp-worker"


def _doc() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(WORKFLOW.read_text("utf-8"))
    return loaded


def _step(job: str, name: str) -> dict[str, Any]:
    steps = _doc()["jobs"][job]["steps"]
    found = [s for s in steps if s.get("name") == name]
    assert len(found) == 1, f"{job}: expected one step named {name!r}"
    return dict(found[0])


def _outputs(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        out[key] = value
    return out


def _resolve(tmp_path: Path, image: str) -> subprocess.CompletedProcess[str]:
    script = _step("build", "Resolve image build parameters")["run"]
    script = script.replace("${{ matrix.name }}", image)
    assert "${{" not in script, "the step reads an expression this test does not supply"
    out = tmp_path / "out"
    out.write_text("")
    return subprocess.run(
        ["bash", "-e", "-c", script],
        env={**os.environ, "GITHUB_OUTPUT": str(out)},
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_worker_image_builds_the_prod_target(tmp_path: Path) -> None:
    r = _resolve(tmp_path, "wsp-worker")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _outputs(tmp_path / "out") == {
        "repo": "hitsai/miforge-wsp-worker",
        "context": ".",
        "dockerfile": "./backend/Dockerfile.wsp-worker",
        "target": "prod",
    }


@pytest.mark.parametrize("image", ["backend", "frontend"])
def test_the_other_images_keep_their_default_target(tmp_path: Path, image: str) -> None:
    r = _resolve(tmp_path, image)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _outputs(tmp_path / "out").get("target", "") == ""


def test_the_build_step_passes_the_resolved_target() -> None:
    build = [
        s
        for s in _doc()["jobs"]["build"]["steps"]
        if str(s.get("uses", "")).startswith("docker/build-push-action")
    ]
    assert len(build) == 1
    assert build[0]["with"]["target"] == "${{ steps.img.outputs.target }}"
    assert build[0]["with"]["file"] == "${{ steps.img.outputs.dockerfile }}"


def test_the_last_stage_of_the_worker_dockerfile_is_test() -> None:
    """Why the pin matters: an untargeted build would take the last stage, which ships pytest."""
    stages = [
        line.split()[-1]
        for line in DOCKERFILE.read_text().splitlines()
        if line.startswith("FROM ") and " AS " in line
    ]
    assert stages == ["prod", "test"]


# ----------------------------------------------------------------------------- detect


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def _detect(tmp_path: Path, changed: list[str] | None) -> list[str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "ci@example.invalid")
    _git(repo, "config", "user.name", "ci")
    (repo / "README").write_text("base\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    for rel in changed or []:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("changed\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "change")
    head = _git(repo, "rev-parse", "HEAD")
    out = tmp_path / "out"
    out.write_text("")
    script = _step("detect", "Detect changed images")["run"]
    env = {
        **os.environ,
        "GITHUB_OUTPUT": str(out),
        "GITHUB_SHA": head,
        "BEFORE": base if changed is not None else "",
        "REF": "refs/heads/main",
    }
    r = subprocess.run(
        ["bash", "-e", "-c", script], cwd=repo, env=env, capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stdout + r.stderr
    images: list[str] = json.loads(_outputs(out)["images"])
    return images


@pytest.mark.parametrize(
    "path",
    [
        "backend/src/wsp_worker/supervisor.py",
        "backend/Dockerfile.wsp-worker",
        "backend/requirements-wsp-worker.txt",
        "backend/scripts/check_installed_matches_lock.py",
        ".dockerignore",
        "LICENSE",
    ],
)
def test_a_change_to_the_worker_rebuilds_its_image(tmp_path: Path, path: str) -> None:
    assert "wsp-worker" in _detect(tmp_path, [path])


def test_a_frontend_change_does_not_rebuild_the_worker(tmp_path: Path) -> None:
    assert _detect(tmp_path, ["frontend/src/App.tsx"]) == ["frontend"]


def test_a_full_rebuild_lists_the_worker(tmp_path: Path) -> None:
    assert set(_detect(tmp_path, None)) == {"backend", "frontend", "wsp-worker"}
