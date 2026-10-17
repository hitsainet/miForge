"""F11's hostIPC grant is opt-in: absent from what ArgoCD deploys, present when applied.

Suite decision **D1** (operator, 2026-10-16): *"`hostIPC: true` on both pods, shipped as an
opt-in overlay applied only when hot-update is used."* miForge's half is the kustomize
Component ``gitops/components/hostipc/`` and, for Compose, the ``hotupdate``-profiled
``celery-train-hotupdate`` service. F11 task 14.2.

Two directions are pinned, and both matter:

* **The default must not carry it.** hostIPC shares the node's IPC namespace with every
  container in the pod. A component accidentally listed in ``gitops/pilot/kustomization.yaml``
  (the path the ``miforge`` Application reads) would grant it permanently, on every deploy,
  and nothing else would notice.
* **The opt-in must actually produce it.** A component whose patch targets a renamed
  Deployment, or the wrong path, renders cleanly and grants nothing — the hot-update session
  then fails at the first CUDA IPC rebuild, on the GPU node, which is the most expensive
  place to find a typo.

The assertions RENDER the manifests (``kubectl kustomize``, ``docker compose config``)
rather than reading the YAML, because a static read of the component would pass while its
target matched nothing. Where a tool is missing the render test **skips loudly locally and
fails in CI** (``CI`` is set on every GitHub runner, and ubuntu-latest ships both tools), so
the guard cannot fail open on the runner that matters.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[3]
GITOPS = REPO_ROOT / "gitops"
DEPLOYED = GITOPS / "pilot"
COMPONENT = GITOPS / "components" / "hostipc"
PREVIEW = GITOPS / "overlays" / "hotupdate"
APPLICATION = DEPLOYED / "argocd" / "miforge-app.yaml"
COMPOSE = REPO_ROOT / "docker-compose.yml"

DEPLOYMENT = "miforge-backend"


def _require_tool(*argv: str) -> None:
    exe = shutil.which(argv[0])
    if exe is None:
        message = f"{argv[0]} is not installed; the hostIPC render check cannot run here"
        if os.environ.get("CI"):
            pytest.fail(message + " - and CI must run it (ubuntu-latest ships it)")
        pytest.skip(message)


def _kustomize(path: Path) -> list[dict[str, Any]]:
    _require_tool("kubectl")
    out = subprocess.run(
        ["kubectl", "kustomize", str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    return [doc for doc in yaml.safe_load_all(out) if doc]


def _deployments(docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {d["metadata"]["name"]: d for d in docs if d.get("kind") == "Deployment"}


def _pod_specs_with_host_ipc(docs: list[dict[str, Any]]) -> list[str]:
    """Every workload whose pod template sets hostIPC, by name."""
    found = []
    for doc in docs:
        spec = doc.get("spec", {}).get("template", {}).get("spec", {})
        if spec.get("hostIPC"):
            found.append(f"{doc['kind']}/{doc['metadata']['name']}")
    return found


def _compose_config(*profiles: str) -> dict[str, Any]:
    _require_tool("docker")
    argv = ["docker", "compose", "-f", str(COMPOSE)]
    for profile in profiles:
        argv += ["--profile", profile]
    argv += ["config", "--format", "json"]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 and "is not a docker command" in result.stderr:
        if os.environ.get("CI"):
            pytest.fail("docker compose is unavailable in CI: " + result.stderr)
        pytest.skip("docker compose plugin unavailable: " + result.stderr.strip())
    assert result.returncode == 0, result.stderr
    loaded: dict[str, Any] = json.loads(result.stdout)
    return loaded


#: ⚠ HOT-UPDATE SESSION STATE (suite decision D38, 2026-10-17). The deployed path carries
#: the component for the run window ONLY; the revert commit restores the "no components"
#: assertions this class held before (``git log -p`` this file).
SESSION_COMPONENTS = ["../components/hostipc"]


def _kustomize_deployed_without_components(tmp_path: Path) -> list[dict[str, Any]]:
    """The deployed render with its ``components:`` line removed - what is left is every
    OTHER source of hostIPC, and there must be none."""
    copy = tmp_path / "gitops"
    shutil.copytree(GITOPS, copy)
    kfile = copy / "pilot" / "kustomization.yaml"
    doc = yaml.safe_load(kfile.read_text("utf-8"))
    doc.pop("components", None)
    kfile.write_text(yaml.safe_dump(doc), "utf-8")
    return _kustomize(copy / "pilot")


class TestTheDeployedPathCarriesHostIpcOnlyThroughTheComponent:
    def test_the_application_reads_gitops_pilot(self) -> None:
        """The premise of every other assertion here: what ArgoCD deploys is gitops/pilot."""
        docs = [d for d in yaml.safe_load_all(APPLICATION.read_text("utf-8")) if d]
        apps = [d for d in docs if d.get("kind") == "Application"]
        assert len(apps) == 1
        assert apps[0]["spec"]["source"]["path"] == "gitops/pilot"

    def test_the_deployed_path_lists_exactly_the_hostipc_component(self) -> None:
        """Static: during the D38 session gitops/pilot names the hostIPC component and
        nothing else. Reverting the session must change this test back in the same commit."""
        kustomization = yaml.safe_load((DEPLOYED / "kustomization.yaml").read_text("utf-8"))
        assert kustomization.get("components") == SESSION_COMPONENTS

    def test_the_deployed_render_sets_host_ipc_on_the_trainer_pod_only(self) -> None:
        docs = _kustomize(DEPLOYED)
        assert DEPLOYMENT in _deployments(docs), "the render lost the backend Deployment"
        assert _pod_specs_with_host_ipc(docs) == [f"Deployment/{DEPLOYMENT}"]

    def test_the_component_is_the_only_source_of_host_ipc(self, tmp_path: Path) -> None:
        """Strip the ``components:`` line and nothing may set hostIPC: a field written into
        backend.yaml directly would survive the revert and grant it permanently."""
        docs = _kustomize_deployed_without_components(tmp_path)
        assert DEPLOYMENT in _deployments(docs)
        assert _pod_specs_with_host_ipc(docs) == []
        assert "hostIPC" not in (DEPLOYED / "backend.yaml").read_text("utf-8")


class TestTheComponentGrantsIt:
    def test_the_component_render_sets_host_ipc_on_the_trainer_pod_only(self) -> None:
        docs = _kustomize(PREVIEW)
        # Exactly one workload, and it is the pod the train worker lives in.
        assert _pod_specs_with_host_ipc(docs) == [f"Deployment/{DEPLOYMENT}"]
        backend = _deployments(docs)[DEPLOYMENT]
        assert backend["spec"]["template"]["spec"]["hostIPC"] is True
        names = [c["name"] for c in backend["spec"]["template"]["spec"]["containers"]]
        assert any("train" in n for n in names), names

    def test_the_component_changes_nothing_else(self, tmp_path: Path) -> None:
        """The overlay is the deployed render plus one field - not a fork of it."""
        default = _kustomize_deployed_without_components(tmp_path)
        opted = _kustomize(PREVIEW)
        for doc in opted:
            spec = doc.get("spec", {}).get("template", {}).get("spec", {})
            spec.pop("hostIPC", None)
        assert opted == default

    def test_it_is_a_component_not_a_kustomization(self) -> None:
        """A `Kustomization` here could be pointed at by an Application by mistake; a
        Component cannot be rendered on its own."""
        doc = yaml.safe_load((COMPONENT / "kustomization.yaml").read_text("utf-8"))
        assert doc["kind"] == "Component"


class TestComposeHotUpdateProfile:
    def test_the_default_compose_has_no_host_ipc(self) -> None:
        config = _compose_config()
        assert "celery-train-hotupdate" not in config["services"]
        assert {name: s.get("ipc") for name, s in config["services"].items() if s.get("ipc")} == {}

    def test_the_hotupdate_profile_runs_the_train_worker_with_host_ipc(self) -> None:
        config = _compose_config("hotupdate")
        service = config["services"]["celery-train-hotupdate"]
        assert service["ipc"] == "host"
        # It is the train worker, by command - not a copy that consumes a different queue.
        default_train = config["services"]["celery-train"]
        assert service["command"] == default_train["command"]
        assert "train" in service["command"]
        # And only it: the default worker stays unchanged under the profile.
        assert default_train.get("ipc") is None
