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
import shlex
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


class TestTheDeployedPathDoesNotCarryHostIpc:
    """hostIPC is opt-in per suite decision D1, and on only for an operator-approved run
    window (D38). F11 window 2 closed 2026-10-18; this class pins the OFF state again."""

    def test_the_application_reads_gitops_pilot(self) -> None:
        """The premise of every other assertion here: what ArgoCD deploys is gitops/pilot."""
        docs = [d for d in yaml.safe_load_all(APPLICATION.read_text("utf-8")) if d]
        apps = [d for d in docs if d.get("kind") == "Application"]
        assert len(apps) == 1
        assert apps[0]["spec"]["source"]["path"] == "gitops/pilot"

    def test_the_deployed_path_does_not_include_it(self) -> None:
        """Static: gitops/pilot names no component. Enabling it is a deliberate one-line
        commit that must change this test too - see the component's header."""
        kustomization = yaml.safe_load((DEPLOYED / "kustomization.yaml").read_text("utf-8"))
        assert not kustomization.get("components"), (
            "gitops/pilot/kustomization.yaml lists a component. hostIPC is opt-in per suite "
            "decision D1; if this is a deliberate hot-update session, change this test in "
            "the same commit and revert both when it ends."
        )

    def test_the_default_render_has_no_host_ipc_and_runs_everything_as_10001(self) -> None:
        docs = _kustomize(DEPLOYED)
        assert DEPLOYMENT in _deployments(docs), "the render lost the backend Deployment"
        assert _pod_specs_with_host_ipc(docs) == []
        pod = _deployments(docs)[DEPLOYMENT]["spec"]["template"]["spec"]
        assert "hotupdate-shared-dirs" not in [c["name"] for c in pod["initContainers"]]
        trainer = next(c for c in pod["containers"] if c["name"] == "worker-train")
        assert (trainer.get("securityContext") or {}).get("runAsUser") in (None, 10001)

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
        """The overlay is the deployed render plus the session's additions - not a fork."""
        default = _kustomize_deployed_without_components(tmp_path)
        opted = _kustomize(PREVIEW)
        for doc in opted:
            spec = doc.get("spec", {}).get("template", {}).get("spec", {})
            spec.pop("hostIPC", None)
            if spec.get("initContainers"):
                spec["initContainers"] = [
                    c for c in spec["initContainers"] if c["name"] != SHARED_DIRS_INIT
                ]
            for container in spec.get("containers", []):
                if container["name"] != TRAINER:
                    continue
                container["securityContext"].pop("runAsUser")
                container["securityContext"].pop("runAsGroup")
                container["env"] = [
                    e for e in container["env"] if e["name"] not in ("HOME", "HF_HOME")
                ]
                container["command"] = _default_trainer(default)["command"]
        # Containers compared BY NAME: the strategic merge moves the patched worker-train to
        # the front of the list (harmless to Kubernetes; during a session `kubectl exec`
        # without -c lands in worker-train rather than api - always pass -c).
        for doc in (*opted, *default):
            spec = doc.get("spec", {}).get("template", {}).get("spec", {})
            if spec.get("containers"):
                spec["containers"] = sorted(spec["containers"], key=lambda c: c["name"])
        assert opted == default


#: Step 2 (operator-approved 2026-10-17): the uid pairing. miLLM's uid, which CUDA's 0600
#: /dev/shm IPC files require (F11 14.3 failed on exactly this); miForge's own group.
MILLM_UID = 1000
MIFORGE_GID = 10001
TRAINER = "worker-train"
SHARED_DIRS_INIT = "hotupdate-shared-dirs"
MILLM_COMPONENT = Path("k8s/components/hostipc/kustomization.yaml")
#: worker-train's command while the component is on: the default, as a CHILD of a shell that
#: stays PID 1 (see the component), under umask 0002. One template for the test and the check.
WRAPPER = (
    "umask 0002\n"
    "{cmd} &\n"
    "child=$!\n"
    "trap 'kill -TERM \"$child\" 2>/dev/null' TERM INT\n"
    'wait "$child"; status=$?\n'
    'if kill -0 "$child" 2>/dev/null; then wait "$child"; status=$?; fi\n'
    'exit "$status"\n'
)


def _backend(docs: list[dict[str, Any]]) -> dict[str, Any]:
    spec: dict[str, Any] = _deployments(docs)[DEPLOYMENT]["spec"]["template"]["spec"]
    return spec


def _default_trainer(docs: list[dict[str, Any]]) -> dict[str, Any]:
    return next(c for c in _backend(docs)["containers"] if c["name"] == TRAINER)


class TestTheUidPairing:
    """While the component is on, ONLY worker-train runs as miLLM's uid, with miforge's group."""

    def test_the_default_render_runs_everything_as_10001(self, tmp_path: Path) -> None:
        """The default = the deployed path with the session's components line stripped."""
        pod = _backend(_kustomize_deployed_without_components(tmp_path))
        assert (pod["securityContext"]["runAsUser"], pod["securityContext"]["runAsGroup"]) == (
            10001,
            10001,
        )
        for container in pod["containers"]:
            assert "runAsUser" not in (container.get("securityContext") or {}), container["name"]
        assert SHARED_DIRS_INIT not in [c["name"] for c in pod["initContainers"]]

    def test_the_component_runs_only_the_trainer_as_millms_uid_with_miforges_group(
        self,
    ) -> None:
        pod = _backend(_kustomize(PREVIEW))
        overridden = {
            c["name"]: (c["securityContext"]["runAsUser"], c["securityContext"]["runAsGroup"])
            for c in pod["containers"]
            if "runAsUser" in (c.get("securityContext") or {})
        }
        assert overridden == {TRAINER: (MILLM_UID, MIFORGE_GID)}
        # The pod default stays 10001 for every other container.
        assert pod["securityContext"]["runAsUser"] == 10001

    def test_the_trainer_command_is_the_default_under_the_pid_wrapper(self, tmp_path: Path) -> None:
        """Derived from the default render through ONE template, so the two cannot drift."""
        default = _default_trainer(_kustomize_deployed_without_components(tmp_path))["command"]
        opted = next(c for c in _backend(_kustomize(PREVIEW))["containers"] if c["name"] == TRAINER)
        assert opted["command"] == ["sh", "-c", WRAPPER.format(cmd=shlex.join(default))]

    def test_the_wrapper_runs_the_worker_as_a_child_and_forwards_term(self, tmp_path: Path) -> None:
        """The wrapper itself, executed: the worker is NOT the wrapper's PID (CUDA's IPC
        refuses a handle whose creator PID equals the opener's, and miLLM's server is PID 1
        of its namespace - found live 2026-10-17), TERM reaches it, its status comes back."""
        import signal
        import subprocess
        import sys
        import time

        marker = tmp_path / "child"
        child = f"{shlex.quote(sys.executable)} -c " + shlex.quote(
            "import os,signal,sys,time\n"
            f"open({str(marker)!r},'w').write(str(os.getpid()))\n"
            "signal.signal(signal.SIGTERM, lambda *a: sys.exit(7))\n"
            "time.sleep(60)\n"
        )
        proc = subprocess.Popen(["sh", "-c", WRAPPER.format(cmd=child)])
        deadline = time.monotonic() + 20
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert marker.exists(), "the wrapper never started the worker"
        assert int(marker.read_text()) != proc.pid, "the worker IS the wrapper's process"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=20) == 7, "TERM was not forwarded or the status was lost"

    def test_the_trainer_never_writes_the_shared_hf_cache_as_millms_uid(self) -> None:
        trainer = next(
            c for c in _backend(_kustomize(PREVIEW))["containers"] if c["name"] == TRAINER
        )
        env = {e["name"]: e.get("value") for e in trainer["env"]}
        assert env["HF_HOME"].startswith("/tmp/") and env["HOME"].startswith("/tmp/")
        mounts = {m["mountPath"] for m in trainer["volumeMounts"]}
        assert not any(env["HF_HOME"].startswith(m) for m in mounts)

    def test_the_shared_dirs_init_runs_last_and_touches_directories_only(self) -> None:
        pod = _backend(_kustomize(PREVIEW))
        names = [c["name"] for c in pod["initContainers"]]
        assert names[-1] == SHARED_DIRS_INIT, names
        init = pod["initContainers"][-1]
        script = "\n".join(init["command"])
        assert "-type d" in script and "chmod g+rwxs" in script
        assert "chown" not in script and " -R" not in script
        owner = next(c for c in pod["initContainers"] if c["name"] == "data-volume-owner")
        assert {m["mountPath"] for m in init["volumeMounts"]} == {
            m["mountPath"] for m in owner["volumeMounts"]
        } - {"/data/hf-cache"}

    def test_the_uid_is_the_one_millms_component_declares(self) -> None:
        """Cross-repo: miLLM's hostipc component states its backend's uid; ours must equal it.
        Skips loudly without the sibling, fails under the require switch."""
        root = Path(os.environ.get("MIFORGE_MILLM_ROOT") or REPO_ROOT.parent / "miLLM")
        component = root / MILLM_COMPONENT
        if not component.is_file():
            message = f"miLLM is not checked out at {root}; cannot compare the uid pairing"
            if any(
                os.environ.get(n) == "1"
                for n in ("MIFORGE_REQUIRE_CROSS_REPO_CHECKS", "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS")
            ):
                pytest.fail(message)
            pytest.skip(message)
        doc = yaml.safe_load(component.read_text("utf-8"))
        (patch,) = doc["patches"]
        body = yaml.safe_load(patch["patch"])
        (backend,) = body["spec"]["template"]["spec"]["containers"]
        assert backend["securityContext"]["runAsUser"] == MILLM_UID
        assert f"useradd --uid {MILLM_UID}" in (root / "Dockerfile").read_text("utf-8")

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
