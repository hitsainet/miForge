"""FR-016.14 (D14): no Kubernetes permission in miForge (FTDD §8.3; FTASKS 4.10), with the ONE
exception the operator chose on 2026-11-26 (a fresh WSP worker pod per session, option B):

- no ``Role``, ``ClusterRole``, ``RoleBinding``, ``ClusterRoleBinding`` or ``ServiceAccount`` object
  in any manifest under ``gitops/`` (the rendered set and everything applied by hand or by
  overlays) except exactly the three of ``gitops/pilot/wsp-driver-rbac.yaml`` (their content is
  ``test_wsp_driver_rbac.py``'s); never a cluster-scoped one;
- ``automountServiceAccountToken: false`` on EVERY pod miForge declares, present or added later
  (ENH-011 FG-1, E11-10). The pods are DERIVED from the manifests, never listed by hand: the
  hand-kept list of three this file used to carry could not see that frontend, postgres and
  redis mounted the namespace's default token;
- no ``serviceAccountName`` on any pod but the backend's ``miforge-wsp-driver``, and the only
  mounted service-account token is that account's projected one, in the ``worker-wsp``
  container alone;
- no module under ``backend/src`` imports ``kubernetes``.

Where the pods come from: every top-level document of every YAML file under ``gitops/`` (so the
kustomization's resources, ``storage.yaml`` and anything else applied by hand), the kustomization's
rendered set, and ``kubectl kustomize`` of ``gitops/pilot`` and of every overlay, which is how the
components' patches are seen applied. Each source is cross-checked against a structural count
(every mapping that carries ``containers`` is a pod spec), so a walk that reads nothing, a kind
``pod_specs`` does not know, or a shortened list cannot pass.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.support.gitops_docs import walk
from tests.support.wsp_manifests import GITOPS, pod_specs, rendered

REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "backend" / "src"
GITOPS_ROOT = REPO / "gitops"
FORBIDDEN_KINDS = {"Role", "ClusterRole", "RoleBinding", "ClusterRoleBinding", "ServiceAccount"}
#: Operator 2026-11-26: the only RBAC objects miForge carries, all in one file.
ALLOWED_RBAC = {
    ("gitops/pilot/wsp-driver-rbac.yaml", "ServiceAccount", "miforge-wsp-driver"),
    ("gitops/pilot/wsp-driver-rbac.yaml", "Role", "miforge-wsp-worker-recycler"),
    ("gitops/pilot/wsp-driver-rbac.yaml", "RoleBinding", "miforge-wsp-worker-recycler"),
}
#: The one pod that names a service account, and the account.
SERVICE_ACCOUNT_OF = {"miforge-backend": "miforge-wsp-driver"}
#: The one (pod, container) that mounts a service-account token.
TOKEN_MOUNTED_IN = {("miforge-backend", "worker-wsp")}
#: Where the kubelet mounts a service-account token; nothing else may mount under it.
TOKEN_DIR = "/var/run/secrets/kubernetes.io/"
#: A FLOOR, not the list the rule is checked against: the walk must find at least these, so a
#: walk that reads nothing cannot pass. The CPU editor is temporary (019 P4); drop it here when
#: its manifest goes.
KNOWN_AT_LEAST = {
    "miforge-backend",
    "miforge-mcp",
    "miforge-frontend",
    "miforge-postgres",
    "miforge-redis",
    "miforge-wsp-worker",
    "miforge-wsp-gateway",
    "miforge-wsp-cpu-editor",
}

Pod = tuple[str, str, dict[str, Any]]  # (source, pod name, pod spec)


# ── where the pods come from ───────────────────────────────────────────────────────────────────


def every_gitops_document() -> list[tuple[str, dict[str, Any]]]:
    found = [
        (str(doc.path.relative_to(REPO)), doc.data)
        for doc in walk(GITOPS_ROOT)
        if doc.embedded_in is None and isinstance(doc.data, dict)
    ]
    assert len(found) > 10, "the walk read almost nothing"
    return found


def _structural_pod_count(data: Any) -> int:
    """Every mapping that carries a ``containers`` list is a pod spec, whatever its kind."""
    if isinstance(data, dict):
        here = 1 if isinstance(data.get("containers"), list) else 0
        return here + sum(_structural_pod_count(v) for v in data.values())
    if isinstance(data, list):
        return sum(_structural_pod_count(v) for v in data)
    return 0


def _pods_of(source: str, docs: list[dict[str, Any]]) -> list[Pod]:
    return [(source, name, spec) for name, spec in pod_specs([d for d in docs if "kind" in d])]


def every_pod() -> list[Pod]:
    """Every pod spec in every file under ``gitops/``, and in the kustomization's rendered set."""
    pods: list[Pod] = []
    for path, doc in every_gitops_document():
        pods.extend(_pods_of(path, [doc]))
    pods.extend(_pods_of("rendered gitops/pilot", rendered()))
    return pods


def _require_kubectl() -> None:
    if shutil.which("kubectl") is None:
        message = "kubectl is not installed; the kustomize render check cannot run here"
        if os.environ.get("CI"):
            pytest.fail(message + " - and CI must run it (ubuntu-latest ships it)")
        pytest.skip(message)


def _kustomize(path: Path) -> list[dict[str, Any]]:
    _require_kubectl()
    out = subprocess.run(
        ["kubectl", "kustomize", str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    return [doc for doc in yaml.safe_load_all(out) if doc]


def _render_roots() -> list[Path]:
    """``gitops/pilot`` and every overlay: each is something an operator can apply."""
    overlays = sorted(p.parent for p in (GITOPS_ROOT / "overlays").glob("*/kustomization.yaml"))
    assert overlays, "no overlay found under gitops/overlays"
    return [GITOPS, *overlays]


# ── the rules, applied to any list of pods ─────────────────────────────────────────────────────


def _token_volumes(spec: dict[str, Any]) -> set[str]:
    return {
        v["name"]
        for v in spec.get("volumes") or []
        if any("serviceAccountToken" in s for s in (v.get("projected") or {}).get("sources") or [])
    }


def _containers(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *(spec.get("initContainers") or []),
        *(spec.get("containers") or []),
        *(spec.get("ephemeralContainers") or []),
    ]


def automount_left_on(pods: list[Pod]) -> list[str]:
    return sorted(
        f"{source}: {name}"
        for source, name, spec in pods
        if spec.get("automountServiceAccountToken") is not False
    )


def accounts_named(pods: list[Pod]) -> set[tuple[str, str]]:
    named = set()
    for _source, name, spec in pods:
        for key in ("serviceAccountName", "serviceAccount"):
            if key in spec:
                named.add((name, f"{key}={spec[key]}"))
    return named


def token_mounts(pods: list[Pod]) -> set[tuple[str, str]]:
    mounted = set()
    for _source, name, spec in pods:
        tokens = _token_volumes(spec)
        for container in _containers(spec):
            for mount in container.get("volumeMounts") or []:
                if mount["name"] in tokens or str(mount.get("mountPath", "")).startswith(TOKEN_DIR):
                    mounted.add((name, container["name"]))
    return mounted


def pods_declaring_a_token(pods: list[Pod]) -> set[str]:
    return {name for _source, name, spec in pods if _token_volumes(spec)}


def _check_every_rule(pods: list[Pod]) -> None:
    assert not automount_left_on(pods), automount_left_on(pods)
    expected = {(pod, f"serviceAccountName={sa}") for pod, sa in SERVICE_ACCOUNT_OF.items()}
    assert accounts_named(pods) == expected, accounts_named(pods) ^ expected
    assert token_mounts(pods) == TOKEN_MOUNTED_IN, token_mounts(pods) ^ TOKEN_MOUNTED_IN
    assert pods_declaring_a_token(pods) == set(SERVICE_ACCOUNT_OF)


# ── the tests ──────────────────────────────────────────────────────────────────────────────────


def test_no_rbac_or_service_account_object_but_the_drivers_three() -> None:
    found = {
        (p, d.get("kind"), (d.get("metadata") or {}).get("name"))
        for p, d in every_gitops_document()
        if d.get("kind") in FORBIDDEN_KINDS
    }
    assert found == ALLOWED_RBAC, found ^ ALLOWED_RBAC


def test_the_walk_finds_every_pod_and_at_least_the_known_ones() -> None:
    pods = every_pod()
    from_files = [p for p in pods if p[0] != "rendered gitops/pilot"]
    from_render = [p for p in pods if p[0] == "rendered gitops/pilot"]
    structural = sum(_structural_pod_count(doc) for _path, doc in every_gitops_document())
    assert len(from_files) == structural, (len(from_files), structural)
    assert len(from_render) == sum(_structural_pod_count(doc) for doc in rendered())
    for found in (from_files, from_render):
        missing = KNOWN_AT_LEAST - {name for _source, name, _spec in found}
        assert not missing, f"the walk did not find {sorted(missing)}"


def test_no_pod_automounts_a_service_account_token() -> None:
    assert not automount_left_on(every_pod()), automount_left_on(every_pod())


def test_only_the_backend_names_a_service_account_and_only_the_drivers() -> None:
    expected = {(pod, f"serviceAccountName={sa}") for pod, sa in SERVICE_ACCOUNT_OF.items()}
    assert accounts_named(every_pod()) == expected, accounts_named(every_pod()) ^ expected


def test_the_only_mounted_token_is_in_the_driver_container() -> None:
    pods = every_pod()
    assert token_mounts(pods) == TOKEN_MOUNTED_IN, token_mounts(pods) ^ TOKEN_MOUNTED_IN
    assert pods_declaring_a_token(pods) == set(SERVICE_ACCOUNT_OF)


def test_no_kustomize_patch_turns_a_token_or_an_account_on() -> None:
    """A component's patch is applied on top of a pod that already says ``false``: it may not
    set the field otherwise, name an account, or add a token, by strategic merge or by JSON 6902.
    (The render test below sees the patches applied; this one also runs without kubectl.)"""
    bad: list[str] = []

    def scan(data: Any, where: str) -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "automountServiceAccountToken" and value is not False:
                    bad.append(f"{where}: {key}={value}")
                elif key in ("serviceAccountName", "serviceAccount", "serviceAccountToken"):
                    bad.append(f"{where}: {key}")
                elif key == "path" and isinstance(value, str) and "serviceAccount" in value:
                    bad.append(f"{where}: op on {value}")
                scan(value, where)
        elif isinstance(data, list):
            for item in data:
                scan(item, where)

    patches = [doc for doc in walk(GITOPS_ROOT) if doc.embedded_in is not None]
    assert patches, "no kustomize patch found: the hostIPC component carries two"
    for doc in patches:
        scan(doc.data, doc.embedded_in or "")
    assert not bad, bad


@pytest.mark.parametrize("root", _render_roots(), ids=lambda p: str(p.relative_to(REPO)))
def test_every_kustomize_render_keeps_every_rule(root: Path) -> None:
    docs = _kustomize(root)
    pods = _pods_of(f"kubectl kustomize {root.relative_to(REPO)}", docs)
    assert len(pods) == sum(_structural_pod_count(doc) for doc in docs)
    missing = KNOWN_AT_LEAST - {name for _source, name, _spec in pods}
    assert not missing, f"the render did not find {sorted(missing)}"
    _check_every_rule(pods)


def test_every_component_is_rendered_by_an_overlay() -> None:
    """A component no overlay includes is never seen applied by the render test."""
    used: set[Path] = set()
    for root in _render_roots():
        loaded = yaml.safe_load((root / "kustomization.yaml").read_text()) or {}
        used.update((root / c).resolve() for c in loaded.get("components") or [])
    components = {
        p.parent.resolve() for p in (GITOPS_ROOT / "components").glob("*/kustomization.yaml")
    }
    assert components, "no component found under gitops/components"
    assert components <= used, sorted(str(c) for c in components - used)


def test_no_module_imports_kubernetes() -> None:
    files = sorted(SRC.rglob("*.py"))
    assert len(files) > 100
    bad = []
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            bad.extend(
                f"{path.relative_to(SRC)}:{node.lineno}"
                for n in names
                if n.split(".")[0] == "kubernetes"
            )
    assert not bad, bad


def test_the_walk_covers_the_rendered_set() -> None:
    assert GITOPS.is_dir() and len(rendered()) > 5
