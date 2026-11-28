"""The WSP driver's Kubernetes grant (operator 2026-11-26: a fresh worker pod per session), pinned
from the manifests ArgoCD applies (``tests.support.wsp_manifests``):

- the Role grants exactly ``get`` and ``delete`` on ``pods``, in the core API group, limited by
  ``resourceNames`` to exactly ``miforge-wsp-worker-0 .. <WSP_WORKER_POOL_MAX - 1>``: the names
  ``WSP_WORKER_POD_NAME`` makes for every provisioned worker, and the StatefulSet's pods;
- the RoleBinding binds that Role, and nothing else, to the one ServiceAccount, in ``miforge``;
- the ServiceAccount automounts nothing; the backend pod names it and still automounts nothing;
- the token is a projected volume mounted in the ``worker-wsp`` container (the driver) alone;
- the file is listed in the kustomization, so ArgoCD applies it.
"""

from __future__ import annotations

from typing import Any

from tests.support.wsp_manifests import documents, kustomization, one, rendered

FILE = "wsp-driver-rbac.yaml"
ACCOUNT = "miforge-wsp-driver"
ROLE = "miforge-wsp-worker-recycler"
TOKEN_VOLUME = "wsp-driver-token"
TOKEN_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"


def _config() -> dict[str, str]:
    return dict(one("ConfigMap", "miforge-config")["data"])


def _worker_names() -> list[str]:
    config = _config()
    template = config["WSP_WORKER_POD_NAME"]
    return [template.format(index=i) for i in range(int(config["WSP_WORKER_POOL_MAX"]))]


def _backend_pod() -> dict[str, Any]:
    spec: dict[str, Any] = one("Deployment", "miforge-backend")["spec"]["template"]["spec"]
    return spec


def test_the_file_is_applied_and_holds_exactly_three_objects() -> None:
    assert FILE in kustomization()["resources"]
    assert sorted((d["kind"], d["metadata"]["name"]) for d in documents(FILE)) == [
        ("Role", ROLE),
        ("RoleBinding", ROLE),
        ("ServiceAccount", ACCOUNT),
    ]
    assert all(d["metadata"]["namespace"] == "miforge" for d in documents(FILE))


def test_the_role_grants_get_and_delete_on_exactly_the_worker_pods() -> None:
    role = one("Role", ROLE)
    assert set(role) == {"apiVersion", "kind", "metadata", "rules"}
    assert role["rules"] == [
        {
            "apiGroups": [""],
            "resources": ["pods"],
            "resourceNames": _worker_names(),
            "verbs": ["get", "delete"],
        }
    ]


def test_the_names_are_every_provisioned_worker_and_the_statefulsets_pods() -> None:
    config = _config()
    worker = one("StatefulSet", "miforge-wsp-worker")
    replicas = int(worker["spec"]["replicas"])
    assert int(config["WSP_WORKER_POOL_MAX"]) == replicas
    assert _worker_names() == [f"{worker['metadata']['name']}-{i}" for i in range(replicas)]
    assert config["WSP_WORKER_POD_NAMESPACE"] == worker["metadata"]["namespace"] == "miforge"


def test_the_binding_binds_only_that_role_to_only_that_account() -> None:
    binding = one("RoleBinding", ROLE)
    assert binding["roleRef"] == {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "Role",
        "name": ROLE,
    }
    assert binding["subjects"] == [
        {"kind": "ServiceAccount", "name": ACCOUNT, "namespace": "miforge"}
    ]


def test_nothing_is_cluster_scoped() -> None:
    assert not [d for d in rendered() if d["kind"] in ("ClusterRole", "ClusterRoleBinding")]


def test_the_account_and_the_backend_pod_automount_no_token() -> None:
    assert one("ServiceAccount", ACCOUNT).get("automountServiceAccountToken") is False
    spec = _backend_pod()
    assert spec["serviceAccountName"] == ACCOUNT
    assert spec["automountServiceAccountToken"] is False


def test_the_token_is_projected_and_mounted_in_the_driver_alone() -> None:
    spec = _backend_pod()
    (volume,) = [v for v in spec["volumes"] if v["name"] == TOKEN_VOLUME]
    sources = volume["projected"]["sources"]
    assert sources[0] == {"serviceAccountToken": {"path": "token", "expirationSeconds": 3600}}
    assert sources[1]["configMap"]["name"] == "kube-root-ca.crt"
    mounted = {
        c["name"]: [m for m in c.get("volumeMounts", []) if m["name"] == TOKEN_VOLUME]
        for c in spec["containers"] + spec.get("initContainers", [])
    }
    assert {name for name, mounts in mounted.items() if mounts} == {"worker-wsp"}
    assert mounted["worker-wsp"] == [
        {"name": TOKEN_VOLUME, "mountPath": TOKEN_DIR, "readOnly": True}
    ]
    for name, mounts in mounted.items():
        assert not [m for m in mounts if m["mountPath"] == TOKEN_DIR] or name == "worker-wsp"


def test_no_other_pod_names_an_account_or_mounts_a_token() -> None:
    from tests.support.wsp_manifests import pod_specs

    for name, spec in pod_specs(rendered()):
        if name == "miforge-backend":
            continue
        assert "serviceAccountName" not in spec, name
        projected = [
            v
            for v in spec.get("volumes", [])
            if any(
                "serviceAccountToken" in s for s in (v.get("projected") or {}).get("sources", [])
            )
        ]
        assert not projected, name
