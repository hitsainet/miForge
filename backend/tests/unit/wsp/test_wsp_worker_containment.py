"""FR-016.10: the worker Deployment's containment, one test per D12 line (FTDD §8.1; FTASKS 4.1).

Read from the rendered kustomization, so a Deployment that is not listed fails here rather than
passing as a file nobody applies. Each security line of ``gitops/pilot/wsp-worker.yaml`` turns
exactly
its own test red when deleted (controls C10a-C10l), and a planted ``envFrom`` turns
``test_env_exact``
and ``test_no_secret_reference`` red (C10m).
"""

from __future__ import annotations

from typing import Any

from tests.support.wsp_manifests import kustomization, one, rendered

NAME = "miforge-wsp-worker"
LABELS = {"app.kubernetes.io/name": NAME, "app.kubernetes.io/component": "wsp-worker"}
ENV = {
    "NVIDIA_VISIBLE_DEVICES": "void",
    "WSP_WORKER_PORT": "9000",
    "WSP_ENVIRONMENT": "prod",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
WRITABLE = {"/work": "work", "/out": "out", "/tmp": "tmp", "/dev/shm": "shm"}


def deployment() -> dict[str, Any]:
    return one("Deployment", NAME)


def pod() -> dict[str, Any]:
    spec: dict[str, Any] = deployment()["spec"]["template"]["spec"]
    return spec


def container() -> dict[str, Any]:
    (only,) = pod()["containers"]
    return dict(only)


def test_listed_in_the_kustomization_with_its_image() -> None:
    assert "wsp-worker.yaml" in kustomization()["resources"]
    assert "hitsai/miforge-wsp-worker" in {i["name"] for i in kustomization()["images"]}
    assert container()["image"].split(":")[0] == "hitsai/miforge-wsp-worker"


def test_one_replica_recreated() -> None:
    assert deployment()["spec"]["replicas"] == 1
    assert deployment()["spec"]["strategy"] == {"type": "Recreate"}


def test_non_root() -> None:
    for ctx in (pod()["securityContext"], container()["securityContext"]):
        assert ctx["runAsNonRoot"] is True
        assert ctx["runAsUser"] == 10002 and ctx["runAsGroup"] == 10002


def test_read_only_root() -> None:
    assert container()["securityContext"]["readOnlyRootFilesystem"] is True


def test_only_these_writable() -> None:
    mounts = {m["mountPath"]: m["name"] for m in container()["volumeMounts"]}
    assert mounts == WRITABLE
    volumes = {v["name"]: v for v in pod()["volumes"]}
    assert set(volumes) == set(WRITABLE.values())
    for name, volume in volumes.items():
        assert set(volume) == {"name", "emptyDir"}, f"{name} is not an emptyDir"
        assert volume["emptyDir"].get("sizeLimit"), f"{name} has no sizeLimit"


def test_dev_shm_is_a_memory_volume() -> None:
    volumes = {v["name"]: v for v in pod()["volumes"]}
    assert volumes["shm"]["emptyDir"]["medium"] == "Memory"
    for name in ("out", "tmp"):
        assert volumes[name]["emptyDir"]["medium"] == "Memory"


def test_seccomp() -> None:
    assert pod()["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}
    assert container()["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}


def test_caps() -> None:
    assert container()["securityContext"]["capabilities"] == {"drop": ["ALL"]}


def test_no_privilege_escalation() -> None:
    assert container()["securityContext"]["allowPrivilegeEscalation"] is False


def test_no_sa_token() -> None:
    assert pod()["automountServiceAccountToken"] is False


def test_no_service_account_name() -> None:
    assert "serviceAccountName" not in pod() and "serviceAccount" not in pod()


def test_limits() -> None:
    limits = container()["resources"]["limits"]
    assert {"cpu", "memory", "ephemeral-storage"} <= set(limits)


def test_no_host_access() -> None:
    for key in ("hostNetwork", "hostIPC", "hostPID", "hostUsers"):
        assert not pod().get(key), key
    assert not [v for v in pod()["volumes"] if "hostPath" in v]


def test_own_pid_namespace() -> None:
    assert "shareProcessNamespace" not in pod()


def test_no_dns() -> None:
    assert pod()["dnsPolicy"] == "None"
    assert pod()["dnsConfig"] == {"nameservers": ["127.0.0.1"]}


def test_env_exact() -> None:
    env = container()["env"]
    assert all(set(e) == {"name", "value"} for e in env), "an env entry reads from somewhere"
    assert {e["name"]: e["value"] for e in env} == ENV
    assert "envFrom" not in container()


def test_no_secret_reference() -> None:
    text = repr(deployment())
    assert "secretKeyRef" not in text and "secretRef" not in text
    assert not [v for v in pod()["volumes"] if "secret" in v or "projected" in v]
    assert "envFrom" not in container()


def test_no_service_links() -> None:
    assert pod()["enableServiceLinks"] is False


def test_termination_path() -> None:
    assert container()["terminationMessagePath"] == "/tmp/termination-log"


def test_pod_labels() -> None:
    labels = deployment()["spec"]["template"]["metadata"]["labels"]
    assert {k: labels.get(k) for k in LABELS} == LABELS
    assert deployment()["metadata"]["namespace"] == "miforge"


def test_liveness_probe() -> None:
    probe = container()["livenessProbe"]
    assert probe["httpGet"] == {"path": "/ready", "port": 9000}
    assert probe["periodSeconds"] == 10 and probe["failureThreshold"] == 3


def test_the_supervisor_runs_and_the_service_reaches_it() -> None:
    assert container()["args"] == ["supervisor"]
    service = one("Service", NAME)
    assert service["spec"]["selector"] == {"app.kubernetes.io/name": NAME}
    assert [(p["port"], p["targetPort"]) for p in service["spec"]["ports"]] == [(9000, 9000)]


def test_rendered_set_is_not_empty() -> None:
    assert len(rendered()) > 5
