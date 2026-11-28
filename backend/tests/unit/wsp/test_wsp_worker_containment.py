"""FR-016.10: the worker StatefulSet's containment, one test per D12 line (FTDD §8.1; FTASKS 4.1).

A Deployment until 019 P3, which made it a StatefulSet with a headless Service (D-16): every pin
reads the StatefulSet's pod template. Read from the rendered kustomization, so a workload that is
not listed fails here rather than passing as a file nobody applies. Each security line of
``gitops/pilot/wsp-worker.yaml`` turns exactly its own test red when deleted (controls
C10a-C10l), and a planted ``envFrom`` turns ``test_env_exact`` and ``test_no_secret_reference``
red (C10m).
"""

from __future__ import annotations

from typing import Any

from tests.support.wsp_manifests import documents, kustomization, one, rendered

NAME = "miforge-wsp-worker"
#: 019 P3 (F19-D18): the headless Service each worker is addressed under.
SERVICE = "miforge-wsp-workers"
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
    """The worker workload: a StatefulSet since 019 P3 (the name is kept for the pins below)."""
    return one("StatefulSet", NAME)


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


def _pool_max() -> int:
    (config,) = [d for d in documents("config.yaml") if d["kind"] == "ConfigMap"]
    return int(config["data"]["WSP_WORKER_POOL_MAX"])


def test_replicas_are_the_provisioned_maximum() -> None:
    """019 D-16 (F19-D5, D22): replicas equal WSP_WORKER_POOL_MAX, shipped at 1 and raised to 2 by
    P3b (2026-11-25) in one commit; one pod per worker under the headless Service, started in
    parallel."""
    spec = deployment()["spec"]
    assert spec["replicas"] == _pool_max()
    assert spec["serviceName"] == SERVICE
    assert spec["podManagementPolicy"] == "Parallel"
    assert spec["updateStrategy"] == {"type": "RollingUpdate"}
    assert "strategy" not in spec
    assert not [
        d for d in rendered() if d["kind"] == "Deployment" and d["metadata"]["name"] == NAME
    ]


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


def test_the_supervisor_runs() -> None:
    # The Service it is reached through is test_wsp_worker_service.py's (019 D-17).
    assert container()["args"] == ["supervisor"]


def test_rendered_set_is_not_empty() -> None:
    assert len(rendered()) > 5
