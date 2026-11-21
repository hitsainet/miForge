"""The WSP root's durable storage (FTASKS 4.13): ``/data/wsp`` is a PVC on a ``type: Directory``
Retain volume, declared in ``storage.yaml``, chowned for uid 10001, mounted where the backend reads
and writes bundles and results, and named in the ConfigMap with the settings default.

``test_gitops_is_a_deploy_path.py`` reads ``WSPSettings.wsp_bundle_root`` among its artifact roots
too, so a moved root fails there as well.
"""

from __future__ import annotations

import base64
import re
from typing import Any

from tests.support.wsp_manifests import documents, kustomization, one
from wsp.settings import WSPSettings

ROOT = str(WSPSettings.model_fields["wsp_bundle_root"].default)


def backend_pod() -> dict[str, Any]:
    spec: dict[str, Any] = one("Deployment", "miforge-backend")["spec"]["template"]["spec"]
    return spec


def test_the_root_is_data_wsp() -> None:
    assert ROOT == "/data/wsp"


def test_pv_and_pvc_declared_retain_directory_pinned() -> None:
    docs = documents("storage.yaml")
    pv = one("PersistentVolume", "miforge-wsp-pv", docs)["spec"]
    assert pv["hostPath"] == {"path": "/data/miforge/wsp", "type": "Directory"}
    assert pv["persistentVolumeReclaimPolicy"] == "Retain"
    assert pv["storageClassName"] == "miforge-local-storage"
    terms = pv["nodeAffinity"]["required"]["nodeSelectorTerms"]
    assert terms[0]["matchExpressions"][0]["values"] == ["mcs-lnxhost02"]
    pvc = one("PersistentVolumeClaim", "miforge-wsp", docs)["spec"]
    assert pvc["volumeName"] == "miforge-wsp-pv"


def test_storage_yaml_stays_hand_applied() -> None:
    assert "storage.yaml" not in kustomization()["resources"]


def test_the_backend_mounts_it_from_the_claim() -> None:
    pod = backend_pod()
    volumes = {v["name"]: v for v in pod["volumes"]}
    assert volumes["wsp"]["persistentVolumeClaim"]["claimName"] == "miforge-wsp"
    mounted = {
        c["name"]
        for c in pod["containers"]
        for m in c.get("volumeMounts", [])
        if m["mountPath"] == ROOT and m["name"] == "wsp"
    }
    assert {"api", "worker-default"} <= mounted, mounted


def test_the_root_is_chowned_for_the_application() -> None:
    (init,) = [c for c in backend_pod()["initContainers"] if c["name"] == "data-volume-owner"]
    script = " ".join(init["command"])
    loop = re.search(r"for d in ([^;]*);", script)
    assert loop is not None, "the chown loop is gone"
    assert ROOT in loop.group(1).split()
    assert ROOT in {m["mountPath"] for m in init["volumeMounts"]}


def test_the_configmap_names_the_root() -> None:
    data = one("ConfigMap", "miforge-config")["data"]
    assert data["WSP_BUNDLE_ROOT"] == ROOT
    # Set by the operator at FTASKS 11.2 (2026-11-17): the two in-cluster Services and the one
    # provider.
    # miWorkbench is the bundle source (set 2026-11-17 once 044-A's bundle routes were live).
    assert data["WSP_WORKER_URL"] == "http://miforge-wsp-worker:9000"
    assert data["WSP_GATEWAY_URL"] == "http://miforge-wsp-gateway:8080"
    # 018-P2 (FTASKS 12.1): miwsp_engine enabled beside weightspace; the default is unchanged.
    assert data["WSP_PROVIDERS_AVAILABLE"] == "weightspace,miwsp_engine"
    assert "WSP_DEFAULT_PROVIDER" not in data  # the switch is 018-P3's, with its record
    assert (
        data["MIWORKBENCH_API_URL"]
        == "http://miworkbench-backend.miworkbench.svc.cluster.local:8000"
    )
    # The gateway's public key is the raw 32-byte X25519 key, base64 (sealing.seal refuses
    # anything else).
    assert len(base64.b64decode(data["WSP_GATEWAY_PUBLIC_KEY"], validate=True)) == 32
