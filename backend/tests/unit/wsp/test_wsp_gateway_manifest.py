"""The gateway's Deployment (feature 016; FTASKS 6.5, 6.7; FR-016.15, FR-016.16; D15, R1B-M9).

Read the way ArgoCD applies it (``tests/support/wsp_manifests.rendered``): every document of every
file the kustomization lists. The pattern of ``backend/tests/unit/test_mcp_manifest_isolation.py``.
"""

from __future__ import annotations

import json
from typing import Any

from tests.support.wsp_manifests import kustomization, one, rendered
from wsp.settings import WSPSettings

FORBIDDEN_ENV = {"DATABASE_URL", "REDIS_URL", "CELERY_BROKER_URL", "POSTGRES_PASSWORD"}
PRIVATE_SECRET = "miforge-wsp-gateway"
PRIVATE_KEY = "WSP_GATEWAY_PRIVATE_KEY"


def deployment() -> dict[str, Any]:
    return one("Deployment", "miforge-wsp-gateway")


def container() -> dict[str, Any]:
    (only,) = deployment()["spec"]["template"]["spec"]["containers"]
    return dict(only)


def env() -> dict[str, dict[str, Any]]:
    return {e["name"]: e for e in container()["env"]}


def test_the_manifest_is_listed_in_the_kustomization() -> None:
    assert "wsp-gateway.yaml" in kustomization()["resources"]


def test_it_runs_the_backend_image_as_python_m_wsp_gateway() -> None:
    c = container()
    assert c["image"].startswith("hitsai/miforge-backend:")
    assert c["command"] == ["python", "-m", "wsp_gateway"]
    assert c["ports"] == [{"name": "gateway", "containerPort": 8080}]
    labels = deployment()["spec"]["template"]["metadata"]["labels"]
    assert labels["app.kubernetes.io/name"] == "miforge-wsp-gateway"  # the NetworkPolicy's selector


def test_no_database_redis_volume_or_envfrom() -> None:
    spec = deployment()["spec"]["template"]["spec"]
    names = set(env())
    assert names, "the container declares no env at all: the scan read nothing"
    assert not names & FORBIDDEN_ENV, sorted(names & FORBIDDEN_ENV)
    assert not container().get("envFrom")
    assert not spec.get("volumes") and not container().get("volumeMounts")
    assert spec["automountServiceAccountToken"] is False


def test_no_gpu_no_capability_a_read_only_root_and_limits() -> None:
    c = container()
    assert env()["NVIDIA_VISIBLE_DEVICES"]["value"] == "void"
    ctx = c["securityContext"]
    assert ctx["capabilities"]["drop"] == ["ALL"] and ctx["allowPrivilegeEscalation"] is False
    assert ctx["readOnlyRootFilesystem"] is True and ctx["runAsNonRoot"] is True
    assert {"cpu", "memory"} <= set(c["resources"]["limits"])


def _secret_names(obj: Any) -> set[str]:
    """Every Secret an object references: env, envFrom, volumes and projected sources."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("secretKeyRef", "secretRef", "secret") and isinstance(value, dict):
                found.add(str(value.get("name") or value.get("secretName")))
            found |= _secret_names(value)
    elif isinstance(obj, list):
        for item in obj:
            found |= _secret_names(item)
    return found


def _holds_private_key(doc: dict[str, Any]) -> bool:
    return PRIVATE_SECRET in _secret_names(doc) or PRIVATE_KEY in json.dumps(doc)


def test_private_key_only_in_gateway() -> None:
    """D15, R1B-M9: the Secret and its key are referenced by the gateway Deployment ALONE."""
    holders = [f"{d['kind']}/{d['metadata']['name']}" for d in rendered() if _holds_private_key(d)]
    assert holders == ["Deployment/miforge-wsp-gateway"], holders
    ref = env()[PRIVATE_KEY]["valueFrom"]["secretKeyRef"]
    assert (ref["name"], ref["key"]) == (PRIVATE_SECRET, PRIVATE_KEY)
    assert not any("private" in name for name in WSPSettings.model_fields)


def test_the_private_key_check_sees_a_planted_reference() -> None:
    by_volume = {"spec": {"volumes": [{"name": "k", "secret": {"secretName": PRIVATE_SECRET}}]}}
    by_env_from = {"envFrom": [{"secretRef": {"name": PRIVATE_SECRET}}]}
    assert _holds_private_key(by_volume) and _holds_private_key(by_env_from)
    assert not _holds_private_key({"metadata": {"name": PRIVATE_SECRET}})  # a label is not a ref


def test_service_token_only_in_wsp_internal_and_gateway() -> None:
    ref = env()["WSP_GATEWAY_SERVICE_TOKEN"]["valueFrom"]["secretKeyRef"]
    assert (ref["name"], ref["key"]) == ("miforge-secrets", "WSP_GATEWAY_SERVICE_TOKEN")
    assert env()["WSP_INTERNAL_URL"]["value"] == "http://miforge-wsp-internal:8001"


def test_the_service_serves_8080_inside_the_cluster() -> None:
    svc = one("Service", "miforge-wsp-gateway")
    assert svc["spec"].get("type", "ClusterIP") == "ClusterIP"
    assert [p["port"] for p in svc["spec"]["ports"]] == [8080]
    assert not any(
        "miforge-wsp-gateway" in json.dumps(d) for d in rendered() if d["kind"] == "Ingress"
    )
