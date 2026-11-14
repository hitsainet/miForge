"""G7 (FR-012.3): the MCP pod has no database, Redis, broker, volume or GPU - read from the
manifest ArgoCD applies, and that manifest is LISTED in the kustomization.

The absence is the design (MR-2): every tool reaches miForge over HTTP, so every rule in the
REST layer applies. A tool that imported a session factory would raise on its first call in a
pod with no database - which every test running in a process that HAS one would miss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

GITOPS = Path(__file__).resolve().parents[3] / "gitops" / "pilot"
MANIFEST = GITOPS / "mcp.yaml"
FORBIDDEN_ENV = {"DATABASE_URL", "REDIS_URL", "CELERY_BROKER_URL", "POSTGRES_PASSWORD"}


def documents() -> list[dict[str, Any]]:
    return [doc for doc in yaml.safe_load_all(MANIFEST.read_text()) if doc]


def deployment() -> dict[str, Any]:
    (dep,) = [
        d
        for d in documents()
        if d["kind"] == "Deployment" and d["metadata"]["name"] == "miforge-mcp"
    ]
    return dep


def container() -> dict[str, Any]:
    (only,) = deployment()["spec"]["template"]["spec"]["containers"]
    return dict(only)


def env() -> dict[str, dict[str, Any]]:
    return {e["name"]: e for e in container()["env"]}


def test_the_manifest_is_listed_in_the_kustomization() -> None:
    kustomization = yaml.safe_load((GITOPS / "kustomization.yaml").read_text())
    assert (
        "mcp.yaml" in kustomization["resources"]
    ), "gitops/pilot/mcp.yaml is not in kustomization.yaml: ArgoCD would never apply it"


def test_the_mcp_container_has_no_database_redis_or_broker_env() -> None:
    spec = deployment()["spec"]["template"]["spec"]
    for each in spec["containers"] + spec.get("initContainers", []):
        names = {e["name"] for e in each.get("env", [])}
        assert names, "the container declares no env at all - the scan read nothing"
        assert not names & FORBIDDEN_ENV, sorted(names & FORBIDDEN_ENV)
        assert not each.get("envFrom"), "envFrom could import the backend's whole config"


def test_the_mcp_pod_mounts_no_volume() -> None:
    spec = deployment()["spec"]["template"]["spec"]
    assert not spec.get("volumes"), spec.get("volumes")
    for each in spec["containers"]:
        assert not each.get("volumeMounts"), each.get("volumeMounts")


def test_the_mcp_pod_sees_no_gpu() -> None:
    assert env()["NVIDIA_VISIBLE_DEVICES"]["value"] == "void"
    limits = (container().get("resources") or {}).get("limits") or {}
    assert "nvidia.com/gpu" not in limits


def test_it_runs_the_backend_image_as_the_mcp_server_over_http_with_a_bearer_token() -> None:
    each = container()
    assert each["image"].startswith("hitsai/miforge-backend:")
    assert each["command"] == ["python", "-m", "mcp_server"]
    assert env()["MIFORGE_API_URL"]["value"] == "http://miforge-backend:8000"
    assert env()["MCP_AUTH_TOKEN"]["valueFrom"]["secretKeyRef"] == {
        "name": "miforge-secrets",
        "key": "MCP_BEARER_TOKEN",
    }
    assert "value" not in env()["MCP_AUTH_TOKEN"], "a token value is in the manifest"
    assert env()["MCP_AGENT_IDENTITY"]["value"] == "agent:forge-mcp"
    assert each["ports"] == [{"containerPort": 8765}]
    security = deployment()["spec"]["template"]["spec"]["securityContext"]
    assert security["runAsNonRoot"] is True and security["runAsUser"] == 10001


def test_the_service_and_ingress_publish_port_8765_on_the_three_hosts() -> None:
    (service,) = [d for d in documents() if d["kind"] == "Service"]
    assert service["spec"]["ports"] == [{"port": 8765, "targetPort": 8765}]
    (ingress,) = [d for d in documents() if d["kind"] == "Ingress"]
    rules = {
        rule["host"]: [
            (p["path"], p["backend"]["service"]["name"], p["backend"]["service"]["port"]["number"])
            for p in rule["http"]["paths"]
        ]
        for rule in ingress["spec"]["rules"]
    }
    assert rules == {
        "k8s-miforge.hitsai.local": [("/mcp", "miforge-mcp", 8765)],
        "mcp-forge.hitsai.local": [("/", "miforge-mcp", 8765)],
        "k8s-mcp-forge.hitsai.local": [("/", "miforge-mcp", 8765)],
        "mcp-forge.hitsai.net": [("/", "miforge-mcp", 8765)],
    }
    assert ingress["spec"]["ingressClassName"] == "public"
