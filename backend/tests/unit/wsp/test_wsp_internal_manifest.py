"""The driver's container and the gateway service token's reach (FTASKS 5.6; P60, R2A-4).

Read from the rendered kustomization (what ArgoCD deploys), never from a source scrape.
"""

from __future__ import annotations

from typing import Any

from tests.support.wsp_manifests import pod_specs, rendered

TOKEN_NAME = "WSP_GATEWAY_SERVICE_TOKEN"
#: The only containers that may carry the gateway service token (FTDD §5.4).
TOKEN_HOLDERS = {"wsp-internal", "wsp-gateway"}


def _containers() -> list[dict[str, Any]]:
    out = []
    for _owner, spec in pod_specs(rendered()):
        out += list(spec.get("initContainers") or []) + list(spec.get("containers") or [])
    return out


def _mentions_token(container: dict[str, Any]) -> bool:
    for env in container.get("env") or []:
        ref = ((env.get("valueFrom") or {}).get("secretKeyRef") or {}).get("key")
        if env.get("name") == TOKEN_NAME or ref == TOKEN_NAME:
            return True
    return False


def test_the_scan_sees_every_container() -> None:
    names = {c["name"] for c in _containers()}
    assert {
        "api",
        "worker-default",
        "worker-maintenance",
        "worker-wsp",
        "worker-wsp-push",
        "beat",
    } <= names, names


def test_the_service_token_reaches_only_the_internal_app_and_the_gateway() -> None:
    holders = {c["name"] for c in _containers() if _mentions_token(c)}
    assert holders <= TOKEN_HOLDERS, holders


def test_the_token_check_can_see_a_planted_reference() -> None:
    planted = {
        "name": "api",
        "env": [{"name": "X", "valueFrom": {"secretKeyRef": {"key": TOKEN_NAME}}}],
    }
    assert _mentions_token(planted)


def test_the_driver_runs_solo_on_its_own_queue_with_the_store() -> None:
    worker = next(c for c in _containers() if c["name"] == "worker-wsp")
    command = worker["command"]
    assert command[command.index("-Q") + 1] == "wsp"
    assert "--pool=solo" in command and command[command.index("-c") + 1] == "1"
    # Operator 2026-11-26: plus the projected token that replaces a worker's pod after its
    # session (test_wsp_driver_rbac.py pins it to this container alone).
    assert {m["mountPath"] for m in worker["volumeMounts"]} == {
        "/data/wsp",
        "/var/run/secrets/kubernetes.io/serviceaccount",
    }
    assert not _mentions_token(worker)


def test_the_pusher_runs_solo_on_the_push_queue_with_the_store() -> None:
    """019 P3 (D-18; F19-D19, D35): one push at a time, outside the driver's lock; it reads the
    bundle from /data/wsp and never holds the gateway's service token."""
    worker = next(c for c in _containers() if c["name"] == "worker-wsp-push")
    command = worker["command"]
    assert command[command.index("-Q") + 1] == "wsp_push"
    assert "--pool=solo" in command and command[command.index("-c") + 1] == "1"
    assert command[command.index("-n") + 1] == "wsp-push@%h"
    assert {m["mountPath"] for m in worker["volumeMounts"]} == {"/data/wsp"}
    assert not _mentions_token(worker)


def test_only_the_pusher_consumes_the_push_queue() -> None:
    consumers = [
        c["name"]
        for c in _containers()
        if "-Q" in (c.get("command") or [])
        and "wsp_push" in c["command"][c["command"].index("-Q") + 1].split(",")
    ]
    assert consumers == ["worker-wsp-push"]


def test_no_other_worker_consumes_the_wsp_queue() -> None:
    consumers = [
        c["name"]
        for c in _containers()
        if "-Q" in (c.get("command") or [])
        and "wsp" in c["command"][c["command"].index("-Q") + 1].split(",")
    ]
    assert consumers == ["worker-wsp"]


def test_the_internal_app_runs_in_the_backend_pod_on_8001_with_the_token() -> None:
    internal = next(c for c in _containers() if c["name"] == "wsp-internal")
    command = internal["command"]
    assert "api.wsp_internal_app:create_internal_app" in command and "--factory" in command
    assert command[command.index("--port") + 1] == "8001"
    assert _mentions_token(internal)


def test_the_internal_service_is_cluster_only_and_no_ingress_names_it() -> None:
    docs = rendered()
    service = next(
        d
        for d in docs
        if d["kind"] == "Service" and d["metadata"]["name"] == "miforge-wsp-internal"
    )
    assert service["spec"].get("type", "ClusterIP") == "ClusterIP"
    assert [p["port"] for p in service["spec"]["ports"]] == [8001]
    backends = {
        path["backend"]["service"]["name"]
        for d in docs
        if d["kind"] == "Ingress"
        for rule in d["spec"].get("rules", [])
        for path in rule.get("http", {}).get("paths", [])
    }
    assert backends, "the scan found no Ingress backend at all"
    assert "miforge-wsp-internal" not in backends
