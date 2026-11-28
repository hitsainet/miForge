"""019 P3 (D-17; FR-019.7, F19-D18): each WSP worker is addressed on its own, never through a
load-balanced name.

The StatefulSet's pods are ``miforge-wsp-worker-<i>``; the headless Service ``miforge-wsp-workers``
gives each its own DNS name, published while not ready, so the name survives the container restart
at the end of every session. 016's ClusterIP Service ``miforge-wsp-worker`` is gone from the
rendered set (ArgoCD prune is off, so the operator deletes the live one by hand, FTASKS 20.5).
"""

from __future__ import annotations

from typing import Any

from tests.support.wsp_manifests import documents, one, rendered
from wsp import scheduling
from wsp.settings import WSPSettings

NAME = "miforge-wsp-worker"
SERVICE = "miforge-wsp-workers"


def _config() -> dict[str, Any]:
    (config,) = [d for d in documents("config.yaml") if d["kind"] == "ConfigMap"]
    data: dict[str, Any] = config["data"]
    return data


def test_the_headless_service_selects_the_workers_on_9000() -> None:
    service = one("Service", SERVICE)
    assert service["metadata"]["namespace"] == "miforge"
    assert service["spec"]["selector"] == {"app.kubernetes.io/name": NAME}
    assert service["spec"]["clusterIP"] == "None"
    assert service["spec"]["publishNotReadyAddresses"] is True
    assert [(p["port"], p["targetPort"]) for p in service["spec"]["ports"]] == [(9000, 9000)]


def test_no_load_balanced_worker_service_remains() -> None:
    services = [d for d in rendered() if d["kind"] == "Service"]
    assert len(services) > 3, "the scan read almost nothing"
    assert NAME not in {d["metadata"]["name"] for d in services}
    selecting = [
        d["metadata"]["name"]
        for d in services
        if (d["spec"].get("selector") or {}).get("app.kubernetes.io/name") == NAME
    ]
    assert selecting == [SERVICE], "only the headless Service may select the workers"


def test_the_statefulset_names_the_headless_service() -> None:
    assert one("StatefulSet", NAME)["spec"]["serviceName"] == SERVICE


def test_the_worker_url_expands_to_each_pod_under_the_headless_service() -> None:
    """The driver's template names exactly ``<statefulset>-<i>.<serviceName>.<namespace>``, so
    worker i is the pod with ordinal i, for every i below the provisioned maximum."""
    data = _config()
    url = data["WSP_WORKER_URL"]
    assert scheduling.INDEX_PLACEHOLDER in url
    assert url == f"http://{NAME}-{{index}}.{SERVICE}.miforge.svc.cluster.local:9000"
    maximum = int(data["WSP_WORKER_POOL_MAX"])
    settings = WSPSettings(wsp_worker_url=url, wsp_worker_pool_max=maximum)
    assert scheduling.worker_urls(settings) == [
        f"http://{NAME}-{i}.{SERVICE}.miforge.svc.cluster.local:9000" for i in range(maximum)
    ]


def test_argocd_tolerates_a_restart_of_the_statefulset_as_of_a_deployment() -> None:
    """FTASKS 15.3: ``kubectl rollout restart`` writes ``kubectl.kubernetes.io/restartedAt`` on the
    pod template of a StatefulSet as of a Deployment (kubectl's ``objectrestarter``: one branch per
    kind, the same annotation), so selfHeal must ignore that pointer for both kinds."""
    import yaml

    from tests.support.wsp_manifests import GITOPS

    (app,) = [
        d
        for d in yaml.safe_load_all((GITOPS / "argocd" / "miforge-app.yaml").read_text())
        if d and d.get("kind") == "Application"
    ]
    pointer = "/spec/template/metadata/annotations/kubectl.kubernetes.io~1restartedAt"
    ignored = {
        entry["kind"]: entry["jsonPointers"]
        for entry in app["spec"]["ignoreDifferences"]
        if entry.get("group") == "apps"
    }
    assert ignored == {"Deployment": [pointer], "StatefulSet": [pointer]}
