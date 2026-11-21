"""FR-016.13: the WSP NetworkPolicies, asserted exactly (FTDD §8.2; FTASKS 4.7).

Worker: ingress only from the backend pod on 9000, egress only to the gateway on 8080, no port 53
(P37), no backend egress (P49). Gateway: the four egress peers FTDD §8.2 names. And the labels other
chains rely on (miServer and miWorkbench write no rule against them, D17, but their live checks
select by them).
"""

from __future__ import annotations

from typing import Any

from tests.support.wsp_manifests import kustomization, one, rendered

WORKER = {"app.kubernetes.io/name": "miforge-wsp-worker"}
GATEWAY = {"app.kubernetes.io/name": "miforge-wsp-gateway"}
BACKEND = {"app": "miforge-backend"}


def policy(name: str) -> dict[str, Any]:
    return one("NetworkPolicy", name)["spec"]


def test_listed_in_the_kustomization() -> None:
    assert "wsp-networkpolicy.yaml" in kustomization()["resources"]


def test_worker_policy_selects_the_worker_both_directions() -> None:
    spec = policy("wsp-worker")
    assert spec["podSelector"] == {"matchLabels": WORKER}
    assert sorted(spec["policyTypes"]) == ["Egress", "Ingress"]


def test_worker_ingress_only_from_the_backend() -> None:
    assert policy("wsp-worker")["ingress"] == [
        {
            "from": [{"podSelector": {"matchLabels": BACKEND}}],
            "ports": [{"protocol": "TCP", "port": 9000}],
        }
    ]


def test_worker_egress_only_to_the_gateway() -> None:
    assert policy("wsp-worker")["egress"] == [
        {
            "to": [{"podSelector": {"matchLabels": GATEWAY}}],
            "ports": [{"protocol": "TCP", "port": 8080}],
        }
    ]


def test_worker_has_no_dns_and_no_ip_block() -> None:
    text = repr(policy("wsp-worker"))
    assert "53" not in [
        str(p.get("port")) for rule in policy("wsp-worker")["egress"] for p in rule["ports"]
    ]
    assert "ipBlock" not in text and "namespaceSelector" not in text


def test_gateway_ingress_only_from_the_worker() -> None:
    spec = policy("wsp-gateway")
    assert spec["podSelector"] == {"matchLabels": GATEWAY}
    assert spec["ingress"] == [
        {
            "from": [{"podSelector": {"matchLabels": WORKER}}],
            "ports": [{"protocol": "TCP", "port": 8080}],
        }
    ]


def test_gateway_egress_is_exactly_its_four_peers() -> None:
    egress = policy("wsp-gateway")["egress"]
    assert len(egress) == 4
    miserver, internet, backend, dns = egress
    assert miserver["to"] == [
        {
            "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "miserver"}},
            "podSelector": {"matchLabels": {"app": "miserver-backend"}},
        }
    ] and miserver["ports"] == [{"protocol": "TCP", "port": 8000}]
    assert internet["to"] == [
        {
            "ipBlock": {
                "cidr": "0.0.0.0/0",
                "except": ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"],
            }
        }
    ] and internet["ports"] == [{"protocol": "TCP", "port": 443}]
    assert backend["to"] == [{"podSelector": {"matchLabels": BACKEND}}]
    assert backend["ports"] == [{"protocol": "TCP", "port": 8001}]
    assert dns["ports"] == [{"protocol": "UDP", "port": 53}, {"protocol": "TCP", "port": 53}]


def test_label_pins_for_the_sibling_chains() -> None:
    worker = one("Deployment", "miforge-wsp-worker")
    labels = worker["spec"]["template"]["metadata"]["labels"]
    assert labels["app.kubernetes.io/name"] == "miforge-wsp-worker"
    assert labels["app.kubernetes.io/component"] == "wsp-worker"
    assert worker["metadata"]["namespace"] == "miforge"


def _selects_worker(selector: dict[str, Any]) -> bool:
    wanted = selector.get("matchLabels", {})
    labels = {**WORKER, "app.kubernetes.io/component": "wsp-worker"}
    return not selector.get("matchExpressions") and all(
        labels.get(k) == v for k, v in wanted.items()
    )


def test_no_other_policy_admits_the_worker() -> None:
    """A second policy selecting the worker (by its labels, or an empty selector) would widen it:
    policies are additive."""
    selecting = [
        d["metadata"]["name"]
        for d in rendered()
        if d["kind"] == "NetworkPolicy" and _selects_worker(d["spec"]["podSelector"])
    ]
    assert selecting == ["wsp-worker"]
