"""The CPU-only llama.cpp editor of feature 019's P4 live proof (2026-11-25, TEMPORARY).

``gitops/pilot/wsp-cpu-editor.yaml`` is a second, generic local editor endpoint: a llama.cpp server
on CPU, beside miServer on the GPU, so two WSP runs on two local endpoints can truly overlap. Read
the way ArgoCD applies it (``tests.support.wsp_manifests.rendered``). Pinned here:

- the image, by tag AND digest, outside the kustomization's ``images:`` (Image Updater never
  retags it), on every container of the server and the fetch Job;
- no GPU: no GPU resource on any container, ``NVIDIA_VISIBLE_DEVICES=void`` on every one (nvidia is
  containerd's default runtime on the node), no host namespaces, non-root, read-only root, limits;
- the model: a pinned Hugging Face revision (never ``main``) and one sha256, checked by the fetch
  Job and again by the server's init container; the context size stated, one slot;
- the node: every pod on mcs-lnxhost01 (operator, 2026-11-25: off the GPU node), tolerating that
  node's ``no-gpu=true:NoSchedule`` taint, with the model on a ``type: Directory`` hostPath there
  (the cluster's hostpath provisioner cannot make a volume on that node: its helper pod has no
  toleration), written by the fetch Job only;
- the host: in ``miforge-config``'s ``WSP_LOCAL_HOST_ALLOWLIST``, which the backend (``envFrom``)
  and the gateway (``configMapKeyRef``) both read, resolved the way each pod sees it and passed to
  the REAL checks (``wsp.endpoints.validate_base_url``, ``wsp_gateway.upstream.check_endpoint``);
- the network: the gateway's egress names the server; the server is reached only by the gateway
  and the backend (the window read) and reaches nothing; the worker cannot reach it; only the fetch
  Job has a way out.

The ConfigMap change's ``config-rev`` bump on every consumer is ``test_gitops_config_rev.py``'s.
The negative controls are recorded in ``0xcc/reviews/019_P4_cpu_editor.md``.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import yaml

from tests.support.wsp_manifests import GITOPS, kustomization, one, rendered

IMAGE_NAME = "ghcr.io/ggml-org/llama.cpp"
IMAGE_TAG = "server-v1.0.0-b12359-b1b770d"
IMAGE_DIGEST = "sha256:e3032ea0712da96dec820c770626385de4322dd06619fb5829256a504bbc9375"
IMAGE = f"{IMAGE_NAME}:{IMAGE_TAG}@{IMAGE_DIGEST}"

MODEL_FILE = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
MODEL_REPO = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
MODEL_REVISION = "8ed288026e23958ad9dfa92d53ed773a8eee7125"
MODEL_SHA256 = "b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5"
MODEL_ALIAS = "lfm2.5-1.2b-instruct"
CTX_SIZE = 32_768

SERVER = "miforge-wsp-cpu-editor"
FETCH = "miforge-wsp-cpu-editor-fetch"
NODE = "mcs-lnxhost01"
#: The node's taint (``kubectl get node mcs-lnxhost01``, 2026-11-25), tolerated exactly.
TOLERATION = {"key": "no-gpu", "operator": "Equal", "value": "true", "effect": "NoSchedule"}
#: Created once by the operator, owned 10001:10001 (the runbook's step 0).
MODEL_DIR = "/data/miforge/wsp-cpu-editor"
SERVER_LABELS = {"app.kubernetes.io/name": SERVER}
FETCH_LABELS = {"app.kubernetes.io/name": FETCH}
WORKER_POD = {
    "app.kubernetes.io/name": "miforge-wsp-worker",
    "app.kubernetes.io/component": "wsp-worker",
}
PRIVATE = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]
EXTERNAL_ALLOW = "api.anthropic.com"


# ------------------------------------------------------------------------------------ readers


def deployment(docs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return one("Deployment", SERVER, docs)


def job(docs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return one("Job", FETCH, docs)


def pod(doc: dict[str, Any]) -> dict[str, Any]:
    spec: dict[str, Any] = doc["spec"]["template"]["spec"]
    return spec


def containers(doc: dict[str, Any]) -> list[dict[str, Any]]:
    spec = pod(doc)
    return [*(spec.get("initContainers") or []), *spec["containers"]]


def server_container(docs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    (only,) = pod(deployment(docs))["containers"]
    assert only["name"] == "llama-server"
    return dict(only)


def args_of(container: dict[str, Any]) -> dict[str, str | None]:
    """``--flag value`` pairs (a flag followed by another flag has the value ``None``)."""
    args = [str(a) for a in container["args"]]
    out: dict[str, str | None] = {}
    for i, arg in enumerate(args):
        if arg.startswith("--"):
            nxt = args[i + 1] if i + 1 < len(args) else None
            out[arg] = None if nxt is None or nxt.startswith("--") else nxt
    return out


def env_of(container: dict[str, Any]) -> dict[str, Any]:
    return {e["name"]: e.get("value") for e in container.get("env") or []}


def policy(name: str, docs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    spec: dict[str, Any] = one("NetworkPolicy", name, docs)["spec"]
    return spec


def config_data(docs: list[dict[str, Any]] | None = None) -> dict[str, str]:
    return dict(one("ConfigMap", "miforge-config", docs)["data"])


def template_labels(
    kind: str, name: str, docs: list[dict[str, Any]] | None = None
) -> dict[str, str]:
    labels: dict[str, str] = one(kind, name, docs)["spec"]["template"]["metadata"]["labels"]
    return labels


def selects(selector: dict[str, Any], labels: dict[str, str]) -> bool:
    """Whether a podSelector (matchLabels only; an expression is treated as selecting) admits
    ``labels``. An empty selector selects every pod."""
    if selector.get("matchExpressions"):
        return True
    return all(labels.get(k) == v for k, v in (selector.get("matchLabels") or {}).items())


def service_host(docs: list[dict[str, Any]] | None = None) -> tuple[str, int]:
    svc = one("Service", SERVER, docs)
    (port,) = svc["spec"]["ports"]
    return f"{svc['metadata']['name']}.{svc['metadata']['namespace']}.svc", int(port["port"])


def base_url(docs: list[dict[str, Any]] | None = None) -> str:
    host, port = service_host(docs)
    return f"http://{host}:{port}/v1"


def resolved_env(container: dict[str, Any], data: dict[str, str]) -> dict[str, str]:
    """The environment a container sees from ``miforge-config``: ``envFrom`` first, then ``env``
    (a literal or a ``configMapKeyRef`` into it), as Kubernetes resolves them."""
    out: dict[str, str] = {}
    for source in container.get("envFrom") or []:
        if (source.get("configMapRef") or {}).get("name") == "miforge-config":
            out.update(data)
    for entry in container.get("env") or []:
        if "value" in entry:
            out[entry["name"]] = str(entry["value"])
            continue
        ref = (entry.get("valueFrom") or {}).get("configMapKeyRef") or {}
        if ref.get("name") == "miforge-config" and ref.get("key") in data:
            out[entry["name"]] = data[ref["key"]]
    return out


# ------------------------------------------------------------------------------- the listing


def test_listed_in_the_kustomization() -> None:
    assert "wsp-cpu-editor.yaml" in kustomization()["resources"]
    kinds = sorted(
        (d["kind"], d["metadata"]["name"])
        for d in yaml.safe_load_all((GITOPS / "wsp-cpu-editor.yaml").read_text())
        if d
    )
    assert kinds == [
        ("Deployment", SERVER),
        ("Job", FETCH),
        ("NetworkPolicy", "wsp-cpu-editor"),
        ("NetworkPolicy", "wsp-cpu-editor-fetch"),
        ("Service", SERVER),
    ]
    assert all(
        d["metadata"]["namespace"] == "miforge"
        for d in yaml.safe_load_all((GITOPS / "wsp-cpu-editor.yaml").read_text())
        if d
    )


# ---------------------------------------------------------------------------------- the image


def image_problems(docs: list[dict[str, Any]]) -> list[str]:
    found = []
    for doc in (deployment(docs), job(docs)):
        for c in containers(doc):
            if c["image"] != IMAGE:
                found.append(f"{doc['metadata']['name']}/{c['name']}: {c['image']}")
            if c.get("imagePullPolicy") != "IfNotPresent":
                found.append(
                    f"{doc['metadata']['name']}/{c['name']}: pull {c.get('imagePullPolicy')}"
                )
    return found


def test_the_image_is_pinned_by_tag_and_digest_on_every_container() -> None:
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", IMAGE_DIGEST)
    docs = rendered()
    assert (
        len(containers(deployment(docs))) == 2 and len(containers(job(docs))) == 1
    ), "precondition"
    assert image_problems(docs) == []


def test_the_image_check_sees_a_moved_digest() -> None:
    """Negative control, kept: a different digest, a bare tag, or a floating tag is found."""
    for bad in (
        IMAGE.replace(IMAGE_DIGEST, "sha256:" + "0" * 64),
        f"{IMAGE_NAME}:{IMAGE_TAG}",
        f"{IMAGE_NAME}:server",
    ):
        docs = yaml.safe_load_all(
            (GITOPS / "wsp-cpu-editor.yaml").read_text().replace(IMAGE, bad, 1)
        )
        assert image_problems([d for d in docs if d]), bad


def test_image_updater_never_retags_it() -> None:
    """Outside kustomize's ``images:`` and the Application's image list: the digest above is the
    only one ever deployed (test_image_updater_tracks_every_image.py pins the two lists equal)."""
    names = {entry["name"] for entry in kustomization()["images"]}
    assert not any("llama" in n for n in names), names
    app = (GITOPS / "argocd" / "miforge-app.yaml").read_text()
    assert "llama" not in app


# ------------------------------------------------------------------------------------ no GPU


def gpu_problems(docs: list[dict[str, Any]]) -> list[str]:
    found = []
    for doc in (deployment(docs), job(docs)):
        name = doc["metadata"]["name"]
        spec = pod(doc)
        for field in ("runtimeClassName", "hostIPC", "hostPID", "hostNetwork"):
            if spec.get(field):
                found.append(f"{name}: {field}")
        for c in containers(doc):
            for part in ("requests", "limits"):
                for key in (c.get("resources") or {}).get(part) or {}:
                    if "gpu" in key.lower():
                        found.append(f"{name}/{c['name']}: {part} {key}")
            if env_of(c).get("NVIDIA_VISIBLE_DEVICES") != "void":
                found.append(f"{name}/{c['name']}: NVIDIA_VISIBLE_DEVICES")
            if any(k.startswith("NVIDIA_") and k != "NVIDIA_VISIBLE_DEVICES" for k in env_of(c)):
                found.append(f"{name}/{c['name']}: another NVIDIA_ variable")
    return found


def test_no_gpu_on_any_container() -> None:
    assert gpu_problems(rendered()) == []


def test_the_gpu_check_sees_a_planted_request() -> None:
    """Negative control, kept: a GPU limit on the server, a visible device, or a runtime class."""
    text = (GITOPS / "wsp-cpu-editor.yaml").read_text()
    planted = [
        text.replace(
            '              cpu: "6"\n',
            '              cpu: "6"\n              nvidia.com/gpu: 1\n',
            1,
        ),
        text.replace("              value: void\n", "              value: all\n", 1),
        text.replace(
            "      enableServiceLinks: false\n",
            "      enableServiceLinks: false\n      runtimeClassName: nvidia\n",
            1,
        ),
    ]
    for variant in planted:
        assert variant != text
        assert gpu_problems([d for d in yaml.safe_load_all(variant) if d])


def test_non_root_read_only_no_token_and_limits() -> None:
    for doc in (deployment(), job()):
        spec = pod(doc)
        assert spec["automountServiceAccountToken"] is False and "serviceAccountName" not in spec
        assert spec["enableServiceLinks"] is False
        ctx = spec["securityContext"]
        assert (ctx["runAsNonRoot"], ctx["runAsUser"], ctx["runAsGroup"]) == (True, 10001, 10001)
        assert ctx["seccompProfile"] == {"type": "RuntimeDefault"}
        for c in containers(doc):
            sc = c["securityContext"]
            assert sc["readOnlyRootFilesystem"] is True and sc["runAsNonRoot"] is True, c["name"]
            assert sc["allowPrivilegeEscalation"] is False and sc["privileged"] is False
            assert sc["capabilities"] == {"drop": ["ALL"]}
            assert "runAsUser" not in sc, c["name"]
            assert {"cpu", "memory"} <= set(c["resources"]["limits"]), c["name"]
            assert {"cpu", "memory"} <= set(c["resources"]["requests"]), c["name"]


def test_the_server_leaves_the_node_room() -> None:
    """Six CPUs and 3 GiB at most (about 1.0 GiB measured resident, 2026-11-25), and as many
    threads as CPUs; the figures are deliberate, so a change shows here."""
    c = server_container()
    assert c["resources"] == {
        "requests": {"cpu": "2", "memory": "1536Mi"},
        "limits": {"cpu": "6", "memory": "3Gi"},
    }
    assert args_of(c)["--threads"] == c["resources"]["limits"]["cpu"]
    assert deployment()["spec"]["replicas"] == 1


# ---------------------------------------------------------------------------------- the model


def test_the_fetch_is_pinned_to_a_revision_and_a_sha256() -> None:
    fetch_env = env_of(pod(job())["containers"][0])
    assert fetch_env["MODEL_FILE"] == MODEL_FILE
    assert fetch_env["MODEL_URL"] == (
        f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}"
    )
    assert re.fullmatch(r"[0-9a-f]{40}", MODEL_REVISION)
    assert fetch_env["MODEL_SHA256"] == MODEL_SHA256
    script = pod(job())["containers"][0]["args"][0]
    # The rename happens only after the check, so the file exists only once verified.
    assert script.index("sha256sum -c -") < script.index('mv "$f.partial" "$f"')
    assert "--proto '=https'" in script and "--proto-redir '=https'" in script
    assert "ttlSecondsAfterFinished" not in job()["spec"]  # ArgoCD would re-create it
    assert pod(job())["restartPolicy"] == "OnFailure"


def test_the_server_verifies_the_same_file_before_it_starts() -> None:
    spec = pod(deployment())
    (wait,) = spec["initContainers"]
    assert env_of(wait)["MODEL_FILE"] == MODEL_FILE
    assert (
        env_of(wait)["MODEL_SHA256"]
        == MODEL_SHA256
        == env_of(pod(job())["containers"][0])["MODEL_SHA256"]
    )
    assert "sha256sum -c -" in wait["args"][0]
    assert args_of(server_container())["--model"] == f"/model/{MODEL_FILE}"


def placement_problems(docs: list[dict[str, Any]]) -> list[str]:
    """Every pod on the one node, tolerating its taint, the model on its hostPath there, written
    by the fetch Job only."""
    found = []
    volume = {"name": "model", "hostPath": {"path": MODEL_DIR, "type": "Directory"}}
    for doc, writable in ((deployment(docs), False), (job(docs), True)):
        name = doc["metadata"]["name"]
        spec = pod(doc)
        if spec.get("nodeSelector") != {"kubernetes.io/hostname": NODE}:
            found.append(f"{name}: nodeSelector {spec.get('nodeSelector')}")
        if spec.get("tolerations") != [TOLERATION]:
            found.append(f"{name}: tolerations {spec.get('tolerations')}")
        if spec.get("affinity") or spec.get("nodeName"):
            found.append(f"{name}: another placement rule")
        if spec.get("volumes") != [volume]:
            found.append(f"{name}: volumes {spec.get('volumes')}")
        for c in containers(doc):
            mounts = c.get("volumeMounts") or []
            if [m.get("mountPath") for m in mounts] != ["/model"]:
                found.append(f"{name}/{c['name']}: mounts")
            elif bool(mounts[0].get("readOnly")) is writable:
                found.append(f"{name}/{c['name']}: readOnly {mounts[0].get('readOnly')}")
    if any(
        d["kind"] == "PersistentVolumeClaim" and "cpu-editor" in d["metadata"]["name"] for d in docs
    ):
        found.append("a claim: the hostpath provisioner cannot make one on this node")
    return found


def test_placed_on_lnxhost01_with_the_model_on_its_host() -> None:
    assert placement_problems(rendered()) == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("kubernetes.io/hostname: mcs-lnxhost01\n", "kubernetes.io/hostname: mcs-lnxhost02\n"),
        ("        - key: no-gpu\n", "        - key: gpu\n"),
        ("            type: Directory\n", "            type: DirectoryOrCreate\n"),
        (
            "              mountPath: /model\n              readOnly: true\n",
            "              mountPath: /model\n",
        ),
    ],
    ids=["gpu-node", "wrong-toleration", "created-empty", "server-writes"],
)
def test_the_placement_check_sees_each_planted_change(old: str, new: str) -> None:
    """Negative controls, kept: the GPU node, a toleration for another taint, a directory the
    kubelet would create empty, and the server's mount made writable (each in its first place)."""
    text = (GITOPS / "wsp-cpu-editor.yaml").read_text()
    assert old in text, "the planted edit matched nothing: a no-op control"
    docs = [d for d in yaml.safe_load_all(text.replace(old, new, 1)) if d]
    assert placement_problems(docs), (old, new)


def test_the_server_flags() -> None:
    args = args_of(server_container())
    assert args["--ctx-size"] == str(CTX_SIZE)
    assert args["--parallel"] == "1"  # one slot: /props' per-slot n_ctx is the whole context
    assert args["--alias"] == MODEL_ALIAS
    assert args["--port"] == "8080"
    assert args["--cache-ram"] == "0"
    for flag in ("--jinja", "--offline", "--no-webui", "--no-slots", "--no-agent"):
        assert flag in args and args[flag] is None, flag
    for forbidden in (
        "--tools",
        "--agent",
        "--api-key",
        "--api-key-file",
        "--props",
        "--mcp-servers-json",
    ):
        assert forbidden not in args, forbidden


# ----------------------------------------------------------------------------- the allowlist


def backend_containers(docs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    spec = pod(one("Deployment", "miforge-backend", docs))
    return [*(spec.get("initContainers") or []), *spec["containers"]]


def gateway_container(docs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    (only,) = pod(one("Deployment", "miforge-wsp-gateway", docs))["containers"]
    return dict(only)


def allowlist_problems(docs: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Each pod's view of the allowlist, through the real settings classes and the real checks."""
    from core.errors import Refusal
    from wsp.endpoints import validate_base_url
    from wsp.settings import WSPSettings
    from wsp_gateway.settings import GatewaySettings
    from wsp_gateway.upstream import EndpointRefused, check_endpoint

    data = config_data(docs)
    url = base_url(docs)
    found = []
    # The backend: every container that reads miforge-config sees the key (registration runs in
    # the API container, but the settings are the same class everywhere).
    readers = [c for c in backend_containers(docs) if c.get("envFrom")]
    if not readers:
        found.append("no backend container reads miforge-config")
    for c in readers:
        env = resolved_env(c, data)
        with monkeypatch.context() as m:
            for name in [n for n in env if n.startswith("WSP_")]:
                m.setenv(name, env[name])
            if "WSP_LOCAL_HOST_ALLOWLIST" not in env:
                m.delenv("WSP_LOCAL_HOST_ALLOWLIST", raising=False)
            settings = WSPSettings()
        try:
            locality = validate_base_url(
                url,
                local_allowlist=settings.wsp_local_host_allowlist,
                external_allowlist=settings.wsp_external_host_allowlist,
            )
            if locality != "local":
                found.append(f"backend/{c['name']}: {locality}")
        except Refusal as exc:
            found.append(f"backend/{c['name']}: {exc}")
    # The gateway: its env, resolved against the ConfigMap.
    env = resolved_env(gateway_container(docs), data)
    with monkeypatch.context() as m:
        m.delenv("WSP_LOCAL_HOST_ALLOWLIST", raising=False)
        for name in [
            n for n in env if n.startswith("WSP_") and "KEY" not in n and "TOKEN" not in n
        ]:
            m.setenv(name, env[name])
        gateway = GatewaySettings()
    try:
        locality = check_endpoint(
            {"base_url": url, "kind": "openai_compatible", "locality": "local"},
            local_allowlist=gateway.wsp_local_host_allowlist,
            external_allowlist=gateway.wsp_external_host_allowlist,
        )
        if locality != "local":
            found.append(f"gateway: {locality}")
    except EndpointRefused as exc:
        found.append(f"gateway: {exc}")
    # miServer stays admitted on both sides.
    for host in ("miserver-backend.miserver.svc", service_host(docs)[0]):
        if host not in {h.strip() for h in data.get("WSP_LOCAL_HOST_ALLOWLIST", "").split(",")}:
            found.append(f"configmap: {host}")
    return found


def test_the_host_is_admitted_by_the_backend_and_the_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert base_url() == "http://miforge-wsp-cpu-editor.miforge.svc:8080/v1"
    assert allowlist_problems(rendered(), monkeypatch) == []
    ref = next(e for e in gateway_container()["env"] if e["name"] == "WSP_LOCAL_HOST_ALLOWLIST")
    assert ref == {
        "name": "WSP_LOCAL_HOST_ALLOWLIST",
        "valueFrom": {
            "configMapKeyRef": {"name": "miforge-config", "key": "WSP_LOCAL_HOST_ALLOWLIST"}
        },
    }


def test_the_allowlist_check_sees_a_missing_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative controls, kept: the host dropped from the ConfigMap (both pods refuse it), and the
    gateway back on its old literal (the gateway alone refuses it)."""
    text = {name: (GITOPS / name).read_text() for name in kustomization()["resources"]}
    host = "miforge-wsp-cpu-editor.miforge.svc"

    def render(changed: dict[str, str]) -> list[dict[str, Any]]:
        merged = {**text, **changed}
        return [d for name in merged for d in yaml.safe_load_all(merged[name]) if d]

    dropped = text["config.yaml"].replace("," + host, "")
    assert dropped != text["config.yaml"]
    problems = allowlist_problems(render({"config.yaml": dropped}), monkeypatch)
    assert any(p.startswith("backend/") for p in problems) and any(
        p.startswith("gateway:") for p in problems
    ), problems

    literal = text["wsp-gateway.yaml"].replace(
        "              valueFrom:\n"
        "                configMapKeyRef:\n"
        "                  name: miforge-config\n"
        "                  key: WSP_LOCAL_HOST_ALLOWLIST\n",
        "              value: miserver-backend.miserver.svc\n",
    )
    assert literal != text["wsp-gateway.yaml"]
    problems = allowlist_problems(render({"wsp-gateway.yaml": literal}), monkeypatch)
    assert [p for p in problems if p.startswith("gateway:")] and not [
        p for p in problems if p.startswith("backend/")
    ], problems


def test_the_settings_defaults_stay_miserver_only() -> None:
    """The admission is the deployed config's, not a code default: dev and tests keep miServer."""
    from wsp.settings import WSPSettings
    from wsp_gateway.settings import GatewaySettings

    assert WSPSettings.model_fields["wsp_local_host_allowlist"].default == (
        "miserver-backend.miserver.svc"
    )
    assert GatewaySettings.model_fields["wsp_local_host_allowlist"].default == (
        "miserver-backend.miserver.svc"
    )


# ------------------------------------------------------------------------------- the network


def network_problems(docs: list[dict[str, Any]]) -> list[str]:
    found = []
    server_pod = template_labels("Deployment", SERVER, docs)
    fetch_pod = template_labels("Job", FETCH, docs)
    gateway_pod = template_labels("Deployment", "miforge-wsp-gateway", docs)
    backend_pod = template_labels("Deployment", "miforge-backend", docs)
    worker_pod = template_labels("StatefulSet", "miforge-wsp-worker", docs)
    (port,) = server_container(docs)["ports"]
    svc = one("Service", SERVER, docs)["spec"]
    if svc.get("type", "ClusterIP") != "ClusterIP" or svc["selector"] != SERVER_LABELS:
        found.append("service: not a ClusterIP on the server's pods")
    if [(p["port"], p["targetPort"]) for p in svc["ports"]] != [(8080, port["containerPort"])]:
        found.append("service: port")
    if any(SERVER in repr(d) for d in docs if d["kind"] == "Ingress"):
        found.append("an Ingress names the server")

    # The gateway reaches the server, on its port.
    gateway_rules = policy("wsp-gateway", docs)["egress"]
    if not any(
        any(
            "podSelector" in peer
            and "namespaceSelector" not in peer
            and selects(peer["podSelector"], server_pod)
            for peer in rule["to"]
        )
        and rule["ports"] == [{"protocol": "TCP", "port": port["containerPort"]}]
        for rule in gateway_rules
    ):
        found.append("gateway egress: no rule to the server's port")

    # The server: reached by the gateway and the backend only, on 8080; reaches nothing.
    server = policy("wsp-cpu-editor", docs)
    if server["podSelector"] != {"matchLabels": SERVER_LABELS} or sorted(server["policyTypes"]) != [
        "Egress",
        "Ingress",
    ]:
        found.append("server policy: selector or types")
    if server.get("egress") not in ([], None) or "Egress" not in server["policyTypes"]:
        found.append("server policy: an egress rule")
    peers = [peer for rule in server.get("ingress") or [] for peer in rule.get("from") or []]
    if any("ipBlock" in p or "namespaceSelector" in p or "podSelector" not in p for p in peers):
        found.append("server ingress: a peer outside this namespace's pods")
    if not any(selects(p["podSelector"], gateway_pod) for p in peers if "podSelector" in p):
        found.append("server ingress: the gateway is not admitted")
    if not any(selects(p["podSelector"], backend_pod) for p in peers if "podSelector" in p):
        found.append("server ingress: the backend (window read) is not admitted")
    if any(selects(p["podSelector"], worker_pod) for p in peers if "podSelector" in p):
        found.append("server ingress: the worker is admitted")
    if any(not rule.get("from") for rule in server.get("ingress") or []):
        found.append("server ingress: a rule with no peer admits everyone")
    if any(
        rule.get("ports") != [{"protocol": "TCP", "port": 8080}]
        for rule in server.get("ingress") or []
    ):
        found.append("server ingress: a port other than 8080")

    # The worker reaches the gateway only: none of its egress peers is the server.
    for rule in policy("wsp-worker", docs)["egress"]:
        if any(selects(peer.get("podSelector") or {}, server_pod) for peer in rule["to"]):
            found.append("worker egress: reaches the server")

    # Policies are additive: no other policy may select the server or the fetch pods.
    for d in docs:
        if d["kind"] != "NetworkPolicy":
            continue
        name, selector = d["metadata"]["name"], d["spec"]["podSelector"]
        if selects(selector, server_pod) and name != "wsp-cpu-editor":
            found.append(f"{name} also selects the server")
        if selects(selector, fetch_pod) and name != "wsp-cpu-editor-fetch":
            found.append(f"{name} also selects the fetch Job")

    # The fetch Job: no ingress; 443 outside the private ranges and DNS only.
    fetch = policy("wsp-cpu-editor-fetch", docs)
    if fetch["podSelector"] != {"matchLabels": FETCH_LABELS} or fetch.get("ingress") not in (
        [],
        None,
    ):
        found.append("fetch policy: selector or ingress")
    if fetch["egress"] != [
        {
            "to": [{"ipBlock": {"cidr": "0.0.0.0/0", "except": PRIVATE}}],
            "ports": [{"protocol": "TCP", "port": 443}],
        },
        {
            "to": [
                {
                    "namespaceSelector": {
                        "matchLabels": {"kubernetes.io/metadata.name": "kube-system"}
                    },
                    "podSelector": {"matchLabels": {"k8s-app": "kube-dns"}},
                }
            ],
            "ports": [{"protocol": "UDP", "port": 53}, {"protocol": "TCP", "port": 53}],
        },
    ]:
        found.append("fetch policy: egress")
    if server_pod == fetch_pod or selects({"matchLabels": FETCH_LABELS}, server_pod):
        found.append("the server carries the fetch Job's labels (and its way out)")
    return found


def test_the_network_is_exactly_as_designed() -> None:
    assert template_labels("StatefulSet", "miforge-wsp-worker") == WORKER_POD, "precondition"
    assert network_problems(rendered()) == []


@pytest.mark.parametrize(
    ("file", "old", "new"),
    [
        # The worker admitted to the server.
        (
            "wsp-cpu-editor.yaml",
            "              app: miforge-backend\n      ports:\n"
            "        - protocol: TCP\n          port: 8080\n  egress: []\n",
            "              app: miforge-backend\n        - podSelector:\n            matchLabels:\n"
            "              app.kubernetes.io/name: miforge-wsp-worker\n      ports:\n"
            "        - protocol: TCP\n          port: 8080\n  egress: []\n",
        ),
        # The server given a way out.
        (
            "wsp-cpu-editor.yaml",
            "  egress: []\n",
            "  egress:\n    - to:\n        - ipBlock:\n            cidr: 0.0.0.0/0\n",
        ),
        # The gateway's rule to the server removed.
        (
            "wsp-networkpolicy.yaml",
            "    - to:\n        - podSelector:\n            matchLabels:\n"
            "              app.kubernetes.io/name: miforge-wsp-cpu-editor\n",
            "    - to:\n        - podSelector:\n            matchLabels:\n"
            "              app.kubernetes.io/name: miforge-wsp-elsewhere\n",
        ),
        # The worker's egress widened to the server.
        (
            "wsp-networkpolicy.yaml",
            "  egress:\n    - to:\n        - podSelector:\n            matchLabels:\n"
            "              app.kubernetes.io/name: miforge-wsp-gateway\n",
            "  egress:\n    - to:\n        - podSelector:\n            matchLabels:\n"
            "              app.kubernetes.io/name: miforge-wsp-cpu-editor\n"
            "    - to:\n        - podSelector:\n            matchLabels:\n"
            "              app.kubernetes.io/name: miforge-wsp-gateway\n",
        ),
        # A second policy selecting every pod in the namespace (policies are additive).
        (
            "wsp-cpu-editor.yaml",
            "---\n# The fetch Job: reached by nothing",
            "---\napiVersion: networking.k8s.io/v1\nkind: NetworkPolicy\nmetadata:\n"
            "  name: allow-all\n  namespace: miforge\nspec:\n  podSelector: {}\n"
            "  policyTypes: [Ingress]\n  ingress:\n    - {}\n"
            "---\n# The fetch Job: reached by nothing",
        ),
    ],
    ids=[
        "worker-admitted",
        "server-egress",
        "gateway-rule-gone",
        "worker-egress",
        "another-policy",
    ],
)
def test_the_network_check_sees_each_planted_breach(file: str, old: str, new: str) -> None:
    """Negative controls, kept: each breach of the design is found by the same check."""
    text = {name: (GITOPS / name).read_text() for name in kustomization()["resources"]}
    assert text[file].count(old) == 1, "the planted edit matched nothing: a no-op control"
    text[file] = text[file].replace(old, new)
    docs = [d for name in text for d in yaml.safe_load_all(text[name]) if d]
    assert network_problems(docs), "the planted breach was not found"
