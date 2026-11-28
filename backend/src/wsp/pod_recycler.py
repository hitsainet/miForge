"""A fresh WSP worker pod per session (operator 2026-11-26: option B of the 019 P4 restart debt).

**Why.** The supervisor serves one session per process and, once the result is acknowledged (or the
backend recycles a stale boot), wipes and exits as PID 1 (``wsp_worker.supervisor``; P49, P51).
That stays exactly as it is. Before this module the kubelet then restarted the container, and
because a StatefulSet allows only ``restartPolicy: Always`` each exit counted toward
CrashLoopBackOff: 10 s, 20 s, 40 s ... up to 5 minutes, reset only after about 10 minutes of uptime.
Back-to-back runs waited, and the pool read the worker ``not_answering`` meanwhile (P4: 6 restarts).

**What.** After the driver acks, recycles a stale boot or retires an idle one before its
supervisor's own idle recycle (``dispatch.RETIRE_BEFORE_S``), it deletes the worker's pod
(``WSP_WORKER_POD_NAME`` with the worker's index, in ``WSP_WORKER_POD_NAMESPACE``). The
StatefulSet creates a new pod at once, and a new pod pays no back-off; it also gets new emptyDirs
and a new IPC namespace. Only ``get`` and ``delete`` on exactly the worker pods' names are
granted (``gitops/pilot/wsp-driver-rbac.yaml``), and the token is mounted in the driver's
container only.

**Safe to repeat.** The pod is read first. A pod that is gone (404), or already being deleted
(``deletionTimestamp`` set), is done. A pod created after
the session's boot was first seen is already the replacement and is left alone. Otherwise the
delete carries the read pod's ``uid`` as a precondition, so it can never remove a newer pod: a
409 means the pod was replaced meanwhile, a 404 that it is gone, and both are done.

**Never fails a run.** Every refusal (no token outside a cluster, 401 or 403 without the Role, the
API server unreachable, any other answer) is logged ``wsp_worker_pod_delete_failed`` and returned
as an outcome word. Nothing raises: the container restart of before still happens on its own.

The client speaks plain HTTPS to the API server with ``httpx`` (no ``kubernetes`` package; FR-016.14
keeps it out of ``backend/src``).
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path
from typing import Any, Protocol

import httpx

log = logging.getLogger(__name__)

#: Where the projected token, its CA and nothing else are mounted in the driver's container
#: (gitops/pilot/backend.yaml, ``worker-wsp`` only).
SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
#: One call to the API server; the driver's pass holds its lock meanwhile.
TIMEOUT_S = 5.0
#: The log code of every refusal.
FAILED_CODE = "wsp_worker_pod_delete_failed"

DELETED, GONE, REPLACED, DELETING = (
    "deleted",
    "already_gone",
    "already_replaced",
    "already_deleting",
)


class PodRecycler(Protocol):
    """What the driver calls after an ack: replace worker ``index``'s pod, which has served the
    boot first seen at ``served_since``. Returns an outcome word and never raises."""

    def recycle(self, index: int, served_since: dt.datetime) -> str: ...


def _aware(value: dt.datetime) -> dt.datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)


def _created(pod: dict[str, Any]) -> dt.datetime | None:
    stamp = (pod.get("metadata") or {}).get("creationTimestamp")
    if not isinstance(stamp, str):
        return None
    try:
        return _aware(dt.datetime.fromisoformat(stamp.replace("Z", "+00:00")))
    except ValueError:
        return None


class KubernetesPodRecycler:
    """The production :class:`PodRecycler`: the in-cluster API server, the projected token."""

    def __init__(
        self,
        name_template: str,
        namespace: str,
        *,
        account_dir: Path = SERVICE_ACCOUNT_DIR,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.name_template = name_template
        self.namespace = namespace
        self.account_dir = account_dir
        host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        self.base_url = base_url or f"https://{host}:{port}"
        self.transport = transport

    def pod_name(self, index: int) -> str:
        return self.name_template.format(index=index)

    def _refused(self, index: int, reason: str) -> str:
        log.warning(
            "wsp worker pod not replaced; its container restarts instead",
            extra={"code": FAILED_CODE, "worker_index": index, "reason": reason},
        )
        return f"refused:{reason}"

    def _client(self, token: str) -> httpx.Client:
        ca = self.account_dir / "ca.crt"
        verify: Any = str(ca) if ca.is_file() else True
        return httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=TIMEOUT_S,
            verify=verify,
            transport=self.transport,
        )

    def recycle(self, index: int, served_since: dt.datetime) -> str:
        try:
            token = (self.account_dir / "token").read_text(encoding="utf-8").strip()
        except OSError:
            return self._refused(index, "no_token")
        if not token:
            return self._refused(index, "no_token")
        path = f"/api/v1/namespaces/{self.namespace}/pods/{self.pod_name(index)}"
        try:
            with self._client(token) as client:
                read = client.get(path)
                if read.status_code == 404:
                    return GONE
                if read.status_code != 200:
                    return self._refused(index, f"get_{read.status_code}")
                pod = read.json()
                if (pod.get("metadata") or {}).get("deletionTimestamp"):
                    return DELETING
                created = _created(pod)
                if created is not None and created > _aware(served_since):
                    return REPLACED
                uid = (pod.get("metadata") or {}).get("uid")
                if not isinstance(uid, str) or not uid:
                    return self._refused(index, "no_uid")
                body = {
                    "kind": "DeleteOptions",
                    "apiVersion": "v1",
                    "preconditions": {"uid": uid},
                }
                gone = client.request("DELETE", path, json=body)
        except httpx.HTTPError as exc:
            return self._refused(index, f"unreachable_{type(exc).__name__}")
        except ValueError:  # an answer that is not JSON
            return self._refused(index, "bad_answer")
        if gone.status_code in (200, 202):
            return DELETED
        if gone.status_code == 404:
            return GONE
        if gone.status_code == 409:
            return REPLACED
        return self._refused(index, f"delete_{gone.status_code}")


def from_settings(name_template: str, namespace: str) -> KubernetesPodRecycler | None:
    """The production recycler, or ``None`` when ``WSP_WORKER_POD_NAME`` is empty (dev, tests,
    the local launcher): nothing is deleted and the container restart of before applies."""
    if not name_template:
        return None
    if "{index}" not in name_template:
        log.warning(
            "WSP_WORKER_POD_NAME has no {index}; no worker pod is replaced",
            extra={"code": FAILED_CODE, "reason": "no_index"},
        )
        return None
    return KubernetesPodRecycler(name_template, namespace)
